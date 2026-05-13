from collections import defaultdict
import torch
import torch.nn.functional as F


class RetrievalEvaluator:
    """Handles retrieval evaluation for contrastive models"""

    def __init__(self, max_eval_samples=500, k_values=None, eval_batch_size=32):
        self.config = {
            'max_eval_samples': max_eval_samples,
            'k_values': k_values if k_values is not None else [1, 5, 10],
            'eval_batch_size': eval_batch_size
        }
        self.reset_cache()

    def reset_cache(self):
        """Reset the retrieval cache"""
        self.cache = {
            'embeddings': {'text': [], 'vision_action': []},
            'ground_truth_pairs': [],
            'metadata': {'texts': [], 'batch_indices': []}
        }

    def extract_embeddings_from_batch(self, model, batch):
        """Extract embeddings for retrieval evaluation"""
        captions = batch["captions"]
        instructions = batch["instruction"]
        frames = batch["frames"]
        actions = batch["action"]

        # Limit batch size for efficiency
        batch_size = min(len(captions), self.config['eval_batch_size'])

        text_embeddings = []
        action_vision_embeddings = []
        texts = []

        with torch.no_grad():
            for i in range(batch_size):
                # Extract individual embeddings (modify based on your model architecture)
                _, outputs = model.model_call({
                    "captions": [captions[i]],
                    "instruction": [instructions[i]],
                    "frames": frames[i].unsqueeze(0),
                    "action": actions[i].unsqueeze(0)
                })

                # Assuming your model outputs text and vision-action embeddings
                # Modify these based on your actual model output structure
                text_embed = outputs.text_embeddings.cpu()  # Shape: [1, embed_dim]
                action_vision_embed = outputs.action_vision_embeddings.cpu()  # Shape: [1, embed_dim]

                text_embeddings.append(text_embed)
                action_vision_embeddings.append(action_vision_embed)

                # Store text for logging
                full_text = model.model.processor.apply_answer_template(
                    [instructions[i]], [captions[i]], [actions[i]]
                )
                texts.append(full_text[0])

        return {
            'text_embeddings': torch.cat(text_embeddings, dim=0) if text_embeddings else torch.empty(0),
            'action_vision_embeddings': torch.cat(action_vision_embeddings, dim=0) if action_vision_embeddings else torch.empty(0),
            'texts': texts,
            'batch_size': batch_size
        }

    def accumulate_data(self, model, batch, batch_idx):
        if len(self.cache['embeddings']['text']) >= self.config['max_eval_samples']:
            return

        embeddings_data = self.extract_embeddings_from_batch(model, batch)
        batch_size = embeddings_data['batch_size']

        if batch_size == 0:
            return

        # Get current total count BEFORE appending
        current_count = sum(emb.shape[0] for emb in self.cache['embeddings']['text'])

        # Store embeddings
        self.cache['embeddings']['text'].append(embeddings_data['text_embeddings'])
        self.cache['embeddings']['vision_action'].append(embeddings_data['action_vision_embeddings'])

        # Create ground truth pairs with correct indexing
        for i in range(batch_size):
            self.cache['ground_truth_pairs'].append((current_count + i, current_count + i))

        self.cache['metadata']['texts'].extend(embeddings_data['texts'])
        self.cache['metadata']['batch_indices'].extend([batch_idx] * batch_size)

    def compute_retrieval_metrics(self, query_embeddings, gallery_embeddings, ground_truth_pairs, k_values, apply_normalization=True):
        """Compute retrieval metrics given embeddings"""
        # Normalize embeddings
        # Not necessary if model already outputs normalized embeddings
        if apply_normalization:
            query_norm = torch.nn.functional.normalize(query_embeddings, p=2, dim=1)
            gallery_norm = torch.nn.functional.normalize(gallery_embeddings, p=2, dim=1)
        else:
            query_norm = query_embeddings.float() if query_embeddings.dtype != torch.float32 else query_embeddings
            gallery_norm = gallery_embeddings.float() if gallery_embeddings.dtype != torch.float32 else gallery_embeddings

        # Compute similarity matrix

        similarity_matrix = torch.mm(query_norm, gallery_norm.t())
        if similarity_matrix.dtype != torch.float32:
            similarity_matrix = similarity_matrix.float()

        results = {}

        # Create ground truth mapping
        gt_mapping = defaultdict(list)
        for query_idx, gallery_idx in ground_truth_pairs:
            gt_mapping[query_idx].append(gallery_idx)

        # Compute Recall@K
        for k in k_values:
            correct = 0
            total = 0

            for query_idx in range(similarity_matrix.shape[0]):
                if query_idx in gt_mapping:
                    # Get top-k gallery indices
                    top_k_indices = torch.topk(similarity_matrix[query_idx], k=k).indices.cpu().numpy()

                    # Check if any ground truth is in top-k
                    gt_indices = set(gt_mapping[query_idx])
                    if any(idx in gt_indices for idx in top_k_indices):
                        correct += 1
                    total += 1

            results[f'recall@{k}'] = correct / total if total > 0 else 0.0

        # Compute MRR
        rr_sum = 0
        total = 0

        for query_idx in range(similarity_matrix.shape[0]):
            if query_idx in gt_mapping:
                rankings = torch.argsort(similarity_matrix[query_idx], descending=True).cpu().numpy()
                gt_indices = set(gt_mapping[query_idx])

                for rank, gallery_idx in enumerate(rankings, 1):
                    if gallery_idx in gt_indices:
                        rr_sum += 1.0 / rank
                        break
                total += 1

        results['mrr'] = rr_sum / total if total > 0 else 0.0
        return results

    def evaluate_retrieval(self, logger, global_test_step, current_epoch, apply_normalization=True):
        """Perform retrieval evaluation on accumulated data"""
        if not self.cache['embeddings']['text']:
            logger.log("No data accumulated for retrieval evaluation", level="warning")
            return {}

        # Concatenate all accumulated embeddings
        text_embeds = torch.cat(self.cache['embeddings']['text'], dim=0)
        action_vision_embeds = torch.cat(self.cache['embeddings']['vision_action'], dim=0)

        print(f"Evaluating retrieval on {len(text_embeds)} samples...")

        k_values = self.config['k_values']
        ground_truth_pairs = self.cache['ground_truth_pairs']

        # Different retrieval tasks
        retrieval_tasks = {
            'text_to_vision_action': (text_embeds, action_vision_embeds),
            'vision_action_to_text': (action_vision_embeds, text_embeds)
        }

        all_results = {}

        for task_name, (query_embeds, gallery_embeds) in retrieval_tasks.items():
            print(f"Computing {task_name} retrieval...")
            results = self.compute_retrieval_metrics(
                query_embeds, gallery_embeds, ground_truth_pairs, k_values, apply_normalization=apply_normalization
            )

            # Log metrics
            for metric, value in results.items():
                metric_name = f"retrieval/{task_name}_{metric}"
                all_results[metric_name] = value
                logger.log_metric(
                    value, metric_name,
                    step=global_test_step,
                    epoch=current_epoch,
                    stage="test"
                )

        # Log summary
        for metric, value in all_results.items():
            logger.log_metric(value, metric, step=global_test_step, epoch=current_epoch, stage="test")

        return all_results


class MockLogger:
    """Mock logger for testing"""
    def log(self, message, level="info"):
        print(f"[{level.upper()}] {message}")

    def log_metric(self, value, name, step=0, epoch=0, stage="test"):
        print(f"Metric: {name} = {value:.4f} (step={step}, epoch={epoch}, stage={stage})")


def test_retrieval_evaluator():
    """Test the RetrievalEvaluator with perfect embeddings"""
    print("="*60)
    print("TESTING RETRIEVAL EVALUATOR WITH PERFECT EMBEDDINGS")
    print("="*60)

    # Import your RetrievalEvaluator class (assuming it's in the same file)
    evaluator = RetrievalEvaluator(max_eval_samples=100, k_values=[1, 5, 10])

    # Create perfect embeddings - text and vision should match exactly
    embed_dim = 512
    num_samples = 50

    print(f"Creating {num_samples} perfect embedding pairs with dimension {embed_dim}")

    # Generate random embeddings
    perfect_embeddings = torch.randn(num_samples, embed_dim)

    # Simulate multiple batches to test accumulation
    batch_sizes = [16, 16, 18]  # Total = 50
    start_idx = 0

    for batch_idx, batch_size in enumerate(batch_sizes):
        print(f"\nProcessing batch {batch_idx + 1} with {batch_size} samples")

        # Get embeddings for this batch
        batch_text_embeds = perfect_embeddings[start_idx:start_idx + batch_size]
        batch_vision_embeds = perfect_embeddings[start_idx:start_idx + batch_size].clone()  # Perfect copies

        # Manually populate the cache (simulating what accumulate_data would do)
        current_count = sum(emb.shape[0] for emb in evaluator.cache['embeddings']['text'])
        print(f"Current total count before adding batch: {current_count}")

        # Add to cache
        evaluator.cache['embeddings']['text'].append(batch_text_embeds)
        evaluator.cache['embeddings']['vision_action'].append(batch_vision_embeds)

        # Create ground truth pairs
        for i in range(batch_size):
            evaluator.cache['ground_truth_pairs'].append((current_count + i, current_count + i))

        # Add dummy metadata
        evaluator.cache['metadata']['texts'].extend([f"sample_{start_idx + i}" for i in range(batch_size)])
        evaluator.cache['metadata']['batch_indices'].extend([batch_idx] * batch_size)

        start_idx += batch_size

    print(f"\nFinal cache state:")
    total_text = sum(emb.shape[0] for emb in evaluator.cache['embeddings']['text'])
    total_vision = sum(emb.shape[0] for emb in evaluator.cache['embeddings']['vision_action'])
    print(f"Total text embeddings: {total_text}")
    print(f"Total vision embeddings: {total_vision}")
    print(f"Total ground truth pairs: {len(evaluator.cache['ground_truth_pairs'])}")
    print(f"First 5 GT pairs: {evaluator.cache['ground_truth_pairs'][:5]}")
    print(f"Last 5 GT pairs: {evaluator.cache['ground_truth_pairs'][-5:]}")

    # Test the evaluation
    print("\n" + "="*60)
    print("RUNNING EVALUATION")
    print("="*60)

    mock_logger = MockLogger()
    results = evaluator.evaluate_retrieval(mock_logger, global_test_step=0, current_epoch=0)

    print("\n" + "="*40)
    print("FINAL RESULTS:")
    print("="*40)
    for metric, value in results.items():
        print(f"{metric}: {value:.4f}")

    # Verify results
    print("\n" + "="*40)
    print("VERIFICATION:")
    print("="*40)

    expected_perfect_score = 1.0
    recall_at_1_text_to_vision = results.get('retrieval/text_to_vision_action_recall@1', 0)
    recall_at_1_vision_to_text = results.get('retrieval/vision_action_to_text_recall@1', 0)

    if recall_at_1_text_to_vision == expected_perfect_score:
        print("✅ Text-to-Vision Recall@1 is perfect (1.0)")
    else:
        print(f"❌ Text-to-Vision Recall@1 is {recall_at_1_text_to_vision:.4f}, expected {expected_perfect_score}")

    if recall_at_1_vision_to_text == expected_perfect_score:
        print("✅ Vision-to-Text Recall@1 is perfect (1.0)")
    else:
        print(f"❌ Vision-to-Text Recall@1 is {recall_at_1_vision_to_text:.4f}, expected {expected_perfect_score}")

    # Test with some noise to see if evaluation can distinguish
    print("\n" + "="*60)
    print("TESTING WITH NOISY EMBEDDINGS")
    print("="*60)

    evaluator.reset_cache()

    # Create slightly noisy embeddings
    base_embeddings = torch.randn(30, embed_dim)
    text_embeddings = base_embeddings
    vision_embeddings = base_embeddings + 0.1 * torch.randn_like(base_embeddings)  # Add small noise

    # Add to cache in one go
    evaluator.cache['embeddings']['text'].append(text_embeddings)
    evaluator.cache['embeddings']['vision_action'].append(vision_embeddings)

    for i in range(30):
        evaluator.cache['ground_truth_pairs'].append((i, i))

    evaluator.cache['metadata']['texts'].extend([f"noisy_sample_{i}" for i in range(30)])
    evaluator.cache['metadata']['batch_indices'].extend([0] * 30)

    noisy_results = evaluator.evaluate_retrieval(mock_logger, global_test_step=1, current_epoch=0)

    print(f"\nNoisy results (should be lower than perfect):")
    for metric, value in noisy_results.items():
        if 'recall@1' in metric:
            print(f"{metric}: {value:.4f}")

    return results

def debug_similarity_computation():
    """Debug the similarity computation to understand why noisy embeddings give perfect scores"""

    print("="*60)
    print("DEBUGGING SIMILARITY COMPUTATION")
    print("="*60)

    embed_dim = 512
    num_samples = 10  # Smaller for easier debugging

    # Create base embeddings
    base_embeddings = torch.randn(num_samples, embed_dim)

    # Test 1: Perfect embeddings
    print("\n1. PERFECT EMBEDDINGS TEST:")
    text_embeds = base_embeddings.clone()
    vision_embeds = base_embeddings.clone()

    # Normalize
    text_norm = F.normalize(text_embeds, p=2, dim=1)
    vision_norm = F.normalize(vision_embeds, p=2, dim=1)

    # Compute similarity
    similarity = torch.mm(text_norm, vision_norm.t())

    print(f"Similarity matrix shape: {similarity.shape}")
    print(f"Diagonal (should be 1.0): {similarity.diag()}")
    print(f"Off-diagonal max: {similarity.masked_fill(torch.eye(num_samples).bool(), -2).max():.4f}")
    print(f"Off-diagonal min: {similarity.masked_fill(torch.eye(num_samples).bool(), 2).min():.4f}")

    # Test 2: Noisy embeddings with small noise
    print("\n2. SMALL NOISE TEST (0.1 std):")
    text_embeds = base_embeddings.clone()
    vision_embeds = base_embeddings + 0.1 * torch.randn_like(base_embeddings)

    # Normalize
    text_norm = F.normalize(text_embeds, p=2, dim=1)
    vision_norm = F.normalize(vision_embeds, p=2, dim=1)

    # Compute similarity
    similarity = torch.mm(text_norm, vision_norm.t())

    print(f"Diagonal (should be < 1.0): {similarity.diag()}")
    print(f"Off-diagonal max: {similarity.masked_fill(torch.eye(num_samples).bool(), -2).max():.4f}")

    # Check if diagonal is still highest
    for i in range(num_samples):
        row = similarity[i]
        diagonal_val = row[i]
        max_off_diagonal = row.masked_fill(torch.zeros(num_samples).bool().scatter_(0, torch.tensor([i]), True), -2).max()
        print(f"Sample {i}: diagonal={diagonal_val:.4f}, max_off_diagonal={max_off_diagonal:.4f}, correct={diagonal_val > max_off_diagonal}")

    # Test 3: Much larger noise
    print("\n3. LARGE NOISE TEST (1.0 std):")
    text_embeds = base_embeddings.clone()
    vision_embeds = base_embeddings + 1.0 * torch.randn_like(base_embeddings)

    # Normalize
    text_norm = F.normalize(text_embeds, p=2, dim=1)
    vision_norm = F.normalize(vision_embeds, p=2, dim=1)

    # Compute similarity
    similarity = torch.mm(text_norm, vision_norm.t())

    print(f"Diagonal: {similarity.diag()}")
    print(f"Off-diagonal max: {similarity.masked_fill(torch.eye(num_samples).bool(), -2).max():.4f}")

    # Check if diagonal is still highest
    correct_retrievals = 0
    for i in range(num_samples):
        row = similarity[i]
        diagonal_val = row[i]
        max_off_diagonal = row.masked_fill(torch.zeros(num_samples).bool().scatter_(0, torch.tensor([i]), True), -2).max()
        is_correct = diagonal_val > max_off_diagonal
        if is_correct:
            correct_retrievals += 1
        print(f"Sample {i}: diagonal={diagonal_val:.4f}, max_off_diagonal={max_off_diagonal:.4f}, correct={is_correct}")

    print(f"\nRecall@1 with large noise: {correct_retrievals/num_samples:.4f}")

    # Test 4: Completely random embeddings
    print("\n4. RANDOM EMBEDDINGS TEST:")
    text_embeds = torch.randn(num_samples, embed_dim)
    vision_embeds = torch.randn(num_samples, embed_dim)

    # Normalize
    text_norm = F.normalize(text_embeds, p=2, dim=1)
    vision_norm = F.normalize(vision_embeds, p=2, dim=1)

    # Compute similarity
    similarity = torch.mm(text_norm, vision_norm.t())

    print(f"Diagonal: {similarity.diag()}")
    print(f"Off-diagonal max: {similarity.masked_fill(torch.eye(num_samples).bool(), -2).max():.4f}")

    # Check if diagonal is still highest
    correct_retrievals = 0
    for i in range(num_samples):
        row = similarity[i]
        top_k_indices = torch.topk(row, k=1).indices
        if i in top_k_indices:
            correct_retrievals += 1

    print(f"Recall@1 with random embeddings: {correct_retrievals/num_samples:.4f}")
    print(f"Expected random recall@1: {1/num_samples:.4f}")


def test_your_evaluator_with_random():
    """Test your actual evaluator with completely random embeddings"""
    print("\n" + "="*60)
    print("TESTING YOUR EVALUATOR WITH RANDOM EMBEDDINGS")
    print("="*60)

    from collections import defaultdict

    # Your compute_retrieval_metrics function (simplified)
    def compute_retrieval_metrics(query_embeddings, gallery_embeddings, ground_truth_pairs, k_values):
        # Normalize embeddings
        query_norm = torch.nn.functional.normalize(query_embeddings, p=2, dim=1)
        gallery_norm = torch.nn.functional.normalize(gallery_embeddings, p=2, dim=1)

        # Compute similarity matrix
        similarity_matrix = torch.mm(query_norm, gallery_norm.t())
        if similarity_matrix.dtype != torch.float32:
            similarity_matrix = similarity_matrix.float()

        results = {}

        # Create ground truth mapping
        gt_mapping = defaultdict(list)
        for query_idx, gallery_idx in ground_truth_pairs:
            gt_mapping[query_idx].append(gallery_idx)

        # Compute Recall@K
        for k in k_values:
            correct = 0
            total = 0

            for query_idx in range(similarity_matrix.shape[0]):
                if query_idx in gt_mapping:
                    # Get top-k gallery indices
                    top_k_indices = torch.topk(similarity_matrix[query_idx], k=k).indices.cpu().numpy()

                    # Check if any ground truth is in top-k
                    gt_indices = set(gt_mapping[query_idx])
                    if any(idx in gt_indices for idx in top_k_indices):
                        correct += 1
                    total += 1

            results[f'recall@{k}'] = correct / total if total > 0 else 0.0

        return results

    # Test with random embeddings
    num_samples = 100
    embed_dim = 512

    query_embeddings = torch.randn(num_samples, embed_dim)
    gallery_embeddings = torch.randn(num_samples, embed_dim)  # Completely random
    ground_truth_pairs = [(i, i) for i in range(num_samples)]

    results = compute_retrieval_metrics(query_embeddings, gallery_embeddings, ground_truth_pairs, [1, 5, 10])

    print("Results with random embeddings (should be ~0.01, 0.05, 0.10):")
    for metric, value in results.items():
        expected = float(metric.split('@')[1]) / num_samples
        print(f"{metric}: {value:.4f} (expected ~{expected:.4f})")

if __name__ == "__main__":
    # Run the test when file is executed directly
    test_results = test_retrieval_evaluator()
    debug_similarity_computation()
    test_your_evaluator_with_random()
