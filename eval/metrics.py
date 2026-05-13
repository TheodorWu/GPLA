import torch
import torch.nn.functional as F
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.translate.meteor_score import single_meteor_score
from nltk.tokenize import regexp_tokenize
from nltk import accuracy
import nltk
from rouge_score import rouge_scorer
from rouge_score.scoring import BootstrapAggregator
from einops import rearrange
from tqdm import tqdm
from pathlib import Path

from aac_metrics.functional.bert_score_mrefs import bert_score
from aac_metrics.functional import cider_d
from aac_metrics.utils.tokenization import preprocess_mono_sents, preprocess_mult_sents
import wandb

class Evaluator():
    def __init__(self) -> None:
        nltk.download("wordnet")
        self.clip_model = None
        run_name = wandb.run.name if wandb.run else "no_run"

        project_dir = Path(__file__).resolve().parent.parent
        output_dir = project_dir / "eval_outputs"
        output_dir.mkdir(parents=True, exist_ok=True)
        run_name = output_dir / run_name
        # Closing handled in __del__
        self.gen_file = open(f"{run_name}_generated_captions.txt", "a", encoding="utf-8") # pylint: disable=consider-using-with
        self.tgt_file = open(f"{run_name}_target_captions.txt", "a", encoding="utf-8") # pylint: disable=consider-using-with

    def safe_eval(self, metric_func, *args, output_dict, key=None, is_dict=False):
        """
        Safely evaluates a metric function and updates output_dict.
        If is_dict is True, updates output_dict with the returned dict.
        Otherwise, sets output_dict[key] = result.
        """
        try:
            result = metric_func(*args)
            if is_dict and isinstance(result, dict):
                output_dict.update(result)
            elif key is not None:
                output_dict[key] = result
        except Exception as e:
            print(f"Metric '{key or metric_func.__name__}' failed: {e}")

    def evaluate_all(self, generated, target):
        output_dict = {}

        # Trajectory based metrics
        # self.safe_eval(self.action_token_accuracy,
        #         generated["action_ids"], target["action_ids"],
        #         output_dict=output_dict, key="action_token_accuracy")

        # Prepare actions for trajectory metrics
        try:
            if isinstance(target["action"], list):
                target["action"] = torch.stack(target["action"], dim=0)
            target_action = target["action"].to(
                device=generated["actions"].device,
                dtype=generated["actions"].dtype
            )
            target_action = rearrange(target_action, "b h d -> b (h d)")
            generated_actions = generated["actions"]
            # Get target length
            target_len = target_action.shape[1]  # (h d)
            gen_len = generated_actions.shape[1]

            if gen_len < target_len:
                # Pad with zeros
                pad_size = target_len - gen_len
                generated_actions = F.pad(generated_actions, (0, pad_size), value=0)
            elif gen_len > target_len:
                # Truncate
                generated_actions = generated_actions[:, :target_len]
        except Exception as e:
            print(f"Action preparation failed: {e}")
            target_action = None
            generated_actions = None

        # Only compute these if preparation succeeded
        if target_action is not None and generated_actions is not None:
            self.safe_eval(self.mse, generated_actions, target_action,
                    output_dict=output_dict, key="mse")
            self.safe_eval(self.cos_sim, generated_actions, target_action,
                    output_dict=output_dict, key="cos_sim")
            self.safe_eval(self.mae, generated_actions, target_action,
                    output_dict=output_dict, key="mae")

        # Language based metrics
        output_dict = self.evaluate_language_only(
            generated, target, output_dict=output_dict
        )

        return output_dict

    def evaluate_vla_all(self, generated, target):
        output_dict = {}

        # Language based metrics
        output_dict = self.evaluate_language_only(
            generated, target, output_dict=output_dict
        )

        # Vision-Language based metrics
        output_dict = self.evaluate_vision_language(
            generated, target, output_dict=output_dict
        )

        # Action based metrics
        output_dict = self.evaluate_vla_actions(
            generated, target, output_dict=output_dict
        )

        return output_dict

    def evaluate_vision_language(self, generated, target, output_dict=None):
        output_dict = output_dict or {}
        # CLIP Score
        try:
            images = target["frames"]
            texts = generated["tokens"]
            device = images.device
            self.safe_eval(self.clip_score, images, texts, device,
                    output_dict=output_dict, key="clip_score")
        except Exception as e:
            print(f"CLIP score evaluation failed: {e}")

        return output_dict

    def evaluate_vla_actions(self, generated, target, output_dict=None):
        output_dict = output_dict or {}

        # Prepare actions for trajectory metrics
        try:
            generated_actions = generated.actions
            if isinstance(target["action"], list):
                target["action"] = torch.stack(target["action"], dim=0)
            target_action = target["action"].to(
                device=generated_actions.device,
                dtype=generated_actions.dtype
            )
            target_action = rearrange(target_action, "b h d -> b (h d)")
            generated_actions = rearrange(generated_actions, "b h d -> b (h d)")
            # Get target length
            target_len = target_action.shape[1]  # (h d)
            gen_len = generated_actions.shape[1]

            if gen_len < target_len:
                # Pad with zeros
                pad_size = target_len - gen_len
                generated_actions = F.pad(generated_actions, (0, pad_size), value=0)
            elif gen_len > target_len:
                # Truncate
                generated_actions = generated_actions[:, :target_len]
        except Exception as e:
            print(f"Action preparation failed: {e}")
            target_action = None
            generated_actions = None

        # Only compute these if preparation succeeded
        if target_action is not None and generated_actions is not None:
            # print(f"Generated actions shape: {generated_actions.shape}, Target actions shape: {target_action.shape}")
            self.safe_eval(self.mse, generated_actions, target_action,
                    output_dict=output_dict, key="mse")
            self.safe_eval(self.cos_sim, generated_actions, target_action,
                    output_dict=output_dict, key="cos_sim")
            self.safe_eval(self.mae, generated_actions, target_action,
                    output_dict=output_dict, key="mae")

        return output_dict

    def evaluate_language_only(self, generated, target, output_dict=None):
        output_dict = output_dict or {}

        # Language based metrics
        try:
            tokens = generated["tokens"]
            captions = target["captions"]
        except Exception as e:
            print(f"Token/caption preparation failed: {e}")
            tokens = None
            captions = None

        if tokens is not None and captions is not None:
            try:
                # Append output to files
                for gen, tgt in zip(tokens, captions):
                    self.gen_file.write(gen + "\n")
                    self.tgt_file.write(tgt + "\n")

                # Flush to ensure data is written immediately
                self.gen_file.flush()
                self.tgt_file.flush()
            except Exception as e:
                print(f"Failed to write captions to file: {e}")

            self.safe_eval(self.bleu, tokens, captions,
                    output_dict=output_dict, key="bleu_score")
            self.safe_eval(self.rouge, tokens, captions,
                    output_dict=output_dict, is_dict=True)
            self.safe_eval(self.meteor, tokens, captions,
                    output_dict=output_dict, key="meteor_score")
            self.safe_eval(self.bertscore, tokens, captions,
                    output_dict=output_dict, key="bertscore")
            self.safe_eval(self.cider, tokens, captions,
                    output_dict=output_dict, key="cider")

        return output_dict

    def confidence(self, logits):
        max_probs = logits.softmax(dim=-1).max(dim=-1)[0]  # (batch_size, seq_len)
        return max_probs.mean()  # Single scalar across entire batch

    def action_token_accuracy(self, generated, target):
        score = 0.0
        for g, t in zip(generated, target):
            score += accuracy(g, t)
        return score/len(generated)

    def bleu(self, generated, target):
        score = 0.0
        smoothing = SmoothingFunction()
        for g, t in zip(generated, target):
            g = regexp_tokenize(g, pattern=r"\w+")
            t = [regexp_tokenize(t, pattern=r"\w+")]
            score += sentence_bleu(t, g, weights=(0.25, 0.25, 0.25, 0.25),
                                   smoothing_function=smoothing.method1)
        return score/len(generated)

    def rouge(self, generated, target):
        scorer = rouge_scorer.RougeScorer(["rouge1", "rougeL"], use_stemmer=True)
        aggregator = BootstrapAggregator()
        for g, t in zip(generated, target):
            aggregator.add_scores(scorer.score(t, g))

        total_scores = aggregator.aggregate()

        result = {}
        for interval, values in total_scores["rougeL"]._asdict().items():
            for metric, value in values._asdict().items():
                result[f"ROUGE L {metric} {interval}"] = value

        for interval, values in total_scores["rouge1"]._asdict().items():
            for metric, value in values._asdict().items():
                result[f"ROUGE 1 {metric} {interval}"] = value

        return result

    def meteor(self, generated, target):
        score = 0.0
        for g, t in zip(generated, target):
            g = regexp_tokenize(g, pattern=r"\w+")
            t = regexp_tokenize(t, pattern=r"\w+")
            score += single_meteor_score(t, g)

        return score/len(generated)

    def bertscore(self, generated, target):
        # bert_score expects lists of strings for both candidates and references
        # It returns a tuple: (P, R, F1), each as a tensor of scores
        f1 = bert_score(generated, target, lang='en', rescale_with_baseline=True, model_name_or_path="microsoft/deberta-xlarge-mnli")["f1"]
        # Return the average F1 score
        return f1.mean().item()

    def cider(self, generated, target):
        # Each generated sentence can have multiple references
        # For single reference per generated sentence, wrap each target in a list
        references = [[t] for t in target]
        # Preprocess sentences as required by aac-metrics
        candidates = preprocess_mono_sents(generated)
        mult_references = preprocess_mult_sents(references)
        # Compute CIDEr-D scores
        _, sent_scores = cider_d(candidates, mult_references)
        # sent_scores['cider_d'] is a tensor of per-sentence scores
        return sent_scores['cider_d'].mean().item()

    def cos_sim(self, generated, target):
        cos_sim = F.cosine_similarity(generated, target) # pylint: disable=not-callable
        return torch.mean(cos_sim)

    def mse(self, generated, target):
        return F.mse_loss(generated, target)

    def mae(self, generated, target):
        return F.l1_loss(generated, target)

    def clip_score(self, images, texts, device):
        if not self.clip_model:
            print("Loading CLIP model for the first time...")
            from transformers import CLIPModel, CLIPProcessor
            self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32", dtype=torch.bfloat16, attn_implementation="sdpa")
            self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

        with torch.inference_mode():
            inputs = self.clip_processor(text=texts, images=images, return_tensors="pt", padding=True).to(device)
            outputs = self.clip_model(**inputs)
            logits_per_image = outputs.logits_per_image  # this is the image-text similarity score
            return torch.mean(logits_per_image)

    def perplexity(self, model, encodings, device):
        # Not yet tested. Code from Huggingface https://huggingface.co/docs/transformers/en/perplexity
        max_length = model.config.n_positions
        stride = 512
        seq_len = encodings.input_ids.size(1)

        nll_sum = 0.0
        n_tokens = 0
        prev_end_loc = 0
        for begin_loc in tqdm(range(0, seq_len, stride)):
            end_loc = min(begin_loc + max_length, seq_len)
            trg_len = end_loc - prev_end_loc  # may be different from stride on last loop
            input_ids = encodings.input_ids[:, begin_loc:end_loc].to(device)
            target_ids = input_ids.clone()
            target_ids[:, :-trg_len] = -100

            with torch.no_grad():
                outputs = model(input_ids, labels=target_ids) # TODO: add images

                # loss is calculated using CrossEntropyLoss which averages over valid labels
                # N.B. the model only calculates loss over trg_len - 1 labels, because it internally shifts the labels
                # to the left by 1.
                neg_log_likelihood = outputs.loss

            # Accumulate the total negative log-likelihood and the total number of tokens
            num_valid_tokens = (target_ids != -100).sum().item()  # number of valid tokens in target_ids
            batch_size = target_ids.size(0)
            num_loss_tokens = num_valid_tokens - batch_size  # subtract batch_size due to internal label shift
            nll_sum += neg_log_likelihood * num_loss_tokens
            n_tokens += num_loss_tokens

            prev_end_loc = end_loc
            if end_loc == seq_len:
                break

        avg_nll = nll_sum / n_tokens  # average negative log-likelihood per token
        ppl = torch.exp(avg_nll)
        return ppl

    def close_files(self):
        """Close caption files safely."""
        if hasattr(self, 'gen_file'):
            self.gen_file.close()
        if hasattr(self, 'tgt_file'):
            self.tgt_file.close()

    def __del__(self):
        """Ensure files are closed when object is destroyed."""
        self.close_files()
