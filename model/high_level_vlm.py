import torch
import torch.nn as nn

from transformers import Gemma3ForConditionalGeneration, AutoProcessor

from utils import printable_params, batch_of_dicts_to_outer_dict, DotDict

@printable_params
class HighLevelVLM(nn.Module):
    def __init__(self, cfg, device='cuda', dtype=torch.bfloat16):
        super(HighLevelVLM, self).__init__()
        self.cfg = cfg
        self.generation_kwargs = cfg.model.vla.get("generation_kwargs", {})
        self.device = device
        self.dtype = dtype
        self.processor = AutoProcessor.from_pretrained("google/gemma-3-4b-it")
        self.processor.tokenizer.padding_side = "left"
        self.model = Gemma3ForConditionalGeneration.from_pretrained(
                "google/gemma-3-4b-it", device_map="auto", torch_dtype=dtype,
            )
        self.model.model.vision_tower.requires_grad_(False)  # Freeze vision tower
        # self.model.model.audio_tower.requires_grad_(False)  # Freeze audio tower
        # self.model = nn.Parameter(torch.tensor(0.0))  # Dummy parameter to register device and dtype

    def process_batch(self, batch, generation=False):
        B = batch["frames"].shape[0] if "frames" in batch else 1
        inputs = []
        for i in range(B):
            messages = [
                    {
                        "role": "system",
                        "content": [
                            {"type": "text", "text": f"You are controlling a robotic agent. Your task is to {batch['instruction'][i]}."}
                        ]
                    },
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": batch["frames"][i]},
                            {"type": "text", "text": "What should the robot do next?"},
                        ]
                    }]
            prepared_text = self.processor.apply_chat_template(messages,
                            tokenize=True,
                            padding=True,
                            return_dict=True,
                            return_tensors="pt",
                            add_generation_prompt=True)
            prepared_text = {k: v.squeeze(0) for k, v in prepared_text.items() if torch.is_tensor(v)}
            if not generation:
                messages.append(
                    {
                        "role": "assistant",
                        "content": [
                            {"type": "text", "text": batch["captions"][i]},
                        ]
                    }
                )
                labels = self.processor.apply_chat_template(messages,
                            tokenize=True,
                            return_dict=True,
                            return_tensors="pt",
                            add_generation_prompt=False)
                labels = {k: v.squeeze(0) for k, v in labels.items() if torch.is_tensor(v)}

                prompt_length = prepared_text["input_ids"].shape[0]
                prepared_text["input_ids"] = labels["input_ids"].clone()
                labels["input_ids"][:prompt_length] = -100
                prepared_text["attention_mask"] = labels["attention_mask"].clone()
                prepared_text["labels"] = labels["input_ids"]

                if "token_type_ids" in prepared_text:
                    prepared_text["token_type_ids"] = labels["token_type_ids"]

            prepared_text = {k: v.to(device=self.device) for k, v in prepared_text.items()}
            inputs.append(prepared_text)

        inputs = batch_of_dicts_to_outer_dict(inputs, padding_side='left')

        if "labels" in inputs:
            inputs["labels"][inputs["labels"] == self.processor.tokenizer.pad_token_id] = -100
        return inputs

    def forward(self, input_ids, attention_mask, pixel_values, labels=None, **kwargs):
        return self.model(input_ids=input_ids, attention_mask=attention_mask, pixel_values=pixel_values, labels=labels, **kwargs)

    def clean_sequences(self, sequences, inputs):
        all_input_ids = inputs["input_ids"]
        cleaned_sequences = []

        for i, sequence in enumerate(sequences):
            # Remove padding tokens first
            input_ids = all_input_ids[i]
            cleaned_input_ids = input_ids[input_ids != self.processor.tokenizer.pad_token_id]
            cleaned_sequence = sequence[sequence != self.processor.tokenizer.pad_token_id]

            # Remove prompt from cleaned sequence
            prompt_length = cleaned_input_ids.shape[0]
            cleaned_sequence = cleaned_sequence[prompt_length:]
            cleaned_sequences.append(cleaned_sequence)

        return cleaned_sequences

    def generate_and_decode(self, batch):
        inputs = self.process_batch(batch, generation=True)
        generated_ids = self.model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            pixel_values=inputs["pixel_values"],
            max_new_tokens=50,
            do_sample=self.generation_kwargs.get("do_sample", False),
            num_beams=self.generation_kwargs.get("num_beams", 1),
            temperature=self.generation_kwargs.get("temperature", 1.0),
            top_p=self.generation_kwargs.get("top_p", 1.0),
            top_k=self.generation_kwargs.get("top_k", 50),
            return_dict_in_generate=True,
            output_scores=False,
        ).sequences
        cleaned_ids = self.clean_sequences(generated_ids, inputs)
        cleaned_output = self.processor.batch_decode(cleaned_ids, skip_special_tokens=True)
        full_output = self.processor.batch_decode(generated_ids, skip_special_tokens=False)
        return DotDict({ "tokens": cleaned_output, "full_output": full_output }), cleaned_ids
