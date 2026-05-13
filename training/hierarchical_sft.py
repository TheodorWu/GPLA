from training.sft import SFT

class HierarchicalSFT(SFT):
    def __init__(self, mode, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.mode = mode

    def model_call(self, batch, **kwargs):
        model_inputs = self.model.process_batch(batch)
        # debug_batch_data(model_inputs)  # Debugging function to check input data
        outputs = self.model(**model_inputs, mode=self.mode, **kwargs)

        return model_inputs, outputs

    def inference_step(self, batch, stage):
        if stage=="val":
            step = self.global_val_step
        elif stage=="test":
            step = self.global_test_step
        else:
            step=-1

        _, outputs = self.model_call(batch)

        self.logger.log_metric(outputs.loss, "Loss", step=step, epoch=self.current_epoch, stage=stage)

        batch_without_captions = {k: v for k, v in batch.items() if k != "captions"}
        generation_inputs = self.model.process_batch(batch_without_captions)

        if self.mode == "full":
            decoded_outputs = self.model.generate_action_and_language(**generation_inputs)
            metrics = self.evaluator.evaluate_vla_all(decoded_outputs, batch)
        elif self.mode == "backbone":
            decoded_outputs = self.model.generate_language(**generation_inputs)
            metrics = self.evaluator.evaluate_language_only(decoded_outputs, batch)
        else:
            raise ValueError(f"Invalid mode: {self.mode}")

        for k, v in metrics.items():
            self.logger.log_metric(
                value=v, name=k, step=step, epoch=self.current_epoch, stage=stage)

        self.inference_log_table(batch, decoded_outputs, step, stage)
