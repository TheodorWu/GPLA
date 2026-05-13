from training.hierarchical_sft import HierarchicalSFT
from utils import test_gpu_availability, get_dtype, DIR_PATH
from data.language_table.preference_dataset import load_language_table_preferences
from data.language_table.dataset import load_language_table
from training.preference import PreferenceGenerator
from training.gpla import GPLA
from training.optimizer import get_optimizer, get_scheduler
from training.sft import SFT
from training.contrastive import Contrastive
from model.action_clip import ActionCLIP, VisionActionGroundedCLIP
import torch
from torch.nn.parameter import UninitializedParameter, Parameter
from pathlib import Path
from omegaconf import OmegaConf
import wandb
from coolname import generate_slug

class SingleBatchDataLoader:
    """A DataLoader wrapper that stores the first batch and returns it repeatedly."""

    def __init__(self, dataloader):
        self.dataloader = dataloader
        self.batch = None
        self._load_first_batch()

    def _load_first_batch(self):
        """Load and store the first batch from the original dataloader."""
        self.batch = next(iter(self.dataloader))

    def __iter__(self):
        """Return an iterator that yields the same batch repeatedly."""
        return self

    def __next__(self):
        """Return the stored batch."""
        if self.batch is None:
            raise StopIteration
        return self.batch

    def __len__(self):
        """Return the length of the original dataloader."""
        return len(self.dataloader)

class Initializer():
    def __init__(self, cfg) -> None:
        self.cfg = cfg
        self.device = test_gpu_availability(cfg=self.cfg)
        self.dtype = get_dtype(cfg.model.get("dtype"))

        self.checkpoint_dir = f"{DIR_PATH}/checkpoints"
        Path(self.checkpoint_dir).mkdir(exist_ok=True, parents=True)

        print(
            f"Effective batch size: {self.cfg.training.batch_size * self.cfg.training.optimizer_step_freq}")

    def init_wandb(self):
        name = f"{self.cfg.training.strategy}-{generate_slug(2)}-h{self.cfg.environment.kwargs.horizon}-e{self.cfg.training.epochs}"

        if self.cfg.training.strategy.lower() == "sft":
            name = self.cfg.model.vla.id + "-" + name

        wandb.init(name=name,
                   entity=self.cfg.wandb.entity,
                   project=self.cfg.wandb.project,
                   config=OmegaConf.to_container(
                       self.cfg, resolve=True, throw_on_missing=True)
                   )
        print(f"Starting run: {wandb.run.name}")

    def init_language_table(self, subset):
        return load_language_table(mode=subset,
                                   batch_size=self.cfg.training.batch_size,
                                   enforce_caption_uniqueness=self.cfg.training.get(
                                       "enforce_caption_uniqueness", False),
                                   **self.cfg.environment.kwargs)

    def init_language_table_preferences(self, subset):
        root = f"{DIR_PATH}/data/language_table/preference/{self.cfg.environment.task}/init_h_{self.cfg.environment.horizon}/"
        return load_language_table_preferences(mode=subset,
                                               root=root,
                                               **self.cfg.environment.kwargs)

    def init_datasets(self):
        target_datasets = self.cfg.environment.name

        if target_datasets == "language_table":
            # if self.cfg.training.strategy.lower() == "GPLA":
            #     train_ds = self.init_language_table_preferences("train")
            #     val_ds = self.init_language_table_preferences("validation")
            #     test_ds = self.init_language_table_preferences("test")
            # else:
            train_ds = self.init_language_table("train")
            val_ds = self.init_language_table("validation")
            test_ds = self.init_language_table("test")
        else:
            raise ValueError(f"Unknown dataset: {target_datasets}")

        if self.cfg.training.get("single_batch_debug", False):
            print("WARNING: Single Batch Loading Activated. Trying to overfit the same batch.")
            train_ds = SingleBatchDataLoader(train_ds)
            val_ds = train_ds
            test_ds = train_ds

        return train_ds, val_ds, test_ds

    def init_model(self):
        if self.cfg.model.vla.get("id") == "hierarchical":
            from model.full_hierarchical import FullHierarchicalModel
            from model.high_level_vlm import HighLevelVLM
            from model.vla import VLA

            vlm = HighLevelVLM(cfg=self.cfg,
                                 device=self.device,
                                 dtype=self.dtype)
            if high_level_checkpoint := self.cfg.model.vla.get("high_level_checkpoint", False):
                vlm.load_state_dict(torch.load(
                    f"{self.checkpoint_dir}/{high_level_checkpoint}", weights_only=True))
                print(f"Loaded High Level VLM checkpoint: {high_level_checkpoint}")

            vla = VLA(cfg=self.cfg,
                        device=self.device,
                        dtype=self.dtype)
            if vla_checkpoint := self.cfg.model.vla.get("vla_checkpoint", False):
                vla.load_state_dict(torch.load(
                    f"{self.checkpoint_dir}/{vla_checkpoint}", weights_only=True))
                print(f"Loaded VLA checkpoint: {vla_checkpoint}")

            model = FullHierarchicalModel(
                high_level_vlm=vlm,
                vla=vla,
                cfg=self.cfg,
                device=self.device,
                dtype=self.dtype)
        elif self.cfg.model.vla.get("id") == "high_level_vlm":
            from model.high_level_vlm import HighLevelVLM
            model = HighLevelVLM(cfg=self.cfg,
                                 device=self.device,
                                 dtype=self.dtype)

        elif self.cfg.model.vla.get("id") == "vla":
            from model.vla import VLA
            model = VLA(cfg=self.cfg,
                        device=self.device,
                        dtype=self.dtype)
        else:
            raise ValueError(f"Unknown model: {self.cfg.model.vla.id}")

        if self.cfg.model.vla.get("checkpoint"):
            model.load_state_dict(torch.load(
                f"{self.checkpoint_dir}/{self.cfg.model.vla.checkpoint}", weights_only=True))
        model.print_trainable_parameters()
        return model

    def init_clip_model(self):
        if self.cfg.model.clip.get("id", "action_clip").lower() == "action_clip":
            model = ActionCLIP(cfg=self.cfg,
                               device=self.device,
                               dtype=self.dtype)
        elif self.cfg.model.clip.get("id", "action_clip").lower() == "vision_action_grounded_clip":
            model = VisionActionGroundedCLIP(
                cfg=self.cfg,
                device=self.device,
                dtype=self.dtype)
        else:
            raise ValueError(f"Unknown model: {self.cfg.model.clip.id}")

        if self.cfg.model.clip.get("checkpoint"):
            with torch.serialization.safe_globals([UninitializedParameter]):
                model.load_state_dict(torch.load(
                    f"{self.checkpoint_dir}/{self.cfg.model.clip.checkpoint}", weights_only=True))
        model.print_trainable_parameters()
        return model

    def init_strategy(self):
        if self.cfg.training.strategy.lower() == "sft":
            return self.init_sft()
        elif self.cfg.training.strategy.lower() == "GPLA":
            return self.init_GPLA()
        elif self.cfg.training.strategy.lower() == "contrastive":
            return self.init_contrastive()
        elif self.cfg.training.strategy.lower() == "hierarchical_sft":
            return self.init_hierarchical_sft()
        else:
            raise ValueError(
                f"Unknown strategy: {self.cfg.training.strategy}. Should be one of ['sft', 'GPLA', 'contrastive', 'hierarchical_sft']")

    def init_optimizer_and_scheduler(self, model):
        optimizer = get_optimizer(self.cfg.training.optimizer.name,
                                  model_params=model.parameters(),
                                  optimizer_params=self.cfg.training.optimizer.kwargs)

        if self.cfg.training.scheduler.active:
            print(f"Using scheduler: {self.cfg.training.scheduler.name}")
            scheduler = get_scheduler(self.cfg.training.scheduler.name,
                                      optimizer=optimizer,
                                      scheduler_params=self.cfg.training.scheduler.kwargs)
        else:
            scheduler = None

        if self.cfg.training.get("optimizer_checkpoint"):
            optimizer.load_state_dict(torch.load(
                f"{self.checkpoint_dir}/{self.cfg.training.optimizer_checkpoint}"))
            print(
                f"Loaded optimizer state from checkpoint: {self.cfg.training.optimizer_checkpoint}")

        if self.cfg.training.get("scheduler_checkpoint") and scheduler is not None:
            scheduler.load_state_dict(torch.load(
                f"{self.checkpoint_dir}/{self.cfg.training.scheduler_checkpoint}"))
            print(
                f"Loaded scheduler state from checkpoint: {self.cfg.training.scheduler_checkpoint}")
        return optimizer, scheduler

    def init_sft(self):
        model = self.init_model()
        optimizer, scheduler = self.init_optimizer_and_scheduler(model)

        train_ds, val_ds, test_ds = self.init_datasets()
        strategy = SFT(
            cfg=self.cfg,
            model=model,
            optimizer=optimizer,
            train_dataset=train_ds,
            val_dataset=val_ds,
            test_dataset=test_ds,
            checkpoint_dir=self.checkpoint_dir,
            scheduler=scheduler
        )
        return strategy

    def init_contrastive(self):
        model = self.init_clip_model()

        if self.cfg.training.optimizer.get("lr_by_group", {}) != {}:

            def check_cross_attn_name(name):
                for attn_name in ["vision_action_crossattention", "attention_based_fusion"]:
                    if attn_name in name:
                        return True
                return False

            other_param_names = [name for name, _ in model.named_parameters()]
            lr_by_group_keys = list(self.cfg.training.optimizer.lr_by_group.keys())
            if "others" in lr_by_group_keys:
                lr_by_group_keys.remove("others")

            def check_key_in_name(name):
                for key in lr_by_group_keys:
                    if key in name:
                        return True
                return False

            other_param_names = [ name for name in other_param_names if not check_key_in_name(name)]
            model_params = []
            for key in lr_by_group_keys:
                if hasattr(model, key):
                    layer = getattr(model, key)
                    try:
                        params = next(layer.parameters())
                    except AttributeError:
                        print(f"Warning: No parameters found for layer '{key}'. Skipping.")
                        continue
                    if not params.requires_grad:
                        continue
                    model_params.append(
                        {"params": layer.parameters(), "lr": self.cfg.training.optimizer.lr_by_group[key]},
                    )

            cross_attention_names = [name for name, _ in model.named_parameters() if check_cross_attn_name(name)]
            cross_attention_names = [name.split(".")[0] for name in cross_attention_names] # group on first level
            cross_attention_names = list(set(cross_attention_names))  # remove duplicates
            cross_attention_params = [{"params": getattr(model, name).parameters(), "lr": self.cfg.training.optimizer.lr_by_group.attention}
                                      for name in cross_attention_names]
            model_params.extend(cross_attention_params)

            other_param_names = [name for name in other_param_names if not check_cross_attn_name(name)]
            other_param_names = [name.split(".")[0] for name in other_param_names] # group on first level
            other_param_names = list(set(other_param_names))  # remove duplicates
            other_params = [ getattr(model, name) if isinstance(getattr(model, name), Parameter) else getattr(model, name).parameters() for name in other_param_names ]
            other_params = [
                {"params": params, "lr": self.cfg.training.optimizer.lr_by_group.others}
                for params in other_params
            ]
            model_params.extend(other_params)
        else:
            model_params = model.parameters()

        optimizer = get_optimizer(self.cfg.training.optimizer.name,
                                  model_params=model_params,
                                  optimizer_params=self.cfg.training.optimizer.kwargs)
        if self.cfg.training.scheduler.active:
            print(f"Using scheduler: {self.cfg.training.scheduler.name}")
            scheduler = get_scheduler(self.cfg.training.scheduler.name,
                                      optimizer=optimizer,
                                      scheduler_params=self.cfg.training.scheduler.kwargs)
        else:
            scheduler = None

        train_ds, val_ds, test_ds = self.init_datasets()
        kwargs = self.cfg.training.get("contrastive_kwargs", {})
        strategy = Contrastive(
            cfg=self.cfg,
            model=model,
            optimizer=optimizer,
            train_dataset=train_ds,
            val_dataset=val_ds,
            test_dataset=test_ds,
            checkpoint_dir=self.checkpoint_dir,
            scheduler=scheduler,
            **kwargs
        )
        return strategy

    def init_hierarchical_sft(self):
        model = self.init_model()
        optimizer = get_optimizer(self.cfg.training.optimizer.name,
                                  model_params=model.parameters(),
                                  optimizer_params=self.cfg.training.optimizer.kwargs)

        if self.cfg.training.scheduler.active:
            print(f"Using scheduler: {self.cfg.training.scheduler.name}")
            scheduler = get_scheduler(self.cfg.training.scheduler.name,
                                      optimizer=optimizer,
                                      scheduler_params=self.cfg.training.scheduler.kwargs)
        else:
            scheduler = None

        train_ds, val_ds, test_ds = self.init_datasets()
        print(f"Hierarchical training mode: {self.cfg.training.get('mode', 'full')}")
        strategy = HierarchicalSFT(
            cfg=self.cfg,
            model=model,
            optimizer=optimizer,
            train_dataset=train_ds,
            val_dataset=val_ds,
            test_dataset=test_ds,
            checkpoint_dir=self.checkpoint_dir,
            scheduler=scheduler,
            mode=self.cfg.training.get("mode", "full")
        )
        return strategy

    def init_GPLA(self):
        model = self.init_model()
        optimizer = get_optimizer(self.cfg.training.optimizer.name,
                                  model_params=model.parameters(),
                                  optimizer_params=self.cfg.training.optimizer.kwargs)
        if self.cfg.training.scheduler.active:
            print(f"Using scheduler: {self.cfg.training.scheduler.name}")
            scheduler = get_scheduler(self.cfg.training.scheduler.name,
                                      optimizer=optimizer,
                                      scheduler_params=self.cfg.training.scheduler.kwargs)
        else:
            scheduler = None

        train_ds, val_ds, test_ds = self.init_datasets()
        contrastive_model = self.init_clip_model()
        preference_generator = PreferenceGenerator(
            self.cfg, contrastive_model=contrastive_model)
        strategy = GPLA(
            cfg=self.cfg,
            model=model,
            optimizer=optimizer,
            train_dataset=train_ds,
            val_dataset=val_ds,
            test_dataset=test_ds,
            checkpoint_dir=self.checkpoint_dir,
            preference_generator=preference_generator,
            scheduler=scheduler
        )
        return strategy
