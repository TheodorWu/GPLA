import os
import math
import torch

if os.environ.get('BNB_ACTIVE', 'False') == 'True':
    import bitsandbytes as bnb
else:
    print("BNB_ACTIVE not set or set to False. Not importing BitsAndBytes.")


def get_optimizer(name, model_params, optimizer_params):
    if name.lower()=="adam":
        return torch.optim.Adam(model_params, **optimizer_params)
    elif name.lower()=="adam8bit":
        return bnb.optim.Adam8bit(model_params, **optimizer_params) # pylint: disable=possibly-used-before-assignment
    elif name.lower()=="adamw":
        return torch.optim.AdamW(model_params, **optimizer_params)
    else:
        raise NameError(f"Unknown optimizer specified: {name}")

def get_scheduler(name, optimizer, scheduler_params):
    if name.lower() == "linear":
        return torch.optim.lr_scheduler.LinearLR(optimizer, **scheduler_params)
    elif name.lower() == "cosine_annealing":
        eta_min_factor = scheduler_params.get("eta_min_factor", 0.01)

        base_lrs = [group['lr'] for group in optimizer.param_groups]
        eta_mins = [base_lr * eta_min_factor for base_lr in base_lrs]

        cosine_scheduler = MultiGroupCosineAnnealingLR(
            optimizer,
            T_max=scheduler_params.get("T_max", 10000),  # Remaining steps after warmup
            eta_mins=eta_mins
        )
        return cosine_scheduler
    elif name.lower() == "cosine_with_warmup":
        total_training_steps = scheduler_params.get("T_max", 10000)
        warmup_iters = scheduler_params.get("warmup_iters", 1000)
        eta_min_factor = scheduler_params.get("eta_min_factor", 0.01)

        base_lrs = [group['lr'] for group in optimizer.param_groups]
        eta_mins = [base_lr * eta_min_factor for base_lr in base_lrs]

        # Warmup scheduler
        warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=eta_min_factor,  # Start at 1% of base LR
            total_iters=warmup_iters    # Warmup for 1000 steps
        )

        # Cosine annealing scheduler
        cosine_scheduler = MultiGroupCosineAnnealingLR(
            optimizer,
            T_max=total_training_steps - warmup_iters,  # Remaining steps after warmup
            eta_mins=eta_mins
        )
        return torch.optim.lr_scheduler.SequentialLR(optimizer, [warmup_scheduler, cosine_scheduler], milestones=[warmup_iters])
    elif name.lower() == "cosine_with_warmup_and_restarts":
        warmup_iters = scheduler_params.get("warmup_iters", 1000)
        eta_min_factor = scheduler_params.get("eta_min_factor", 0.01)

        T_0 = scheduler_params.get("restart_period", 2000)  # First restart after 2k steps
        T_mult = scheduler_params.get("restart_multiplier", 1)  # Keep same period, or use 2 for doubling


        base_lrs = [group['lr'] for group in optimizer.param_groups]
        eta_mins = [base_lr * eta_min_factor for base_lr in base_lrs]

        # Warmup scheduler
        warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=eta_min_factor,  # Start at 1% of base LR
            total_iters=warmup_iters    # Warmup for 1000 steps
        )

        restart_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=T_0,  # Restart every 2000 steps
            T_mult=T_mult,  # Period multiplier (1 = same period, 2 = doubling)
            eta_min=min(eta_mins)  # Note: single value, not list
        )

        return torch.optim.lr_scheduler.SequentialLR(optimizer, [warmup_scheduler, restart_scheduler], milestones=[warmup_iters])
    else:
        raise NameError(f"Unknown scheduler specified: {name}")

class MultiGroupCosineAnnealingLR(torch.optim.lr_scheduler.LRScheduler):
        def __init__(self, optimizer, T_max, eta_mins, last_epoch=-1):
            self.T_max = T_max
            self.eta_mins = eta_mins
            super().__init__(optimizer, last_epoch)

        def get_lr(self):
            if self.last_epoch == 0:
                return [group['lr'] for group in self.optimizer.param_groups]

            return [
                eta_min + (base_lr - eta_min) * (1 + math.cos(math.pi * self.last_epoch / self.T_max)) / 2
                for base_lr, eta_min in zip(self.base_lrs, self.eta_mins)
            ]

