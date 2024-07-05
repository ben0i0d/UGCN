import math
import torch

class CosineWarmupScheduler(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, args, epoch_len, last_epoch=-1):
        self.args = args
        self.total_steps = args.epochs * epoch_len
        self.warmup_steps = args.warmup_epochs * epoch_len
        self.cos_total_steps = self.total_steps - self.warmup_steps
        self.base_lr = args.learning_rate * args.batch_size / 256
        self._step_count = last_epoch * epoch_len
        super(CosineWarmupScheduler, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        if self._step_count < self.warmup_steps:
            lr = (self.base_lr * self._step_count) / self.warmup_steps
        else:
            q = 0.5 * (1 + math.cos(math.pi * (self._step_count - self.warmup_steps) / self.cos_total_steps))
            lr = self.base_lr * q + (self.base_lr * 0.001) * (1 - q)
        return [lr] * len(self.optimizer.param_groups)
    
    def step(self):
        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group['lr'] = lr
        self._step_count += 1


def adjust_learning_rate(args, optimizer, loader, step):
    max_steps = args.epochs * len(loader)
    warmup_steps = 5 * len(loader)
    base_lr = args.learning_rate * args.batch_size / 256
    if step < warmup_steps:
        lr = base_lr * step / warmup_steps
    else:
        step -= warmup_steps
        max_steps -= warmup_steps
        q = 0.5 * (1 + math.cos(math.pi * step / max_steps))
        end_lr = base_lr * 0.001
        lr = base_lr * q + end_lr * (1 - q)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr