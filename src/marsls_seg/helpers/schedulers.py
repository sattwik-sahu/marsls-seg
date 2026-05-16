import torch


def get_lr_scheduler(optimizer, warmup_epochs, total_epochs):
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return float(epoch) / float(max(1, warmup_epochs))
        return 1.0  # Constant after warmup, or add decay logic here

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
