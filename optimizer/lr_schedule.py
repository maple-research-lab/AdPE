import math

def adjust_learning_rate(optimizer, epoch, args):
    """Decay the learning rate with half-cycle cosine after warmup"""
    if epoch < args.warmup_epochs:
        lr = args.lr * epoch / args.warmup_epochs
    else:
        lr = args.min_lr + (args.lr - args.min_lr) * 0.5 * \
            (1. + math.cos(math.pi * (epoch - args.warmup_epochs) / (args.epochs - args.warmup_epochs)))
    for param_group in optimizer.param_groups:
        if "lr_scale" in param_group:
            param_group["lr"] = lr * param_group["lr_scale"]
        else:
            param_group["lr"] = lr
    return lr
def adjust_learning_rate_adv(optimizer, epoch, args):
    """Decay the learning rate with half-cycle cosine after warmup"""
    init_lr = args.lr
    if epoch < args.warmup_epochs:
        lr = args.lr * epoch / args.warmup_epochs
    else:
        lr = args.min_lr + (args.lr - args.min_lr) * 0.5 * \
            (1. + math.cos(math.pi * (epoch - args.warmup_epochs) / (args.epochs - args.warmup_epochs)))
    adv_lr = args.adv_lr
    if epoch < args.warmup_epochs:
        adv_lr = -args.adv_lr * epoch / args.warmup_epochs
    else:
        adv_lr = args.min_lr + (args.adv_lr - args.min_lr) * 0.5 * \
            (1. + math.cos(math.pi * (epoch - args.warmup_epochs) / (args.epochs - args.warmup_epochs)))
        adv_lr = -adv_lr
    feedback_adv_lr=0
    for param_group in optimizer.param_groups:
        if "lr_scale" in param_group:
            param_group["lr"] = lr * param_group["lr_scale"]
        else:
            param_group["lr"] = lr
        if 'fix_lr' in param_group:
            if param_group['fix_lr']==1:
                param_group["lr"] = -args.adv_lr
                if "lr_scale" in param_group:
                    param_group["lr"] = -args.adv_lr * param_group["lr_scale"]
            else:
                param_group["lr"] = adv_lr
                if "lr_scale" in param_group:
                    param_group["lr"] = adv_lr * param_group["lr_scale"]
            feedback_adv_lr = param_group["lr"]
    return lr,feedback_adv_lr
