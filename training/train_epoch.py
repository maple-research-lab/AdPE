
import math
import sys
from typing import Iterable
import torch
import time

from ops.Logger import MetricLogger,SmoothedValue
from ops.distribute_utils import all_reduce_mean
from optimizer.lr_schedule import adjust_learning_rate_adv

def train_epoch(model: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler,
                    log_writer=None,
                    args=None):
    model.train(True)
    metric_logger = MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('adv_lr', SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 20

    accum_iter = args.accum_iter

    optimizer.zero_grad()

    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.log_dir))
    print("number of iterations: ",len(data_loader))
    #for data_iter_step, (samples, _) in enumerate(data_loader):
    for data_iter_step, (samples, _) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):

        #start_time = time.time()
        # we use a per iteration (instead of per epoch) lr scheduler
        tmp_lr,tmp_adv_lr= adjust_learning_rate_adv(optimizer, data_iter_step / len(data_loader) + epoch, args)
        for k in range(len(samples)):
            samples[k]= samples[k].to(device, non_blocking=True)
        samples = torch.cat(samples,dim=0)
        gpu_time1 = time.time()
        with torch.cuda.amp.autocast():
            loss, _, _ = model(samples, mask_ratio=args.mask_ratio)
        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        loss /= accum_iter
        loss_scaler(loss, optimizer, parameters=model.parameters(),
                    update_grad=(data_iter_step + 1) % accum_iter == 0)
        if (data_iter_step + 1) % accum_iter == 0:
            optimizer.zero_grad()

        gpu_time2 = time.time()
        metric_logger.update(gpu_time=gpu_time2-gpu_time1)
        torch.cuda.synchronize()

        metric_logger.update(loss=loss_value)

        lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(lr=lr)
        metric_logger.update(adv_lr=tmp_adv_lr)
        loss_value_reduce = all_reduce_mean(loss_value)
        if log_writer is not None and (data_iter_step + 1) % accum_iter == 0:
            """ We use epoch_1000x as the x-axis in tensorboard.
            This calibrates different curves when batch size changes.
            """
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
            log_writer.add_scalar('train_loss', loss_value_reduce, epoch_1000x)
            log_writer.add_scalar('lr', lr, epoch_1000x)


    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
