from ops.distribute_utils import init_distributed_mode
import os
import torch
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from ops.distribute_utils import get_rank,get_world_size,is_main_process
from optimizer.config_wd import add_weight_decay
from optimizer.NativeScaler import NativeScalerWithGradNormCount as NativeScaler
from training.io_utils import save_model,load_model
import time
import datetime
import json


def main_worker(gpu, ngpus_per_node,args):
    init_distributed_mode(gpu,ngpus_per_node,args)

    print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(', ', ',\n'))

    cudnn.benchmark = True
    device = torch.device(args.device)
    #config transformation
    transform_train = transforms.Compose([
            transforms.RandomResizedCrop(args.input_size, scale=(args.scale_min,args.scale_max),
                                         interpolation=transforms.InterpolationMode.BICUBIC),  # 3 is bicubic
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    from data_processing.MultiTransform import MultiTransform
    transform_train = MultiTransform(transform_train,args.num_crop)
    dataset_train = datasets.ImageFolder(os.path.join(args.data_path, 'train'), transform=transform_train)
    print(dataset_train)

    if  args.distributed:
        num_tasks = get_world_size()
        global_rank = get_rank()
        sampler_train = torch.utils.data.DistributedSampler(
            dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
        )
        print("Sampler_train = %s" % str(sampler_train))

    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)

    if global_rank == 0 and args.log_dir is not None:
        os.makedirs(args.log_dir, exist_ok=True)
        log_writer = SummaryWriter(log_dir=args.log_dir)
    else:
        log_writer = None
    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
    )
    if args.pe_type==0:
        # Absolute Position Embedding (APE)
        import model.ad_ape as adpe

    else:
        # Relative Position Embedding (RPE)
        import model.ad_rpe as adpe
    model = adpe.__dict__[args.model](args=args,norm_pix_loss=args.norm_pix_loss,
                                                     img_size=args.input_size)

    model.to(device)
    model_without_ddp = model

    #configure batch size, learning rate
    print("Model = %s" % str(model_without_ddp))
    eff_batch_size = args.batch_size * args.accum_iter * get_world_size()
    args.lr = args.blr * eff_batch_size / 256
    args.adv_lr = args.adv_lr * eff_batch_size / 256
    print("base lr: %.2e" % (args.lr * 256 / eff_batch_size))
    print("actual lr: %.2e" % args.lr)
    print("adversarial lr:%.2e"%args.adv_lr)
    print("accumulate grad iterations: %d" % args.accum_iter)
    print("effective batch size: %d" % eff_batch_size)

    #configure optimizer
    use_adv_in_optimizer = True# if args.norm==0 else False
    param_groups = add_weight_decay(model,args.lr,args.adv_lr,args.weight_decay,args.adv_wd,use_adv_in_optimizer)
    print(param_groups)
    if args.norm!=0:
        from optimizer.adv_optimizer import adv_optimizer
        optimizer = adv_optimizer(param_groups, lr=args.lr, betas=(0.9, 0.95),norm_type=args.norm,limit=args.eps)
    else:
        optimizer = torch.optim.AdamW(param_groups, lr=args.lr, betas=(0.9, 0.95))

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
        model_without_ddp = model.module
    loss_scaler = NativeScaler()

    load_model(args=args, model_without_ddp=model_without_ddp, optimizer=optimizer, loss_scaler=loss_scaler)

    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)
        train_stats = train_epoch(
            model, data_loader_train,
            optimizer, device, epoch, loss_scaler,
            log_writer=log_writer,
            args=args
        )
        if args.output_dir and (epoch % 10 == 0 or epoch + 1 == args.epochs):
            save_model(
                args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                loss_scaler=loss_scaler, epoch=epoch)

        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                        'epoch': epoch,}

        if args.output_dir and is_main_process():
            if log_writer is not None:
                log_writer.flush()
            with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                f.write(json.dumps(log_stats) + "\n")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))




