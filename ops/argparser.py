import argparse
def argparser():
    parser = argparse.ArgumentParser('AdPE pre-training', add_help=False)
    parser.add_argument('--batch_size', default=128, type=int,
                        help='Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus')
    parser.add_argument('--epochs', default=400, type=int,help="number of epochs for pre-training")
    parser.add_argument('--accum_iter', default=4, type=int,
                        help='Accumulate gradient iterations (for increasing the effective batch size under memory constraints)')

    # Model parameters
    parser.add_argument('--model', default='adpe_vit_large_patch16', type=str, metavar='MODEL',
                        help='Name of model to train')


    parser.add_argument('--mask_ratio', default=0.75, type=float,
                        help='Masking ratio (percentage of removed patches).')

    parser.add_argument('--norm_pix_loss', action='store_true',
                        help='Use (per-patch) normalized pixels as targets for computing loss')
    parser.set_defaults(norm_pix_loss=True)#we always use norm pixel

    # Optimizer parameters
    parser.add_argument('--weight_decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')
    parser.add_argument('--lr', type=float, default=None, metavar='LR',
                        help='learning rate (absolute lr)')
    parser.add_argument('--blr', type=float, default=1.5e-3, metavar='LR',
                        help='base learning rate: absolute_lr = base_lr * total_batch_size / 256')
    parser.add_argument('--min_lr', type=float, default=0., metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0')

    parser.add_argument('--warmup_epochs', type=int, default=20, metavar='N',
                        help='epochs to warmup LR')

    # Dataset parameters
    parser.add_argument('--data_path', default='imagenet', type=str, help='dataset path')

    parser.add_argument('--output_dir', default='./adpe_pretrain',
                        help='path where to save, empty for no saving')
    parser.add_argument('--log_dir', default='./adpe_pretrain',
                        help='path where to tensorboard log')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default='',help='resume from checkpoint')

    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.set_defaults(pin_mem=True)

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist_url', default='tcp://localhost:10001', help='url used to set up distributed training')
    parser.add_argument('--rank', default=-1, type=int,help="specify the rank of script")

    parser.add_argument("--pe_type",type=int,default=0,help="Position Embedding (PE) Type : \n 0: Absolute Position Embedding (APE)\n"
                                                         "1: Relative Position Embedding (RPE)")

    parser.add_argument("--adv_lr",type=float, default=1.5e-3, help="base adversarial learning rate for training")
    parser.add_argument("--adv_wd",type=float, default=0, help="adversarial weight decay for training")

    parser.add_argument("--num_crop",type=int,default=4,help="number of crops for training")
    parser.add_argument('--input_size', default=112, type=int,
                        help='images input size')
    parser.add_argument("--scale_min",type=float,default=0.08,help="resized crop scale min")
    parser.add_argument("--scale_max",type=float,default=1.0,help="resized crop scale max")

    parser.add_argument("--adv_type",type=int,default=0,help="Adversarial Type: 0: No adversarial \n 1: Embedding Adversarial \n 2: Coordinate Adversarial")

    #add pgd constraints
    parser.add_argument("--norm",type=int,default=0,help="PGD norm constraint 0:none \n 1: L-2 \n 2:L-inf\n")
    parser.add_argument("--eps",type=float,default=1,help="epsilon constraint for PGD")

    return parser
