import os
import torch
import datetime
import builtins
import numpy as np
import torch.distributed as dist

def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True
def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()
def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()
def all_reduce_mean(x):
    world_size = get_world_size()
    if world_size > 1:
        x_reduce = torch.tensor(x).cuda()
        dist.all_reduce(x_reduce)
        x_reduce /= world_size
        return x_reduce.item()
    else:
        return x
def is_main_process():
    return get_rank() == 0
def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    builtin_print = builtins.print

    def print(*args, **kwargs):

        if is_master:
            now = datetime.datetime.now().time()
            builtin_print('[{}] '.format(now), end='')  # print with time stamp
            builtin_print(*args, **kwargs)

    builtins.print = print

def init_distributed_mode(gpu,ngpus_per_node,args):

    import resource
    rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
    resource.setrlimit(resource.RLIMIT_NOFILE, (2048, rlimit[1]))
    args.gpu = gpu
    args.rank = args.rank * ngpus_per_node + gpu
    os.environ['LOCAL_RANK'] = str(args.gpu)
    os.environ['RANK'] = str(args.rank)
    os.environ['WORLD_SIZE'] = str(args.world_size)
    print("make sure the distributed mode is ",args.dist_url)



    args.distributed = True

    torch.cuda.set_device(args.gpu)
    args.dist_backend = 'nccl'
    print('| distributed init (rank {}): {}, gpu {}'.format(
        args.rank, args.dist_url, args.gpu), flush=True)
    torch.distributed.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                         timeout=datetime.timedelta(seconds=36000),
                                         world_size=args.world_size, rank=args.rank)

    setup_for_distributed(args.rank == 0)



    # fix the seed for reproducibility
    seed = args.seed + get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
