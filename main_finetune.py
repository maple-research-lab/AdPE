
#Copyright (C) 2022 Xiao Wang
#License: MIT for academic use.
#Contact: Xiao Wang (wang3702@purdue.edu, xiaowang20140001@gmail.com)

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# BEiT: https://github.com/microsoft/unilm/tree/master/beit
# MAE: https://github.com/facebookresearch/mae
# --------------------------------------------------------
from ops.argparser import argparser_finetune
import torch
import torch.multiprocessing as mp
import timm
assert timm.__version__ == "0.3.2" # version check
def main(args):
    import socket
    hostname = socket.gethostname()
    local_ip = socket.gethostbyname(hostname)
    print("local ip: ",local_ip)

    ngpus_per_node = torch.cuda.device_count()
    args.world_size = args.world_size*ngpus_per_node
    from training.main_worker_finetune import main_worker

    mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node,  args))


if __name__ == '__main__':
    use_cuda = torch.cuda.is_available()
    print("starting check cuda status",use_cuda)
    args = argparser_finetune()
    args = args.parse_args()
    main(args)
