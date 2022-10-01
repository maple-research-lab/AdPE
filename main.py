
#Copyright (C) 2020 Xiao Wang
#License: MIT for academic use.
#Contact: Xiao Wang (wang3702@purdue.edu, xiaowang20140001@gmail.com)

import os
from ops.argparser import  argparser
import torch
import torch.multiprocessing as mp
def main(args):
    import socket
    hostname = socket.gethostname()
    local_ip = socket.gethostbyname(hostname)
    print("local ip: ",local_ip)

    ngpus_per_node = torch.cuda.device_count()
    args.world_size = args.world_size*ngpus_per_node
    from training.main_worker import main_worker
    mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node,  args))
if __name__ == '__main__':
    use_cuda = torch.cuda.is_available()
    print("starting check cuda status",use_cuda)
    #if use_cuda:
    parser = argparser()
    args = parser.parse_args()
    #if args.nodes_num<=1:
    main(args)
