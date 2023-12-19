import torch
from dist_utils import dist_launcher
from column_parallel import ColumnParallelLinear
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn


def my_test(weight,bias,x):
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    input_size, output_size_per_partition = 40, 100
    weight_per_rank = torch.split(weight, output_size_per_partition, -1)[rank]
    bias_per_rank = torch.split(bias, output_size_per_partition, -1)[rank]
    myColParallelModule = ColumnParallelLinear(rank, world_size, weight_per_rank, bias_per_rank).to(torch.cuda.get_device())
    out, out_per_rank = myColParallelModule(x.to(torch.cuda.get_device()))
    #print("My rank",rank)
    #print(torch.cuda.get_device())
    print("Rank output:",out_per_rank.cpu().shape)
    print("Output:",out.cpu().shape)


if __name__=='__main__':
    mp.set_start_method('spawn')
    weight_tensor = torch.randn(40, 200, dtype=torch.float32)
    bias_tensor = torch.randn(200, dtype=torch.float32)
    x = torch.randn(1024, 40)
    dist_launcher(2,my_test,weight_tensor,bias_tensor,x)
