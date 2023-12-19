import torch
from dist_utils import dist_launcher
from column_parallel import ColumnParallelLinear
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn


HIDDEN_DIM = 40
BATCH_SIZE = 1024

def my_test(weight_layer1,bias_layer1,weight_layer2,bias_layer2, x):
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    input_size, output_size_per_partition = 40, HIDDEN_DIM*2
    weight_per_rank = torch.split(weight_layer1, output_size_per_partition, -1)[rank]
    bias_per_rank = torch.split(bias_layer1, output_size_per_partition, -1)[rank]
    myColParallelModule = ColumnParallelLinear(rank, world_size, weight_per_rank, bias_per_rank).to(torch.cuda.get_device())
    out, out_per_rank = myColParallelModule(x.to(torch.cuda.get_device()))
    #print("My rank",rank)
    #print(torch.cuda.get_device())
    print("Rank output:",out_per_rank.cpu().shape)
    print("Output:",out.cpu().shape)


if __name__=='__main__':
    mp.set_start_method('spawn')
    weight_layer1 = torch.randn(HIDDEN_DIM, HIDDEN_DIM*4, dtype=torch.float32)
    bias_layer1 = torch.randn(HIDDEN_DIM*4, dtype=torch.float32)
    weight_layer2 = torch.randn(HIDDEN_DIM*4, HIDDEN_DIM, dtype=torch.float32)
    bias_layer2 = torch.randn(HIDDEN_DIM, dtype=torch.float32)
    x = torch.randn(BATCH_SIZE, HIDDEN_DIM)
    dist_launcher(2,my_test,weight_layer1,bias_layer1,x)
