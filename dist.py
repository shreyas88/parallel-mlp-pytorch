import torch
from dist_utils import dist_launcher
from column_parallel import ColumnParallelLinear
from row_parallel import RowParallelLinear
from base_mlp import BaseMLPLayers
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn


HIDDEN_DIM = 40
BATCH_SIZE = 1024
SEQ_LEN = 128

def my_test(rank, queue, weight_layer1, bias_layer1, weight_layer2,bias_layer2, x):
    rank = dist.get_rank()
    output_size_per_partition = HIDDEN_DIM*2
    weight_per_rank_layer1 = torch.split(weight_layer1, output_size_per_partition, -1)[rank]
    bias_per_rank_layer1 = torch.split(bias_layer1, output_size_per_partition, -1)[rank]
    
    weight_per_rank_layer2 = torch.split(weight_layer2, output_size_per_partition, 0)[rank]

    myColParallelModule = ColumnParallelLinear(weight_per_rank_layer1, bias_per_rank_layer1).to(torch.cuda.current_device())
    out_layer1_per_rank = myColParallelModule(x.to(torch.cuda.current_device()))
    
    relu = nn.ReLU()
    out_relu_per_rank = relu(out_layer1_per_rank)

    rowParallelLinearModule = RowParallelLinear(weight_per_rank_layer2, bias_layer2).to(torch.cuda.current_device())
    out_layer2 = rowParallelLinearModule(out_relu_per_rank)

    if rank == 0:
        queue.put(out_layer2.clone.detach())

if __name__=='__main__':
    mp.set_start_method('spawn')
    weight_layer1 = torch.randn(HIDDEN_DIM, HIDDEN_DIM*4, dtype=torch.float32)
    bias_layer1 = torch.randn(HIDDEN_DIM*4, dtype=torch.float32)
    
    weight_layer2 = torch.randn(HIDDEN_DIM*4, HIDDEN_DIM, dtype=torch.float32)
    bias_layer2 = torch.randn(HIDDEN_DIM, dtype=torch.float32)

    x = torch.randn(BATCH_SIZE, SEQ_LEN, HIDDEN_DIM)

    base_mlp = BaseMLPLayers(weight_layer1, bias_layer1, weight_layer2, bias_layer2)
    base_output = base_mlp(x).cpu()
    dist_out = dist_launcher(2,my_test,weight_layer1,bias_layer1,weight_layer2, bias_layer2, x)
    assert torch.allclose(base_output, dist_out)
