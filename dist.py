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


def my_test(rank, queue, weight_layer1, bias_layer1, weight_layer2,bias_layer2, x, dummy_labels):
    rank = dist.get_rank()
    device_id = torch.cuda.current_device()
    output_size_per_partition = HIDDEN_DIM*2
    weight_per_rank_layer1 = torch.split(weight_layer1, output_size_per_partition, -1)[rank].to(device_id)
    bias_per_rank_layer1 = torch.split(bias_layer1, output_size_per_partition, -1)[rank].to(device_id)
    
    weight_per_rank_layer2 = torch.split(weight_layer2, output_size_per_partition, 0)[rank].to(device_id)

    myColParallelModule = ColumnParallelLinear(rank, weight_per_rank_layer1, bias_per_rank_layer1).to(device_id)
    x_cuda = x.to(device_id)
    out_layer1_per_rank = myColParallelModule(x_cuda)
    
    relu = nn.ReLU().to(device_id)
    out_relu_per_rank = relu(out_layer1_per_rank)

    rowParallelLinearModule = RowParallelLinear(rank, weight_per_rank_layer2, bias_layer2).to(
        device_id)
    out_layer2 = rowParallelLinearModule(out_relu_per_rank)

    # trigger backward pass
    loss = torch.square(out_layer2.to(device_id)- dummy_labels.to(device_id)).sum()
    loss.backward()

    if rank == 0:
        queue.put(out_layer2.cpu().clone().detach())
        queue.put(x_cuda.grad.clone().cpu().detach())

if __name__=='__main__':
    mp.set_start_method('spawn')
    weight_layer1 = torch.randn(HIDDEN_DIM, HIDDEN_DIM*4, dtype=torch.float32)
    bias_layer1 = torch.randn(HIDDEN_DIM*4, dtype=torch.float32)
    
    weight_layer2 = torch.randn(HIDDEN_DIM*4, HIDDEN_DIM, dtype=torch.float32)
    bias_layer2 = torch.randn(HIDDEN_DIM, dtype=torch.float32)

    x = torch.randn(BATCH_SIZE, SEQ_LEN, HIDDEN_DIM)
    # dummy labels used for loss calculation
    dummy_labels = torch.randn(BATCH_SIZE, SEQ_LEN, HIDDEN_DIM)

    base_mlp = BaseMLPLayers(weight_layer1, bias_layer1, weight_layer2, bias_layer2)

    # check forward pass output with base MLP
    base_output = base_mlp(x).cpu()
    activations, grad_actual = dist_launcher(2,my_test,weight_layer1,bias_layer1,weight_layer2, 
                                             bias_layer2, x, dummy_labels)
    print(base_output[0][0][0:10])
    print(activations[0][0][0:10])

    assert torch.allclose(base_output, activations, atol=1e-4)
    print("Parallel MLP output matched with base MLP output")

    # dummy loss function
    loss = torch.square(activations-dummy_labels).sum()
    loss.backward()
    # calculated gradient for input
    grad_expected = x.grad
    print(base_output[0][0][0:10])
    print(activations[0][0][0:10])
    assert torch.allclose(grad_expected, grad_actual, atol=1e-4)
    print("Parallel MLP gradient matched with base MLP gradient")

