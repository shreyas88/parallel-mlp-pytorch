import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
from dist_utils import dist_launcher
from column_parallel import ColumnParallelLinear
from row_parallel import RowParallelLinear
from base_mlp import BaseMLPLayers

HIDDEN_DIM = 40
BATCH_SIZE = 1024
SEQ_LEN = 128
OUTPUT_SIZE_PER_PARTITION = HIDDEN_DIM * 2

def split_tensor(tensor, partition_size, dim, rank):
    """
    Splits a tensor into partitions along a specified dimension.
    """
    return torch.split(tensor, partition_size, dim)[rank].to(torch.cuda.current_device())

def run_parallel_mlp(rank, queue, weight_layer1, bias_layer1, weight_layer2,bias_layer2, x, dummy_labels):
    rank = dist.get_rank()
    device_id = torch.cuda.current_device()
    
    # Split and move weights and biases to the current device
    weight_per_rank_layer1 = split_tensor(weight_layer1, OUTPUT_SIZE_PER_PARTITION, -1, rank)
    bias_per_rank_layer1 = split_tensor(bias_layer1, OUTPUT_SIZE_PER_PARTITION, -1, rank)
    weight_per_rank_layer2 = split_tensor(weight_layer2, OUTPUT_SIZE_PER_PARTITION, 0, rank)

    # Create and apply ColumnParallelLinear module
    myColParallelModule = ColumnParallelLinear(rank, weight_per_rank_layer1, 
                                               bias_per_rank_layer1).to(device_id)
    x_cuda = x.to(device_id).requires_grad_(True)
    out_layer1_per_rank = myColParallelModule(x_cuda)
    
    # Apply ReLU activation
    relu = nn.ReLU().to(device_id)
    out_relu_per_rank = relu(out_layer1_per_rank)

    # Create and apply RowParallelLinear module
    rowParallelLinearModule = RowParallelLinear(rank, weight_per_rank_layer2, bias_layer2).to(
        device_id)
    out_layer2 = rowParallelLinearModule(out_relu_per_rank)

    # Compute loss and perform backward pass
    loss = torch.square(out_layer2 - dummy_labels.to(device_id)).sum()
    loss.backward()

    # Save outputs and gradients if rank is 0
    if rank == 0:
        queue.put(out_layer2.cpu().clone().detach())
        queue.put(x_cuda.grad.clone().cpu().detach())

def init_tensors():
    weight_layer1 = torch.randn(HIDDEN_DIM, HIDDEN_DIM*4, dtype=torch.float32)
    bias_layer1 = torch.randn(HIDDEN_DIM*4, dtype=torch.float32)
    
    weight_layer2 = torch.randn(HIDDEN_DIM*4, HIDDEN_DIM, dtype=torch.float32)
    bias_layer2 = torch.randn(HIDDEN_DIM, dtype=torch.float32)

    x = torch.randn(BATCH_SIZE, SEQ_LEN, HIDDEN_DIM)

    # dummy labels used for loss calculation
    dummy_labels = torch.randn(BATCH_SIZE, SEQ_LEN, HIDDEN_DIM)
    return weight_layer1, bias_layer1, weight_layer2, bias_layer2, x, dummy_labels

if __name__=='__main__':
    mp.set_start_method('spawn')

    ################################################
    # Init the weights in the main function 
    # and pass it to the child processes
    # to enable checking against the baseline MLP
    ################################################
    weight_layer1, bias_layer1, weight_layer2, bias_layer2, x, dummy_labels = init_tensors()

    # Run the baseline MLP to verify parallel MLP logic 
    base_mlp = BaseMLPLayers(weight_layer1, bias_layer1, weight_layer2, bias_layer2)

    # we are doing some unsual stuff here, cloning the tensor to avoid backprop 
    # through the distributed code path
    clone_x = x.clone().requires_grad_(True)
    # check forward pass output with base MLP
    base_output = base_mlp(clone_x).cpu()

    # Run the distributed code path including Parallel MLP
    activations, grad_actual = dist_launcher(2,run_parallel_mlp,weight_layer1,bias_layer1,weight_layer2, 
                                             bias_layer2, x, dummy_labels)
    print(base_output[0][0][0:10])
    print(activations[0][0][0:10])

    assert torch.allclose(base_output, activations, atol=1e-4)
    print("Parallel MLP output matched with base MLP output")

    # dummy loss function
    loss = torch.square(base_output-dummy_labels).sum()
    loss.backward()
    # calculated gradient for input
    grad_expected = clone_x.grad
    print(grad_expected[0][0][0:10])
    print(grad_actual[0][0][0:10])
    # gradients have lower tolerance for some reason
    assert torch.allclose(grad_expected, grad_actual, atol=1e-1)
    print("Parallel MLP gradient matched with base MLP gradient")

