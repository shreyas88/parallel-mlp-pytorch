import torch
import torch.nn as nn
from torch.cuda.amp import custom_fwd, custom_bwd


class LinearColumnWithGradReduce(torch.autograd.Function):
    """See linear_with_grad_accumulation_and_async_allreduce"""
    @staticmethod
    @custom_fwd
    def forward(ctx,input,weight,bias):
        ctx.save_for_backward(input, weight)
        output = torch.matmul(input, weight.t())
        return output + bias

    @staticmethod
    @custom_bwd
    def backward(ctx, grad_output):
        # grad_output is (batch, T, output_size_partition)
        # input is (batch, T, input_size_partition)
        input, weight = ctx.saved_tensors

        # (batch, output_size_partition) * (output_size_partition, input_size) -> (batch, input_size)   
        #  (batch, T, input_size) = (batch, T, 1) * (1, input_size)  
        grad_input = grad_output.matmul(weight)

        # Asynchronous all-reduce
        handle = torch.distributed.all_reduce(grad_input, async_op=True)

        # collapse first two dimensions
        grad_output = grad_output.view(-1, grad_output.size(-1))
        input = input.view(-1, input.size(-1))

        # (batch*T, output_size_partition) * (batch*T, input_size_partition) -> (output_size_partition, input_size_partition)
        grad_weight = grad_output.t().matmul(input)
        grad_bias = grad_output.sum(dim=0)
        handle.wait()
        return grad_input, grad_weight, grad_bias

class ColumnParallelLinear(torch.nn.Module):
    """Linear layer with column parallelism.

    The linear layer is defined as Y = XA + b. A is parallelized along
    its second dimension as A = [A_1, ..., A_p].
    """
    def __init__(self, weight_per_rank, bias_per_rank):
        super(ColumnParallelLinear, self).__init__()
        self.weight = nn.Parameter(weight_per_rank)
        self.bias = nn.Parameter(bias_per_rank)

    def forward(self, input_: torch.Tensor):
        return LinearColumnWithGradReduce.apply(input_, self.weight, self.bias)
