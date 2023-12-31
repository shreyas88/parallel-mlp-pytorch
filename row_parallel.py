import torch
import torch.nn as nn
from torch.cuda.amp import custom_fwd, custom_bwd


class LinearRowWithTensorReduce(torch.autograd.Function):
    @staticmethod
    @custom_fwd
    def forward(ctx,input,weight,bias,rank):
        ctx.save_for_backward(input, weight)
        if rank == 0:
            output = torch.matmul(input, weight) + bias
        else:
            output = torch.matmul(input, weight)
        # all reduce along tensor parallel dimension
        torch.distributed.all_reduce(output)
        return output
        

    @staticmethod
    @custom_bwd
    def backward(ctx, grad_output):
        input, weight = ctx.saved_tensors
        # (batch, T, input_size) * (output_size_partition, input_size) -> (batch, T, input_size)
        grad_input = grad_output.matmul(weight.t())
        grad_output = grad_output.view(-1, grad_output.size(-1))
        input = input.view(-1, input.size(-1))
        # (output_size_partition,batch*T) * (batch*T, input_size) -> (output_size_partition, input_size)
        grad_weight = input.T.matmul(grad_output)
        grad_bias = grad_output.sum(dim=0)
        return grad_input, grad_weight, grad_bias, None

class RowParallelLinear(torch.nn.Module):
    """Linear layer with row parallelism.
    its second dimension as Z =   X  [ Y1
                                       Y2 ]
    """
    def __init__(self, rank, weight_per_rank, bias_per_rank):
        super(RowParallelLinear, self).__init__()
        self.rank = rank
        # weight_per_rank is (output_size_partition, input_size)
        self.weight = nn.Parameter(weight_per_rank)
        # bias_per_rank is (input_size,)
        self.bias = nn.Parameter(bias_per_rank)

    def forward(self, input_: torch.Tensor):
        # input_ is (batch, T, output_size_partition)
        return LinearRowWithTensorReduce.apply(input_, self.weight, self.bias, self.rank)
