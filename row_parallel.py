import torch
import torch.nn as nn


class LinearRowWithTensorReduce(torch.autograd.Function):
    @staticmethod
    @custom_fwd
    def forward(ctx,input,weight,bias):
        ctx.save_for_backward(input, weight)
        output = torch.matmul(input, weight) + bias
        # all reduce along tensor parallel dimension
        return torch.distributed.all_reduce(output)
        

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
        return grad_input, grad_weight, grad_bias

class RowParallelLinear(torch.nn.Module):
    """Linear layer with row parallelism.
    its second dimension as Z =   X  [ Y1
                                       Y2 ]
    """
    def __init__(self, weight_per_rank, bias_per_rank):
        super(RowParallelLinear, self).__init__()
        # weight_per_rank is (output_size_partition, input_size)
        self.weight = nn.Parameter(weight_per_rank)
        # bias_per_rank is (input_size,)
        self.bias = nn.Parameter(bias_per_rank)

    def forward(self, input_: torch.Tensor):
        # input_ is (batch, T, output_size_partition)
        LinearRowWithTensorReduce.apply(input_, self.weight, self.bias)
