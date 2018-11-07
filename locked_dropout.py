import torch.nn as nn
from torch.autograd import Variable


class LockedDropout(nn.Module):
    def __init__(self):
        super().__init__()

    # pylint: disable=arguments-differ
    def forward(self, x, dropout=0.5):
        if not self.training or not dropout:
            return x
        mask = x.new_empty(1, x.size(1), x.size(2), requires_grad = False)
        mask.bernoulli_(1 - dropout)
        mask /= (1 - dropout)
        return mask.expand_as(x) * x
