import torch
from torch.nn import Parameter


class WeightDrop(torch.nn.Module):
    def __init__(self, module, weights, dropout=0, variational=False):
        super(WeightDrop, self).__init__()
        self.module = module
        self.weights = weights
        self.dropout = dropout
        self.variational = variational
        self._setup()

    # pylint: disable=unused-argument,no-self-use,no-method-argument
    def widget_demagnetizer_y2k_edition(*args, **kwargs):
        # We need to replace flatten_parameters with a nothing function
        # It must be a function rather than a lambda as otherwise pickling explodes
        # We can't write boring code though, so ... WIDGET DEMAGNETIZER Y2K EDITION!
        # (╯°□°）╯︵ ┻━┻
        return

    def _setup(self):
        # Terrible temporary solution to an issue regarding compacting weights re: CUDNN RNN
        if issubclass(type(self.module), torch.nn.RNNBase):
            self.module.flatten_parameters = self.widget_demagnetizer_y2k_edition

        for name_w in self.weights:
            print('Applying weight drop of {} to {}'.format(self.dropout, name_w))
            weight = getattr(self.module, name_w)
            del self.module._parameters[name_w]  # pylint: disable=protected-access
            self.module.register_parameter(name_w + '_raw', Parameter(weight.data))

    def _setweights(self):
        for name_w in self.weights:
            raw_w = getattr(self.module, name_w + '_raw')
            new_w = None
            if self.variational:
                mask = raw_w.new_ones((raw_w.size(0), 1))
                mask = torch.nn.functional.dropout(mask, p=self.dropout, training=True)
                new_w = mask.expand_as(raw_w) * raw_w
            else:
                new_w = torch.nn.functional.dropout(raw_w, p=self.dropout, training=self.training)
            setattr(self.module, name_w, new_w)

    def forward(self, *args):
        self._setweights()
        return self.module.forward(*args)


def tests():
    # Input is (seq, batch, input)
    x = torch.autograd.Variable(torch.randn(2, 1, 10))
    h0 = None

    ###

    print('Testing WeightDrop')
    print('=-=-=-=-=-=-=-=-=-=')

    ###

    print('Testing WeightDrop with Linear')

    lin = WeightDrop(torch.nn.Linear(10, 10), ['weight'], dropout=0.9)
    run1 = [x.sum() for x in lin(x).data]
    run2 = [x.sum() for x in lin(x).data]

    print('All items should be different')
    print('Run 1:', run1)
    print('Run 2:', run2)

    assert run1[0] != run2[0]
    assert run1[1] != run2[1]

    print('---')

    ###

    print('Testing WeightDrop with LSTM')

    wdrnn = WeightDrop(torch.nn.LSTM(10, 10), ['weight_hh_l0'], dropout=0.9)

    run1 = [x.sum() for x in wdrnn(x, h0)[0].data]
    run2 = [x.sum() for x in wdrnn(x, h0)[0].data]

    print('First timesteps should be equal, all others should differ')
    print('Run 1:', run1)
    print('Run 2:', run2)

    # First time step, not influenced by hidden to hidden weights, should be equal
    assert run1[0] == run2[0]
    # Second step should not
    assert run1[1] != run2[1]

    print('---')


if __name__ == '__main__':
    tests()
