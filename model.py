import torch
import torch.nn as nn

from embed_regularize import embedded_dropout
from locked_dropout import LockedDropout
from weight_drop import WeightDrop


class RNNModel(nn.Module):
    """Container module with an encoder, a recurrent module, and a decoder."""

    # pylint: disable=too-many-arguments
    def __init__(self, rnn_type, ntoken, ninp, nhid, nlayers, *, dropout=0.5, dropouth=0.5, dropouti=0.5, dropoute=0.1, wdrop=0, tie_weights=False):
        super(RNNModel, self).__init__()
        self.lockdrop = LockedDropout()
        self.idrop = nn.Dropout(dropouti)
        self.hdrop = nn.Dropout(dropouth)
        self.drop = nn.Dropout(dropout)
        self.encoder = nn.Embedding(ntoken, ninp)
        self.rnns = [torch.nn.LSTM(ninp if layer == 0 else nhid, nhid if layer != nlayers - 1 else ninp, 1, dropout=0) for layer in range(nlayers)]
        print(self.rnns)
        if wdrop:
            self.rnns = [WeightDrop(rnn, ['weight_hh_l0'], dropout=wdrop) for rnn in self.rnns]
        self.rnns = torch.nn.ModuleList(self.rnns)
        self.decoder = nn.Linear(nhid, ntoken)

        # Optionally tie weights as in:
        # "Using the Output Embedding to Improve Language Models" (Press & Wolf 2016)
        # https://arxiv.org/abs/1608.05859
        # and
        # "Tying Word Vectors and Word Classifiers: A Loss Framework for Language Modeling" (Inan et al. 2016)
        # https://arxiv.org/abs/1611.01462
        if tie_weights:
            # if nhid != ninp:
            #     raise ValueError('When using the tied flag, nhid must be equal to emsize')
            self.decoder.weight = self.encoder.weight

        self.init_weights()

        self.rnn_type = rnn_type
        self.ninp = ninp
        self.nhid = nhid
        self.nlayers = nlayers
        self.dropout = dropout
        self.dropouti = dropouti
        self.dropouth = dropouth
        self.dropoute = dropoute

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.detach().uniform_(-initrange, initrange)
        self.decoder.bias.detach().fill_(0)
        self.decoder.weight.detach().uniform_(-initrange, initrange)

    # pylint: disable=arguments-differ,too-many-locals
    def forward(self, inputdata, hidden, return_h=False):
        emb = embedded_dropout(self.encoder, inputdata, dropout=self.dropoute if self.training else 0)
        # emb = self.idrop(emb)

        emb = self.lockdrop(emb, self.dropouti)

        raw_output = emb
        new_hidden = []
        # raw_output, hidden = self.rnn(emb, hidden)
        raw_outputs = []
        outputs = []
        for layer, rnn in enumerate(self.rnns):
            raw_output, new_h = rnn(raw_output, hidden[layer])
            new_hidden.append(new_h)
            raw_outputs.append(raw_output)
            if layer != self.nlayers - 1:
                # self.hdrop(raw_output)
                raw_output = self.lockdrop(raw_output, self.dropouth)
                outputs.append(raw_output)
        hidden = new_hidden

        output = self.lockdrop(raw_output, self.dropout)
        outputs.append(output)

        decoded = self.decoder(output.view(output.size(0) * output.size(1), output.size(2)))
        result = decoded.view(output.size(0), output.size(1), decoded.size(1))
        if return_h:
            return result, hidden, raw_outputs, outputs
        return result, hidden

    def sample_sequence(self, idx0, hc0 = None, n_items = 100, sampling_temp = 0.5):
        with torch.no_grad():
            idxs = [idx0]
            hcs = hc0 or self.init_hidden(1)

            top_hs = []
            for _ in range(n_items):
                idx_tensor = next(self.parameters()).data.new(1, 1).long()
                idx_tensor[0, 0] = idxs[-1]
                out_h = self.encoder(idx_tensor)  # length 1, batchsize 1
                new_hcs = []
                for i_layer, rnn in enumerate(self.rnns):
                    out_h, new_hc = rnn(out_h, hcs[i_layer])
                    new_hcs.append(new_hc)
                hcs = new_hcs
                top_hs.append(out_h.data)

                assert out_h.size(0) == 1
                assert out_h.size(1) == 1
                assert out_h.size(2) == self.encoder.weight.size(1)

                [[output_logits]] = self.decoder(out_h)
                hotter_pred = torch.nn.functional.softmax(output_logits / sampling_temp, dim = -1)
                word_idx = int(torch.multinomial(hotter_pred.data, 1, replacement = True))
                idxs.append(word_idx)

            return idxs[1:], new_hcs, top_hs

    def init_hidden(self, bsz):
        weight = next(self.parameters()).data
        return [(weight.new_zeros(1, bsz, self.nhid if layer != self.nlayers - 1 else self.ninp),
                weight.new_zeros(1, bsz, self.nhid if layer != self.nlayers - 1 else self.ninp))
                for layer in range(self.nlayers)]
