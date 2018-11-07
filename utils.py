import torch


def repackage_hidden(hid):
    """Wraps hidden states in new Tensors,
    to detach them from their history."""
    if isinstance(hid, torch.Tensor):
        return hid.detach()
    else:
         return tuple(repackage_hidden(v) for v in hid)


def batchify(data, bsz):
    # Work out how cleanly we can divide the dataset into bsz parts.
    nbatch = data.size(0) // bsz
    # Trim off any extra elements that wouldn't cleanly fit (remainders).
    data = data.narrow(0, 0, nbatch * bsz)
    # Evenly divide the data across the bsz batches.
    return data.view(bsz, -1).t().contiguous()


# split_type = corpusdata.corpus.dictionary.word2idx[data.EOSSYM]
def sentence_batchify(data, bsz, split_type):
    # Sentences are ordered by length before this process to minimize maxlen (nondeterministic!)
    # Returns:
    # 1) a *list* of batchified tensors, each of dimension maxlen x batchsize,
    # 2) the lengths of the (sorted) sentences contained in them in a single tensor, and
    # 3) the second argument of the original sort call, to reorder results once they are obtained.
    assert data.dim() == 1
    assert data[-1] == split_type
    data = torch.cat((torch.tensor([split_type]), data))
    split_indices = (data == split_type).nonzero().squeeze(1)
    start_end_pairs = torch.stack((split_indices[:-1], split_indices[1:] + 1)).t()
    all_sents = [data[start:end] for (start, end) in start_end_pairs]
    all_lens = torch.tensor([s.size(0) for s in all_sents])
    sorted_lens, back_keys = torch.sort(all_lens)
    maxlens = [torch.max(b) for b in sorted_lens.split(bsz)]
    sorted_sents = [all_sents[i] for i in back_keys]
    proto_batches = [sorted_sents[i:i+bsz] for i in range(0, len(sorted_sents), bsz)] # list of lists of sentences
    assert all([max([len(s) for s in b]) == maxlen for b, maxlen in zip(proto_batches, maxlens)])
    # Pad with 1 -- that will always exist and never be UNK.
    batches = [torch.stack([torch.cat((s, s.new_ones(maxlen.item() - s.size(0)))) for s in b], dim = 1) for b, maxlen in zip(proto_batches, maxlens)]
    return batches, sorted_lens, back_keys


def get_batch(source, start_index, seq_len):
    seq_len = min(seq_len, len(source) - 1 - start_index)
    data = source[start_index : start_index+seq_len]  # noqa: E203,E226
    target = source[start_index+1 : start_index+1+seq_len].view(-1)  # noqa: E203,E226
    return data, target
