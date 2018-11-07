import itertools
from collections import Counter
from numpy import unique

import torch


# Sort all entries by length, then cut into batches of max_batchsize (except for last, which may be smaller)
def batchify_seqtensors(lengths, tensorlist, max_batchsize):
  sorted_lengths, sorted_indices = torch.sort(lengths)

  for start in range(0, lengths.size(0), max_batchsize):
    end = start + max_batchsize
    batch_indices = sorted_indices[start:end]
    batch_lengths = sorted_lengths[start:end]
    batch_maxlen = torch.max(batch_lengths).item()
    batch_tensorlist = [tensor[sorted_indices][start:end, 0:batch_maxlen].long() for tensor in tensorlist]
    yield (batch_lengths, batch_indices, batch_tensorlist)


class ConditionedLSTM(torch.nn.Module):
  # A vector of size conditioner_size will be added to the input token embedding of each element of some sequence.
  # If conditioner_size == 0, this is pretty much a standard LSTM.
  def __init__(self, vocab, token_dim, conditioner_size, hidden_units, *, dropout_p, conditioner_dropout_p = None, feed_n_characters = 1, num_layers = 1):
    if conditioner_dropout_p is None:
      conditioner_dropout_p = dropout_p

    super(ConditionedLSTM, self).__init__()
    self.vocab = vocab

    self.feed_n_characters = feed_n_characters

    self.embeddings = torch.nn.Embedding(len(vocab), token_dim)
    self.rnn = torch.nn.LSTM(
        input_size = conditioner_size + feed_n_characters * token_dim,
        hidden_size = hidden_units,
        num_layers = num_layers)
    self.dropout = torch.nn.Dropout(p = dropout_p)
    self.conditioner_dropout = torch.nn.Dropout(p = conditioner_dropout_p)
    self.output_predictor = torch.nn.Linear(
        in_features = hidden_units,
        out_features = len(vocab))

    # TODO: consider overriding pytorchs normal init with orthogonal matrices or something

    print("ConditionedLSTM number of params:", sum(p.numel() for p in self.parameters()))

  # Access to the conditioner input weights and gradients (raw data!)
  def get_conditioner_inputweights(self):
    if self.rnn.weight_ih_l0.grad is None:
      torch.sum(self.rnn.weight_ih_l0).backward()
      self.rnn.weight_ih_l0.grad.zero_()
      print("Manually initializing gradients of input weights.")
    dat = self.rnn.weight_ih_l0.data[:, :-self.embeddings.weight.size(1)]
    gra = self.rnn.weight_ih_l0.grad.data[:, :-self.embeddings.weight.size(1)]
    return (dat, gra)

  # Returns losses (a 1D tensor of batchsize), and, if feed_samples == True, the predicted words of the batch
  # Potential TODO: combine all tensors into one, so we can use http://pytorch.org/docs/master/nn.html#dataparallel-layers-multi-gpu-distributed ?
  # pylint: disable=arguments-differ,too-many-locals
  def forward(self, full_conditioner, goldtensor_in, goldtensor_out, goldlengths, *, sampling_temp = 1.0, feed_samples = False):
    batchsize = full_conditioner.size()[0]
    maxlen = goldtensor_in.size(1)

    try:
      _bos = int(unique(goldtensor_in.t()[0].cpu().numpy()))  # noqa: F841
      eos = int(unique(goldtensor_out.t()[-1].cpu().numpy()))
      # print("Found <bos>:", self.vocab[_bos], "and <eos>:", self.vocab[eos])
    except TypeError:
      print("ConditionedLSTM forward assumes that the first and last column of goldtensor_in and _out only contain <bos> and <eos>, respectively!")
      print(" _in:", [self.vocab[i] for i in unique(goldtensor_in.t()[0].cpu().numpy())])
      print(" _out:", [self.vocab[i] for i in unique(goldtensor_out.t()[-1].cpu().numpy())])
      exit(1)

    # If we feed in more than 1 character, this is the time to preprocess the gold tensors to get some additional prefix space!
    new_prefixes = [goldtensor_in[:, 0].unsqueeze(1)] * (self.feed_n_characters - 1)

    hc = torch.zeros((self.rnn.num_layers, batchsize, self.rnn.hidden_size), requires_grad = False, device = next(self.parameters()).device)
    hc = (hc, hc)

    losses = 0

    dropped_conditioner = self.conditioner_dropout(full_conditioner) if self.rnn.input_size > self.embeddings.embedding_dim else None

    if feed_samples:
      # Timestep-wise processing required
      next_tokens = torch.cat(new_prefixes + [goldtensor_in[:, 0].unsqueeze(1)], dim = 1)
      next_tokens.requires_grad_(False)
      charss = []

      for charidx in range(maxlen):
        # RNN running
        char_emb = self.dropout(self.embeddings(next_tokens)).view(batchsize, -1)
        combi = torch.cat([dropped_conditioner, char_emb], dim = 1) if dropped_conditioner is not None else char_emb
        out, hc = self.rnn(combi.unsqueeze(0), hc)

        # Output prediction
        preds = self.output_predictor(self.dropout(out.squeeze(0)))
        logsoftmax_preds = torch.nn.functional.log_softmax(preds, dim = -1)

        # Loss
        mask = (charidx < goldlengths).unsqueeze(1)
        all_log_likelihoods = torch.gather(logsoftmax_preds, 1, goldtensor_out.t()[charidx].unsqueeze(0).t())
        masked_log_likelihoods = all_log_likelihoods * mask.type_as(all_log_likelihoods)
        losses -= masked_log_likelihoods.squeeze(dim = -1)

        # Sampling the next token
        hotter_preds = torch.nn.functional.softmax(preds / sampling_temp, dim = -1)
        samples = torch.multinomial(hotter_preds.data, 1, replacement = True)
        next_tokens = torch.cat([next_tokens[:, 1:], samples], dim = 1) if self.feed_n_characters > 1 else samples
        charss.append([idx for idx in next_tokens[:, -1]])

      predwords = []
      for chars in zip(*charss):
        if eos in chars:
          predwords.append("".join([self.vocab[idx] for idx in chars[:chars.index(eos)]]))
        else:
          predwords.append("".join([self.vocab[idx] for idx in chars]) + "[...]")

    else:
      # Full RNN running
      all_char = self.dropout(self.embeddings(torch.cat(new_prefixes + [goldtensor_in], dim = 1).t()))
      all_char = all_char.unfold(0, self.feed_n_characters, 1).contiguous().view(maxlen, batchsize, -1)

      combi = torch.cat([dropped_conditioner.unsqueeze(0).repeat(maxlen, 1, 1), all_char], dim = -1) if dropped_conditioner is not None else all_char
      outs, _ = self.rnn(combi, hc)

      # Output prediction
      preds = self.output_predictor(self.dropout(outs))
      logsoftmax_preds = torch.nn.functional.log_softmax(preds, dim = -1).transpose(0, 1)

      # Loss
      mask = torch.arange(0, maxlen, dtype = torch.long, device = next(self.parameters()).device) < goldlengths.unsqueeze(1)
      all_log_likelihoods = torch.gather(logsoftmax_preds, -1, goldtensor_out.unsqueeze(-1)).squeeze(-1)
      masked_log_likelihoods = all_log_likelihoods * mask.type_as(all_log_likelihoods)
      losses -= masked_log_likelihoods.sum(dim = 1)

    return (losses, predwords) if feed_samples else losses


class Vocabizer():
  def __init__(self, *, unk_typ = '<unk>', unk_countlt = None, unk_after = None, bos = '<bos>', eos = '<eos>', seq_sep = '\n', tok_sep = ' '):
    if unk_typ is None:
      assert unk_countlt is None and unk_after is None
    elif unk_countlt is not None:
      assert unk_after is None
      assert unk_countlt > 0
    elif unk_after is not None:
      assert unk_after >= 3

    self.unk_typ = unk_typ
    self.unk_countlt = unk_countlt
    self.unk_after = unk_after
    self.bos = bos
    self.eos = eos
    self.seq_sep = seq_sep
    self.tok_sep = tok_sep
    self.idx2typ = None
    self.typ2idx = None

  def string2tokss(self, text, seq_sep = None, tok_sep = None):
    seq_sep = seq_sep or self.seq_sep
    tok_sep = tok_sep or self.tok_sep
    return [list(seq.split(tok_sep) if tok_sep != '' else seq) for seq in text.split(seq_sep)]

  def tokss2vocab(self, tokss):
    read_vocab = list(Counter(itertools.chain.from_iterable(tokss)).most_common())

    if self.unk_after is not None:
      alive_types = read_vocab[:self.unk_after - 3]
      if len(alive_types) < self.unk_after - 3:
        print("Note: vocab is smaller than the desired upper bound for UNKing.")
    elif self.unk_countlt is not None:
      alive_types = [(w, c) for (w, c) in read_vocab if c >= self.unk_countlt]
    else:
      alive_types = read_vocab

    unked = set(read_vocab) - set(alive_types)
    print("Vocab size:", len(alive_types))
    if len(unked) > 0:
      print("Voluntarily UNKing out", len(unked), "tokens:", sorted(list(unked)))

    self.idx2typ = [self.bos, self.eos] + ([self.unk_typ] if self.unk_typ else []) + [w for (w, c) in alive_types]
    self.typ2idx = {t: i for (i, t) in enumerate(self.idx2typ)}

  def tokss2seqtensors(self, tokss):
    lengths = torch.LongTensor([len(s) + 1 for s in tokss])  # plus BOS or EOS
    maxlen = int(torch.max(lengths))
    inputs = torch.LongTensor(len(tokss), maxlen)
    outputs = torch.LongTensor(len(tokss), maxlen)
    inputs[:] = self.typ2idx[self.eos]
    outputs[:] = self.typ2idx[self.eos]
    inputs[:, 0] = self.typ2idx[self.bos]
    for i_seq, seq in enumerate(tokss):
      for i_tok, tok in enumerate(seq):
        idx = self.typ2idx[tok] if tok in self.typ2idx else self.typ2idx[self.unk_typ]
        inputs[i_seq, i_tok + 1] = idx
        outputs[i_seq, i_tok] = idx
    return (inputs, outputs, lengths)


class DiscreteConditionedSequenceModel():
  def __init__(self, *, token_vocab, token_dim, tag_vocab, tag_dim, hidden_units, lr, wdecay, clip_gradients, dropout_p):
    self.lstm = ConditionedLSTM(
        vocab = token_vocab,
        token_dim = token_dim,
        conditioner_size = tag_dim,
        hidden_units = hidden_units,
        dropout_p = dropout_p)

    self.tagembeddings = torch.nn.Embedding(
        num_embeddings = len(tag_vocab),
        embedding_dim = tag_dim)

    self.lr = lr
    self.wdecay = wdecay
    self.clip_gradients = clip_gradients
    self.idx2tag = tag_vocab
    self.tag2idx = {t: i for (i, t) in enumerate(tag_vocab)}

  # Convenience function to run tags (strings or anything in tag_vocab) through the embedding layer by using the tag_vocab to intify
  def embed_tags(self, tag_sequence):
    tag_idxs = [self.tag2idx[t] for t in tag_sequence]
    tag_idxs = torch.LongTensor(tag_idxs, device = next(self.tagembeddings.parameters()).device)
    return self.tagembeddings(tag_idxs)

  def train_tensors(self, tags, goldtensor_in, goldtensor_out, goldlengths, *, sampling_temp = 1.0, feed_samples = False):
    self.lstm.train()

    _dev = next(self.lstm.parameters()).device
    goldtensor_in = goldtensor_in.to(_dev)
    goldtensor_out = goldtensor_out.to(_dev)
    goldlengths = goldlengths.to(_dev)

    conditioner = self.embed_tags(tags)
    losses = self.lstm.forward(conditioner, goldtensor_in, goldtensor_out, goldlengths, sampling_temp = sampling_temp, feed_samples = feed_samples)
    if feed_samples:
      losses = losses[0]
    loss = torch.sum(losses) / torch.sum(goldlengths)

    paramlist = list(self.lstm.parameters()) + list(self.tagembeddings.parameters())
    optimizer = torch.optim.SGD(paramlist, lr = self.lr, weight_decay = self.wdecay)
    optimizer.zero_grad()
    loss.backward()
    if self.clip_gradients is not None:
      torch.nn.utils.clip_grad_norm(paramlist, self.clip_gradients)
    optimizer.step()

    return loss

  # pylint: disable=too-many-locals
  def train_tensors_batched(self, tags, goldtensor_in, goldtensor_out, goldlengths, *, max_batchsize, sampling_temp = 1.0, feed_samples = False):
    loss = 0.0  # plain numbers, just for prettyprinting at the end!
    batches = batchify_seqtensors(goldlengths, [goldtensor_in, goldtensor_out], max_batchsize = max_batchsize)
    for batch_lengths, batch_indices, [batch_inputs, batch_outputs] in batches:
      batch_tags = [tags[i] for i in batch_indices]
      newloss = self.train_tensors(batch_tags, batch_inputs, batch_outputs, batch_lengths, sampling_temp = sampling_temp, feed_samples = feed_samples)
      loss += newloss.data
    return loss

  def evaluate_tensors(self, tags, goldtensor_in, goldtensor_out, goldlengths, *, sampling_temp = 1.0, feed_samples = False):
    self.lstm.eval()

    _dev = next(self.lstm.parameters()).device
    goldtensor_in = goldtensor_in.to(_dev)
    goldtensor_out = goldtensor_out.to(_dev)
    goldlengths = goldlengths.to(_dev)

    conditioner = self.embed_tags(tags)
    losses = self.lstm.forward(conditioner, goldtensor_in, goldtensor_out, goldlengths, sampling_temp = sampling_temp, feed_samples = feed_samples)
    if feed_samples:
      losses, words = losses
      return (torch.sum(losses) / torch.sum(goldlengths)).item(), words
    else:
      return (torch.sum(losses) / torch.sum(goldlengths)).item()
