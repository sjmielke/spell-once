import numpy as np
import torch


def embedded_dropout(embed, words, dropout=0.1, scale=None):
  if dropout:
    mask = torch.empty_like(embed.weight, requires_grad = False)
    mask.resize_((embed.weight.size(0), 1))
    mask.bernoulli_(1 - dropout)
    mask /= (1 - dropout)
    masked_embed_weight = mask.expand_as(embed.weight) * embed.weight
  else:
    masked_embed_weight = embed.weight
  if scale:
    masked_embed_weight = scale.expand_as(masked_embed_weight) * masked_embed_weight

  padding_idx = embed.padding_idx
  if padding_idx is None:
      padding_idx = -1
  embmatrix = torch.nn.functional.embedding(words, masked_embed_weight,  # pylint: disable=protected-access
    padding_idx, embed.max_norm, embed.norm_type,
    embed.scale_grad_by_freq, embed.sparse
  )
  return embmatrix


def tests():
  vocabsize = 50
  hiddensize = 4
  bptt = 10
  batch_size = 2

  embed = torch.nn.Embedding(vocabsize, hiddensize)

  words = np.random.random_integers(low = 0, high = vocabsize - 1, size = (batch_size, bptt))
  words = torch.LongTensor(words)

  orig_embedmat = embed(words)
  embedmat = embedded_dropout(embed, words)

  print(orig_embedmat)
  print(embedmat)


if __name__ == '__main__':
  tests()
