#!/usr/bin/env python3

import argparse
from collections import namedtuple
import time
import math
import random
import numpy as np
import torch
from scipy.misc import logsumexp

from tensorboardX import SummaryWriter

import data
import model as model_module
from conditioned_rnn import ConditionedLSTM

from utils import batchify, sentence_batchify, get_batch, repackage_hidden

PARSER = argparse.ArgumentParser(description='PyTorch RNN/LSTM Language Model')
PARSER.add_argument('--data', type=str, default='data/wikitext-2-raw',
                    help='location of the data corpus (ending in »-char« or »-bpe« does some magic for bpc stuff, requiring the non-char/-bpe version to be available)')
PARSER.add_argument('--char-min-count', type=int, default=25,
                    help='char mininum count (else replaced by ◊)')
PARSER.add_argument('--vocab-size', type=int, default=10000,
                    help='vocab size (including UNK)')
PARSER.add_argument('--max-type-length', type=int, default=20,
                    help='maximum length of a type to be included in speller training/testing tensors')
PARSER.add_argument('--model', type=str, default='LSTM',
                    help='type of recurrent net (LSTM, QRNN, GRU)')
PARSER.add_argument('--emsize', type=int, default=400,
                    help='size of word embeddings')
PARSER.add_argument('--nhid', type=int, default=1150,
                    help='number of hidden units per layer')
PARSER.add_argument('--nlayers', type=int, default=3,
                    help='number of layers')
PARSER.add_argument('--lr-lm', type=float, default=30,
                    help='initial learning rate')
PARSER.add_argument('--lr-speller', type=float, default=30,
                    help='learning rate for the spelling model')
PARSER.add_argument('--speller-mode', default='full-typ',
                    help='speller mode: {full,1gram,uncond,nobackoff}-{tok,typ},none,backoff-tok')
PARSER.add_argument('--tokmode-oversample', type=int, default=50,
                    help='the tok-level lexeme tensor will contain as many times lexemes as the type-level tensor would')
PARSER.add_argument('--open-vocab', action='store_true',
                    help='be open-vocab (dont UNK out UNKs)')
PARSER.add_argument('--open-vocab-during-training', action='store_true',
                    help='also train open-vocab (predict UNKs in speller)')
PARSER.add_argument('--rescale-h-unk', type=float, default=1.0,
                    help='Multiply every h_unk with this value before feeding it into the speller')
PARSER.add_argument('--normalize-h-unk', action='store_true',
                    help='Normalize every h_unk (before rescaling, of course)')
PARSER.add_argument('--speller-factor', type=float, default=1.0,
                    help='factor in front of spelling loss that is added to LM loss')
PARSER.add_argument('--speller-asgd', type=int, default=999999999,
                    help='use ASGD instead of vanilla SGD for optimizing speller parameters from this epoch on')
PARSER.add_argument('--speller-interval', type=int, default=50,
                    help='train speller on each n-th batch of the main lm')
PARSER.add_argument('--speller-batchsize', type=int, default=1500, metavar='N',
                    help='size of the batch of lexemes that the speller is trained on ever $speller-interval iterations')
PARSER.add_argument('--speller-nuclear-regularization', type=float, default=1,
                    help='nuclear norm loss on the conditioner part of the speller input matrix, global scale (not per-anything), performed during speller batch loss')
PARSER.add_argument('--speller-char-dim', type=int, default=5,
                    help='dimensions of the character embeddings')
PARSER.add_argument('--speller-num-layers', type=int, default=3,
                    help='LSTM layers for the spelling model')
PARSER.add_argument('--speller-feed-n-characters', type=int, default=1,
                    help='feed the speller RNN the last n generated characters (usually 1)')
PARSER.add_argument('--speller-hidden', type=int, default=100,
                    help='hidden units in the speller LSTM')
PARSER.add_argument('--speller-dropout', type=float, default=0.2,
                    help='dropout for speller LSTM')
PARSER.add_argument('--speller-conditioner-dropout', type=float, default=0.5,
                    help='dropout for the conditioning part of the speller LSTM')
PARSER.add_argument('--speller-wdecay', type=float, default=1.2e-6,
                    help='weight decay applied to speller weights')
PARSER.add_argument('--exclude-in-vocab-spellings-for-unk', action='store_true',
                    help='exclude in vocab spellings when spelling UNK (only at test time, but still very costly!)')
PARSER.add_argument('--clip', type=float, default=0.25,
                    help='gradient clipping')
PARSER.add_argument('--epochs', type=int, default=2000,
                    help='upper epoch limit')
PARSER.add_argument('--batch_size', type=int, default=80, metavar='N',
                    help='batch size')
PARSER.add_argument('--bptt', type=int, default=70,
                    help='sequence length')
PARSER.add_argument('--dropout', type=float, default=0.4,
                    help='dropout applied to layers (0 = no dropout)')
PARSER.add_argument('--dropouth', type=float, default=0.3,
                    help='dropout for rnn layers (0 = no dropout)')
PARSER.add_argument('--dropouti', type=float, default=0.65,
                    help='dropout for input embedding layers (0 = no dropout)')
PARSER.add_argument('--dropoute', type=float, default=0.1,
                    help='dropout to remove words from embedding layer (0 = no dropout)')
PARSER.add_argument('--wdrop', type=float, default=0.5,
                    help='amount of weight dropout to apply to the RNN hidden to hidden matrix')
PARSER.add_argument('--not-tied', action='store_true',
                    help='don\'t tie the word embedding and softmax weights')
PARSER.add_argument('--seed', type=int, default=1,
                    help='random seed')
PARSER.add_argument('--bootstrap-with-seed', type=int,
                    help='resample the dev/test sets with this seed when reading them in')
PARSER.add_argument('--bootstrap-training-data', action='store_true',
                    help='resample the training set as well!')
PARSER.add_argument('--nonmono', type=int, default=5,
                    help='iterations before switching (?)')
PARSER.add_argument('--device', default='cpu',
                    help='device to use: cpu, cuda:0, etc.')
PARSER.add_argument('--resume', action='store_true',
                    help='resume training, i.e., load the saved file specified by --save before the first epoch')
PARSER.add_argument('--log-interval', type=int, default=200, metavar='N',
                    help='report interval')
PARSER.add_argument('--overfitting-check-interval', type=int, default=999999, metavar='N',
                    help='interval for checking the nasty sum of in-vocab spellings')
PARSER.add_argument('--sample-words', type=int, default=0,
                    help='sample this many tokens from the full LM at each final check')
PARSER.add_argument('--per-line', action='store_true',
                    help='print losses/logprobs (i.e., nats) per sentence (i.e., line) of the test set')
PARSER.add_argument('--save', type=str, default=''.join(str(time.time()).split('.')) + '.pt',
                    help='path to save the final model')
PARSER.add_argument('--boardcomment', type=str,
                    help='"comment" for the run in tensorboardX (if none, don\'t use tensorboardX')
PARSER.add_argument('--no-histograms', action='store_true',
                    help='don\'t log histograms in tensorboardX (because they\'re buggy in tensorboardX :(')
PARSER.add_argument('--alpha', type=float, default=2,
                    help='alpha L2 regularization on RNN activation (alpha = 0 means no regularization)')
PARSER.add_argument('--beta', type=float, default=1,
                    help='beta slowness regularization applied on RNN activiation (beta = 0 means no regularization)')
PARSER.add_argument('--wdecay', type=float, default=1.2e-6,
                    help='weight decay applied to LM weights')
PARSER.add_argument('--embedding-stdev', type=float, default=0.0,
                    help='Std deviation of the spherical origin Gaussian prior on word embeddings')
PARSER.add_argument('--optimizer', type=str,  default='sgd',
                    help='optimizer to use (sgd, adam)')
PARSER.add_argument('--when', nargs="+", type=int, default=[-1],
                    help='When (which epochs) to divide the learning rate by 10 - accepts multiple')
ARGS = PARSER.parse_args()

# Only batch_size = 1 would be fully fair (doesn't cut off elements and uses the same hidden state throughout)!
VALID_BATCH_SIZE = 10
TEST_BATCH_SIZE = 1

# Set the random seed manually for reproducibility.
np.random.seed(ARGS.seed)
torch.manual_seed(ARGS.seed)
if torch.cuda.is_available():
    if not ARGS.device.startswith('cuda'):
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")
    else:
        torch.cuda.manual_seed(ARGS.seed)
        torch.cuda.LongTensor([0])  # just to reserve the GPU, avoiding race conditions

# Checks that we optimize the actual loss from the paper
if ARGS.lr_lm != ARGS.lr_speller:
    print("WARNING: learning rates for both RNNs are not the same")
if ARGS.speller_factor != 1.0:
    print("WARNING: speller factor is not 1.0")

# Baseline usage hac^W UI improvement
if ARGS.speller_mode.startswith('uncond'):
    ARGS.speller_nuclear_regularization = float('inf')
if ARGS.speller_mode.startswith('nobackoff'):
    ARGS.open_vocab_during_training = False

# Assert some stuff
if ARGS.bootstrap_training_data:
    assert ARGS.bootstrap_with_seed is not None

# Initialize the tensorboard writer
BOARD = SummaryWriter(comment = ARGS.boardcomment) if ARGS.boardcomment is not None else None

if BOARD is not None:
    def logscalar(tag, scalar, global_step):
        BOARD.add_scalar(tag, scalar, global_step = global_step)

    def logtext(tag, text, global_step = 0):
        BOARD.add_text(tag, text, global_step = global_step)

    def loghistogram(tag, values, bins = 'doane', global_step = 0):
        if not ARGS.no_histograms:
            BOARD.add_histogram(tag, values, bins = bins, global_step = global_step)
else:
    def logscalar(tag, scalar, global_step):
        # pylint: disable=unused-argument
        pass

    def logtext(tag, text, global_step = 0):
        # pylint: disable=unused-argument
        pass

    def loghistogram(tag, values, global_step = 0):
        # pylint: disable=unused-argument
        pass


# Caution: mutates lst, just returning for convenience!
def detshuffle(lst):
  random.seed(4)
  random.shuffle(lst)
  return lst


def to_device(thing):
    if isinstance(thing, torch.Tensor):
        return thing.to(ARGS.device)
    else:
        return tuple(to_device(t) for t in thing)


###############################################################################
# Training code
###############################################################################

def unked_tensor(tensor, vocab_size = None):
    # Can't do it in function def, cause it changes after reading in data!
    if vocab_size is None:
        vocab_size = ARGS.vocab_size
    tensor = tensor.clone()
    tensor[tensor >= vocab_size] = 0
    return tensor


# pylint: disable=too-many-locals
def unks_from_hidden_states(*, corpusdata, dropped_rnn_hs, targets, data_for_shape, model_oov_speller, max_type_length = 999999999999, force_prediction = False):
    # from dropped_rnn_hs[-1] select with a 2D-mask of shape (seqlen, batchsize) the hidden states
    # that correspond to the UNKs that are to be predicted (data != unked_data)

    unk_mask = (targets.view_as(data_for_shape) >= ARGS.vocab_size)
    unk_count = torch.sum(unk_mask).item()
    unk_types = targets.view_as(data_for_shape)[unk_mask]
    unk_hiddenstates = dropped_rnn_hs[-1][unk_mask.unsqueeze(2).expand(-1, -1, ARGS.emsize)].view(unk_count, ARGS.emsize)

    if ARGS.normalize_h_unk and len(unk_hiddenstates) > 0:
        unk_hiddenstates /= torch.sum(unk_hiddenstates ** 2, dim = -1).sqrt().unsqueeze(1).expand_as(unk_hiddenstates)
    unk_hiddenstates *= ARGS.rescale_h_unk

    if unk_count == 0:
        nullreturn = torch.FloatTensor([0.0], device = ARGS.device)
        return nullreturn, [], [], []
    elif unk_count > ARGS.speller_batchsize and not force_prediction:
        print("Choosing not to predict", unk_count, "UNKs from the hidden states of the lexeme-level LM, because the speller batch size is only", ARGS.speller_batchsize)
        nullreturn = torch.FloatTensor([0.0], device = ARGS.device)
        return nullreturn, [], [], []
    else:
        unklexeme_lengths, unklexeme_indices, unklexeme_inputs, unklexeme_outputs = \
            to_device(corpusdata.corpus.dictionary.build_wordtensors(vocab_indices = unk_types, max_type_length = max_type_length))

        assert all(unklexeme_indices == unk_types)
        # ^ this need not be true, if max_type_length < \infty! (but for simplicity we don't handle that -- the code below would, but the binning in evaluate() assumes it doesn't)
        # acceptable_indices = set(unklexeme_indices)
        # acceptable_mask = to_device(unk_types.cpu().apply_(lambda x: x in acceptable_indices).byte())
        # unk_hiddenstates = unk_hiddenstates[acceptable_mask.unsqueeze(-1).expand(-1, ARGS.emsize)].view(-1, ARGS.emsize)

        unk_losses = model_oov_speller(unk_hiddenstates, unklexeme_inputs, unklexeme_outputs, unklexeme_lengths)

        # We want to divide the unk_prob by Z = 1 - in_vocab_prob.
        # So the unk_newprob = unk_oldprob / Z, neg log both sides to obtain:
        # unk_newlogloss = -log(exp(-unk_oldlogloss) / Z)
        #                = -log(exp(-unk_oldlogloss - log(Z)))
        #                = unk_oldlogloss + log(Z)
        #                = unk_oldlogloss + log(1 - in_vocab_prob)

        if ARGS.exclude_in_vocab_spellings_for_unk:
            in_vocab_probs = [get_invocab_marginal_for_embedding(probed_embedding = unk_hiddenstates.detach()[i], corpusdata = corpusdata, model_marginal_speller = model_oov_speller) for i in range(unk_hiddenstates.size(0))]
            in_vocab_maxes = torch.FloatTensor([m for (_, m) in in_vocab_probs])
            in_vocab_probs = torch.FloatTensor([p for (p, _) in in_vocab_probs])
            unk_losses += torch.log(1.0 - in_vocab_probs.to(ARGS.device))
        else:
            in_vocab_probs = []
            in_vocab_maxes = []

        return unk_losses, (torch.sum(unk_hiddenstates ** 2, dim = 1) ** 0.5), in_vocab_probs, in_vocab_maxes


# probed_embeddings is not a Variable.
def get_invocab_marginal_for_embedding(*, probed_embedding, corpusdata, model_marginal_speller):
    # Sort tensors for similar-length words within batches
    sorted_lengths, sorted_indices = torch.sort(corpusdata.trainlexeme_lengths)
    sorted_inputs = corpusdata.trainlexeme_inputs[sorted_indices]
    sorted_outputs = corpusdata.trainlexeme_outputs[sorted_indices]

    batchsize = ARGS.speller_batchsize  # gonna be reduced for last batch
    batch_wordembs = probed_embedding.unsqueeze(0).expand(batchsize, ARGS.emsize)
    total_log_prob = float('-inf')
    max_log_prob = float('-inf')
    batchstart = 0
    while batchstart < corpusdata.trainlexeme_lengths.size(0):
        # Last batch is actually smaller
        if batchsize > corpusdata.trainlexeme_lengths.size(0) - batchstart:
            batchsize = corpusdata.trainlexeme_lengths.size(0) - batchstart
            batch_wordembs = probed_embedding.unsqueeze(0).expand(batchsize, ARGS.emsize)

        # Prepare the batch
        batch_lengths = sorted_lengths[batchstart:batchstart + batchsize]
        batch_maxlen = torch.max(batch_lengths).item()
        batch_inputs = sorted_inputs[batchstart:batchstart + batchsize, 0:batch_maxlen]
        batch_outputs = sorted_outputs[batchstart:batchstart + batchsize, 0:batch_maxlen]

        # Run and add
        word_losses = model_marginal_speller(batch_wordembs, batch_inputs, batch_outputs, batch_lengths)
        logprobs = -word_losses.cpu().numpy()
        total_log_prob = np.logaddexp(total_log_prob, float(logsumexp(logprobs)))
        max_log_prob = max(max_log_prob, float(np.amax(logprobs)))

        batchstart += batchsize

    del word_losses, sorted_lengths, sorted_indices, sorted_inputs, sorted_outputs, batch_lengths, batch_inputs, batch_outputs, batch_wordembs, logprobs

    return (math.exp(total_log_prob), math.exp(max_log_prob))


# Oh, what's that doing here? Well, we will not iterate over all spellings within just one epoch, so we have to perform this iteration (which is just a sampling approximation anyway) globally!
SPELLBATCH_I = 0


# pylint: disable=too-many-locals,too-many-statements,too-many-branches
def train_for_epoch(*, i_epoch, corpusdata, model_lexemes, model_invocab_speller, model_oov_speller, optimizer_lm, optimizer_speller):
    # pylint: disable=W0603
    global SPELLBATCH_I

    # Turn on training mode which enables dropout.
    model_lexemes.train()
    if ARGS.model == 'QRNN':
        model_lexemes.reset()
    if ARGS.speller_mode != "none":
        model_invocab_speller.train()
        model_oov_speller.train()


    # The epoch losses are sums (not per token)!
    epoch_tokens = 0
    epoch_lm_loss = 0
    epoch_unk_loss = 0
    epoch_speller_loss = 0
    printout_lm_loss = 0
    printout_unk_loss = 0
    printout_speller_loss = 0
    latest_nuclear_norm = 0
    printout_tokens = 0
    nuclear_scaling_factor = 0
    speller_loss_per_lexeme = 0
    speller_loss_per_token = torch.tensor(0.0)
    start_time = time.time()
    hidden = model_lexemes.init_hidden(ARGS.batch_size)
    batch, startidx = 0, 0
    while startidx < corpusdata.train_data.size(0) - 1 - 1:
        bptt = ARGS.bptt if np.random.random() < 0.95 else ARGS.bptt / 2.
        # Prevent excessively small or negative sequence lengths
        seq_len = max(5, int(np.random.normal(bptt, 5)))
        # There's a very small chance that it could select a very long sequence length resulting in OOM
        seq_len = min(seq_len, ARGS.bptt + 10)

        lr2 = optimizer_lm.param_groups[0]['lr']
        optimizer_lm.param_groups[0]['lr'] = lr2 * seq_len / ARGS.bptt

        inputs, targets = get_batch(corpusdata.train_data, start_index=startidx, seq_len=seq_len)
        # data: (seqlen, batchsize), targets: same (data + 1), but flattened (why?)

        # set seq_len to the *actual* sequence length (the last batch is shorter!)
        seq_len = inputs.size(0)

        # UNK the data
        unked_inputs = unked_tensor(inputs)
        unked_targets = unked_tensor(targets)

        # Starting each batch, we detach the hidden state from how it was previously produced.
        # If we didn't, the model would try backpropagating all the way to start of the dataset.
        hidden = repackage_hidden(hidden)
        optimizer_lm.zero_grad()
        if ARGS.speller_mode != "none":
            optimizer_speller.zero_grad()

        output, hidden, rnn_hs, dropped_rnn_hs = model_lexemes(unked_inputs, hidden, return_h=True)
        # hidden = final hidden state (tuples, cause LSTM)
        # rnn_hs = for each layer, for each time step, batched output (i.e. h)
        # thus we want dropped_rnn_hs[-1], a tensor of shape (seqlen, batchsize, embdim)

        lm_loss = torch.sum(torch.nn.functional.cross_entropy(output.view(-1, ARGS.vocab_size), unked_targets, reduction = 'none'))

        # `loss` will be per-token!
        # Regularization loss
        loss = lm_loss / (ARGS.batch_size * seq_len)
        # Activiation Regularization
        if ARGS.alpha: loss += sum(ARGS.alpha * dropped_rnn_h.pow(2).mean() for dropped_rnn_h in dropped_rnn_hs[-1:])
        # Temporal Activation Regularization (slowness)
        if ARGS.beta: loss += sum(ARGS.beta * (rnn_h[1:] - rnn_h[:-1]).pow(2).mean() for rnn_h in rnn_hs[-1:])

        # Additional Gaussian on lexeme embeddings
        if ARGS.embedding_stdev > 0:
            # log_prob = -torch.sum((model_lexemes.encoder.weight ** 2), dim = -1) / (2 * ARGS.embedding_stdev * ARGS.embedding_stdev) - math.log(math.sqrt(2 * math.pi) * ARGS.embedding_stdev)
            sum_neg_log_probs = torch.sum((model_lexemes.encoder.weight ** 2)) / (2 * ARGS.embedding_stdev * ARGS.embedding_stdev) + ARGS.vocab_size * math.log(math.sqrt(2 * math.pi) * ARGS.embedding_stdev)
            embed_gaussian_loss_per_token = sum_neg_log_probs / (corpusdata.train_data.size(0) * corpusdata.train_data.size(1))
            loss += embed_gaussian_loss_per_token
        else:
            sum_neg_log_probs = 0.0
            embed_gaussian_loss_per_token = 0.0

        # Spelling of in-batch, but OOV spellings (if open-vocab)
        # TODO: scrapping it for training and just doing at test time seems to have small impact, measure exactly!
        if ARGS.open_vocab and ARGS.open_vocab_during_training:
            unkloss = torch.sum(unks_from_hidden_states(corpusdata = corpusdata, dropped_rnn_hs = dropped_rnn_hs, targets = targets, data_for_shape = inputs, model_oov_speller = model_oov_speller)[0])
            loss += unkloss / (ARGS.batch_size * seq_len)
            printout_unk_loss += unkloss.item()
            epoch_unk_loss += unkloss.item()
        else:
            unkloss = 0

        # Spelling loss on in-vocab things
        if ARGS.speller_mode != "none" and batch % ARGS.speller_interval == 0:
            if len(corpusdata.trainlexeme_indices) > 0 and not ARGS.speller_mode.startswith("backoff"):
                num_all_lexemes = corpusdata.trainlexeme_lengths.size(0) // (ARGS.tokmode_oversample if ARGS.speller_mode[-3:] == 'tok' else 1)
                spellbatch_is = torch.arange(SPELLBATCH_I, SPELLBATCH_I + ARGS.speller_batchsize, dtype = torch.long, device = ARGS.device).remainder(num_all_lexemes)
                SPELLBATCH_I = (SPELLBATCH_I + ARGS.speller_batchsize) % num_all_lexemes

                batch_losses = model_invocab_speller(model_lexemes.encoder.weight[corpusdata.trainlexeme_indices[spellbatch_is]], corpusdata.trainlexeme_inputs[spellbatch_is], corpusdata.trainlexeme_outputs[spellbatch_is], corpusdata.trainlexeme_lengths[spellbatch_is])
                speller_loss_per_lexeme = torch.sum(batch_losses) / ARGS.speller_batchsize  # loss per lexeme
                # Now tie it to the token-based loss as described in the paper:
                speller_loss_per_token = speller_loss_per_lexeme * num_all_lexemes * ARGS.speller_interval / (corpusdata.train_data.size(0) * corpusdata.train_data.size(1))
                # And done!
                loss += ARGS.speller_factor * speller_loss_per_token
                epoch_speller_loss += speller_loss_per_token.item() * (corpusdata.train_data.size(0) * corpusdata.train_data.size(1))
                printout_speller_loss += speller_loss_per_token.item() * (corpusdata.train_data.size(0) * corpusdata.train_data.size(1))

            # Regularize the conditioner input weights towards:
            # 1) low-rank-ness (to prevent overfitting)
            # 2) not caring about embeddings (to limit the gradient flow into them)
            # ... by using the "nuclear norm" (trace{sqrt[A* x A]})!
            # Obtain using SVD ( https://math.stackexchange.com/a/701104 + https://math.stackexchange.com/a/1663012 )
            # TODO: once pytorch 0.4 is out, we can backprop through torch.svd()!
            if ARGS.speller_nuclear_regularization < float('inf'):
                latest_nuclear_norm = 0
                sigs = []
                nuclear_scaling_factor = ARGS.speller_nuclear_regularization * ARGS.speller_interval / (corpusdata.train_data.size(0) * corpusdata.train_data.size(1))

                for model in [model_invocab_speller] if model_invocab_speller == model_oov_speller else [model_invocab_speller, model_oov_speller]:
                    iwss, iws_grads = model.get_conditioner_inputweights()

                    # LSTM contains 4 matrices, do it for them all separately
                    for iws, iws_grad in zip(torch.chunk(iwss, 4), torch.chunk(iws_grads, 4)):
                        u, sig, v = torch.svd(iws)  # pylint: disable=invalid-name

                        # Get it for printout
                        latest_nuclear_norm += torch.sum(sig)
                        sigs.append(sig)

                        # We manually add to the the gradient (its okay, we called zero_grad further above)
                        iws_grad.add_(ARGS.lr_speller * nuclear_scaling_factor * u @ v.t())

                        del u, sig, v, iws, iws_grad
                    loghistogram('conditioner/sig', torch.cat(sigs))
                    del iwss, iws_grads

        # Debug output
        printout_lm_loss += lm_loss.item()
        printout_tokens += seq_len * ARGS.batch_size
        epoch_lm_loss += lm_loss.item()
        epoch_tokens += seq_len * ARGS.batch_size

        if batch % ARGS.log_interval == 0 and batch > 0:
            printout_lm_loss /= printout_tokens
            printout_unk_loss /= printout_tokens
            printout_speller_loss /= printout_tokens
            cur_speller_loss = speller_loss_per_token.item() if ARGS.speller_mode != "none" else 0.0
            elapsed = time.time() - start_time
            print('| epoch {:3d} | {:5d}/{:5d} batches | lr {:02.2f} | ms/batch {:5.2f} | comp sigsum {:5.2f} | '
                    'comp loss {:5.2f} | embed gaussian {:5.2f} | spell loss {:5.2f} | unk loss {:5.2f} | lm loss {:5.2f} | '
                    'joint loss {:5.2f} | ppl {:8.2f}'.format(
                i_epoch, batch, len(corpusdata.train_data) // ARGS.bptt, optimizer_lm.param_groups[0]['lr'], elapsed * 1000 / ARGS.log_interval, latest_nuclear_norm,
                latest_nuclear_norm * nuclear_scaling_factor, float(embed_gaussian_loss_per_token), cur_speller_loss, printout_unk_loss, printout_lm_loss, printout_lm_loss + printout_unk_loss,
                math.exp(printout_lm_loss + printout_unk_loss)))
            printout_lm_loss = 0
            printout_unk_loss = 0
            printout_speller_loss = 0
            latest_nuclear_norm = 0
            printout_tokens = 0
            start_time = time.time()

        # Actually perform the updates!
        loss.backward()

        # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
        if ARGS.clip: torch.nn.utils.clip_grad_norm_(list(model_lexemes.parameters()) + list(set(list(model_invocab_speller.parameters()) + list(model_oov_speller.parameters())) if ARGS.speller_mode != "none" else []), ARGS.clip)
        optimizer_lm.step()
        if ARGS.speller_mode != "none":
            optimizer_speller.step()

        # Reset the conditioning input matrix to 0 if it wasn't wanted.
        if ARGS.speller_nuclear_regularization == float('inf'):
            model_invocab_speller.get_conditioner_inputweights()[0].zero_()
            model_oov_speller.get_conditioner_inputweights()[0].zero_()
        # Reset hidden-to-hidden and input-to-hidden if it wasn't wanted (i.e. unigram-only speller) through unsafe .data
        if ARGS.speller_mode[:5] == '1gram':
            model_invocab_speller.rnn.weight_hh_l0.data.zero_()
            model_invocab_speller.rnn.weight_ih_l0.data.zero_()
            model_oov_speller.rnn.weight_hh_l0.data.zero_()
            model_oov_speller.rnn.weight_ih_l0.data.zero_()

        optimizer_lm.param_groups[0]['lr'] = lr2

        batch += 1
        startidx += seq_len

        del loss, embed_gaussian_loss_per_token, unkloss, sum_neg_log_probs, output, rnn_hs, dropped_rnn_hs, unked_inputs, unked_targets

    epoch_lm_loss /= epoch_tokens
    epoch_unk_loss /= epoch_tokens
    epoch_speller_loss /= epoch_tokens

    logscalar('training/lm/loss', epoch_lm_loss, global_step = i_epoch)
    if ARGS.speller_mode != "none" and ARGS.speller_mode != "backoff-tok":
        logscalar('training/speller/loss', epoch_speller_loss, global_step = i_epoch)
    if ARGS.open_vocab and ARGS.open_vocab_during_training:
        logscalar('training/unk/loss', epoch_unk_loss, global_step = i_epoch)

    return (epoch_lm_loss, speller_loss_per_token if ARGS.speller_mode != "none" and ARGS.speller_mode != "backoff-tok" else 0)


# Takes total_length x batch_size tensor data_source
# pylint: disable=too-many-locals,too-many-statements,too-many-branches,too-many-nested-blocks
def evaluate(data_source, *, corpusdata, model_lexemes, model_invocab_speller, model_oov_speller, i_epoch):
    if ARGS.model == 'QRNN':
        model_lexemes.reset()
    # Turn on evaluation mode which disables dropout.
    model_lexemes.eval()
    if ARGS.speller_mode != "none":
        model_invocab_speller.eval()
        model_oov_speller.eval()

    # If it's per-line we get all this stuff to do...
    # ... just maintain a separate simpler branch for it.
    if type(data_source) in [tuple, list]:
        with torch.no_grad():
            data_sources, sorted_lens, back_keys = data_source
            i_sorted_lens = 0
            sorted_losses = []
            for data_source in data_sources:
                batch_size = data_source.size(1)
                batch_losses = None
                hidden = model_lexemes.init_hidden(batch_size)
                for startidx in range(0, data_source.size(0) - 1, ARGS.bptt):
                    inputs, targets = get_batch(data_source, start_index=startidx, seq_len=ARGS.bptt)
                    unked_data = unked_tensor(inputs)
                    unked_targets = unked_tensor(targets)

                    output, hidden, _rnn_hs, dropped_rnn_hs = model_lexemes(unked_data, hidden, return_h=True)
                    output_flat = output.view(-1, ARGS.vocab_size)
                    lmlosses = torch.nn.functional.cross_entropy(output_flat, unked_targets, reduction = 'none')

                    # lmlosses = lmlosses.view_as(inputs)
                    # print(lmlosses)

                    if ARGS.open_vocab:
                        unklosses, _hidden_magnitudes, _in_vocab_probs, _in_vocab_maxes = unks_from_hidden_states(corpusdata = corpusdata, dropped_rnn_hs = dropped_rnn_hs, targets = targets, data_for_shape = inputs, force_prediction = True, model_oov_speller = model_oov_speller)
                        unk_mask = (targets >= ARGS.vocab_size)
                        if torch.sum(unk_mask) > 0:
                            lmlosses[unk_mask] += unklosses

                    # Now the new line-summing-and-masking part
                    lmlosses = lmlosses.view_as(inputs)
                    mask = (torch.arange(0, lmlosses.size(0)).unsqueeze(1) < (sorted_lens[i_sorted_lens:i_sorted_lens + batch_size] - 1).unsqueeze(0))
                    mask = mask.float().to(lmlosses.device)
                    new_losses = (lmlosses * mask).sum(dim = 0)
                    batch_losses = batch_losses + new_losses if batch_losses is not None else new_losses
                i_sorted_lens += batch_size
                sorted_losses.append(batch_losses)
            sorted_losses = torch.cat(sorted_losses)
            output_losses = torch.empty_like(sorted_losses)
            output_losses[back_keys] = sorted_losses
            print('\n'.join([str(l.item()) for l in output_losses]))
        print("Exiting now because I'm lazy.")
        exit(0)

    batch_size = data_source.size(1)

    # These are now summed losses for the whole epoch, no longer per token!
    total_loss = 0
    lm_only_loss = 0
    unk_only_loss = 0

    # Bins
    binlosses = [0.0] * (len(corpusdata.eval_binning_breakpoints) + 1)
    binelems = [0] * (len(corpusdata.eval_binning_breakpoints) + 1)
    binchars = [0] * (len(corpusdata.eval_binning_breakpoints) + 1)

    def get_bin(elem, breakpoints):
        binnr = 0
        while binnr < len(breakpoints) and elem >= breakpoints[binnr]:
            binnr += 1
        return binnr

    # for char2word loss calculation:
    # 1) these are kept across batches, so no word is cut in half and lost :)
    # 2) we need a separate counter for each batch "rail"!
    currentcharprefix = [""] * batch_size
    currentnatcount = [0.0] * batch_size
    if '-char' in ARGS.data:
        eossym_bin = get_bin(corpusdata.wordtraincount[data.EOSSYM], corpusdata.eval_binning_breakpoints)

    hidden_magnitudes = []
    in_vocab_probs = []
    in_vocab_maxes = []
    hidden = model_lexemes.init_hidden(batch_size)
    for startidx in range(0, data_source.size(0) - 1, ARGS.bptt):
        inputs, targets = get_batch(data_source, start_index=startidx, seq_len=ARGS.bptt)
        # print(data, targets)

        unked_data = unked_tensor(inputs)
        unked_targets = unked_tensor(targets)

        output, hidden, _rnn_hs, dropped_rnn_hs = model_lexemes(unked_data, hidden, return_h=True)

        if ARGS.open_vocab:
            unklosses, _hidden_magnitudes, _in_vocab_probs, _in_vocab_maxes = unks_from_hidden_states(corpusdata = corpusdata, dropped_rnn_hs = dropped_rnn_hs, targets = targets, data_for_shape = inputs, force_prediction = True, model_oov_speller = model_oov_speller)
            unkloss = torch.sum(unklosses).item()
            unk_only_loss += unkloss
            total_loss += unkloss
            if not isinstance(_hidden_magnitudes, list):
                hidden_magnitudes.append(_hidden_magnitudes.detach())
            if not isinstance(_in_vocab_probs, list):
                in_vocab_probs.append(_in_vocab_probs)
            if not isinstance(_in_vocab_maxes, list):
                in_vocab_maxes.append(_in_vocab_maxes)

        output_flat = output.view(-1, ARGS.vocab_size)
        lmlosses = torch.nn.functional.cross_entropy(output_flat, unked_targets, reduction = 'none').data
        lmloss = torch.sum(lmlosses).item()
        total_loss += lmloss
        lm_only_loss += lmloss
        hidden = repackage_hidden(hidden)

        # Add UNK losses to LM losses!
        if ARGS.open_vocab:
            unk_mask = (targets >= ARGS.vocab_size)
            if torch.sum(unk_mask) > 0:
                lmlosses[unk_mask] += unklosses

        # Sum words them for the bins
        if '-char' in ARGS.data:
            # We are on character level, so sum them into words to get total nats
            # Gotta do it seperately for each batch "rail"!
            for i_batchrail in range(batch_size):
                for i_outchar, idx in enumerate(targets.view_as(inputs).t()[i_batchrail]):
                    curchar = corpusdata.corpus.dictionary.idx2word[idx]
                    curloss = lmlosses.view_as(inputs).t()[i_batchrail][i_outchar].item()
                    # What delimits words? Well, the space-symbol, of course, but also the EOSSYM!
                    if curchar in ['⁀', data.EOSSYM]:
                        # If we have collected a word at all and not just encountered the next delimiter...
                        word = currentcharprefix[i_batchrail]
                        if word != '':
                            # What bin is the word we just finished in?
                            wordcount = corpusdata.wordtraincount[word] if word in corpusdata.wordtraincount else 0
                            i_bin = get_bin(wordcount, corpusdata.eval_binning_breakpoints)

                            # if i_batchrail == 0:
                            #     print("Rail", i_batchrail, "just completed »", word, "« at position", startidx + i_outchar, "with loss", currentnatcount[i_batchrail] + curloss, "(per char:", (currentnatcount[i_batchrail] + curloss) / len(word), ") and put it in bin", i_bin)

                            # Add loss to the bin
                            binlosses[i_bin] += currentnatcount[i_batchrail] + curloss
                            binchars[i_bin] += len(word) + (0 if curchar == data.EOSSYM else 1)  # spaces are part of the word, EOSSYM is its own word
                            binelems[i_bin] += 1

                        # Now, in the char setup the EOSSYM is its own word as well as a delimiter, so add that too:
                        if curchar == data.EOSSYM:
                            binlosses[eossym_bin] += curloss
                            binchars[eossym_bin] += 1
                            binelems[eossym_bin] += 1

                        # Reset counting
                        currentnatcount[i_batchrail] = 0.0
                        currentcharprefix[i_batchrail] = ''
                    else:
                        # Keep going
                        currentcharprefix[i_batchrail] += curchar
                        currentnatcount[i_batchrail] += curloss
        elif '-bpe' in ARGS.data:
            # We are on bpe word-part level, so sum them into words to get total nats
            # Gotta do it seperately for each batch "rail"!
            for i_batchrail in range(batch_size):
                for i_outpart, idx in enumerate(targets.view_as(inputs).t()[i_batchrail]):
                    curpart = corpusdata.corpus.dictionary.idx2word[idx]
                    curloss = lmlosses.view_as(inputs).t()[i_batchrail][i_outpart]
                    # What delimits words? Well, not ending in '@@'!
                    if curpart[-2:] != '@@':
                        word = currentcharprefix[i_batchrail] + curpart

                        # What bin is the word we just finished in?
                        wordcount = corpusdata.wordtraincount[word] if word in corpusdata.wordtraincount else 0
                        i_bin = get_bin(wordcount, corpusdata.eval_binning_breakpoints)

                        # Add loss to the bin
                        binlosses[i_bin] += currentnatcount[i_batchrail] + curloss
                        binchars[i_bin] += len(word) + 1
                        binelems[i_bin] += 1

                        # Reset counting
                        currentnatcount[i_batchrail] = 0.0
                        currentcharprefix[i_batchrail] = ''
                    else:
                        # Keep going with the word part
                        currentcharprefix[i_batchrail] += curpart[:-2]
                        currentnatcount[i_batchrail] += curloss
        else:
            # "Normal" word-level model
            # Map each target word to its training counts to get frequency-binned losses
            counts = corpusdata.corpus.idx2traincount[targets.cpu()]
            binmasks = []
            binmasks.append( counts <  corpusdata.eval_binning_breakpoints[0])  # noqa: E201,E222
            binmasks.append((counts >= corpusdata.eval_binning_breakpoints[0]) & (counts < corpusdata.eval_binning_breakpoints[1]))
            binmasks.append((counts >= corpusdata.eval_binning_breakpoints[1]) & (counts < corpusdata.eval_binning_breakpoints[2]))
            binmasks.append( counts >= corpusdata.eval_binning_breakpoints[2])  # noqa: E201

            # Now sum them up for each bin
            for i_bin, inds in enumerate(binmasks):
                if len(inds) > 0:
                    inds = inds.to(ARGS.device)
                    binlosses[i_bin] += torch.sum(lmlosses[inds]).item()
                    binelems[i_bin] += torch.sum(inds).item()
                    if torch.sum(inds) == 0:
                        continue
                    # Now get their lengths while filtering EOS, UNK, and the ilk (add these not_lexemes back in manually)
                    lens, inds_copy, _, _ = corpusdata.corpus.dictionary.build_wordtensors(vocab_indices = targets.data[inds], max_type_length = 9999999999)
                    not_lexemes = targets[inds].size(0) - len(inds_copy)
                    if len(inds_copy) == 0:
                        print("No actual words:", targets.data[inds])
                    binchars[i_bin] += torch.sum(lens).item() + not_lexemes

    # Normalize all losses per-token
    alltokens = torch.prod(torch.LongTensor(list(data_source.size()))).item()

    for i_bin, (loss, (elems, chars)) in enumerate(zip(binlosses, zip(binelems, binchars))):
        if elems == 0 and chars == 0:
            continue
        loss_per_elem = loss / elems if elems > 0 else 0
        loss_per_char = loss / chars if chars > 0 else 0
        lowerbound = corpusdata.eval_binning_breakpoints[i_bin - 1] if i_bin > 0 else 0
        upperbound = corpusdata.eval_binning_breakpoints[i_bin] if i_bin < len(corpusdata.eval_binning_breakpoints) else "infty"
        print("Bin", i_bin + 1, "( in [", lowerbound, ";", upperbound, "),", elems, "els,", chars, "chars) -> ", loss_per_elem, "nats/token; ", loss_per_char / math.log(2), "bpc")
        logscalar('eval/bin' + str(i_bin) + '/loss', loss_per_elem, global_step = i_epoch)

    embedding_lengths = (torch.sum((model_lexemes.encoder.weight ** 2), dim = -1) ** 0.5).view(-1)

    loghistogram('vectorlengths/h', (torch.sum((dropped_rnn_hs[-1] ** 2), dim = -1) ** 0.5).view(-1))
    loghistogram('vectorlengths/embs', embedding_lengths)
    loghistogram('vectorlengths/embUNK', embedding_lengths[[0]])
    if ARGS.open_vocab and hidden_magnitudes != []:
        loghistogram('vectorlengths/h_unk', torch.cat(hidden_magnitudes))
    if ARGS.open_vocab and in_vocab_probs != []:
        assert len(in_vocab_probs) == len(in_vocab_maxes)
        loghistogram('prob_unk_in_vocab/total', torch.cat(in_vocab_probs))
        loghistogram('prob_unk_in_vocab/max', torch.cat(in_vocab_maxes))
        with open('probs_unk_in_vocab.bs' + str(batch_size) + '.txt', 'w') as file:
            file.write("total\tmax\n")
            file.write('\n'.join([str(p) + '\t' + str(m) for (p, m) in zip(torch.cat(in_vocab_probs), torch.cat(in_vocab_maxes))]))

    # Print the largest embeddings (who is it?)
    print("Embedding sizes:")
    sortedpairs = sorted(list(zip(corpusdata.corpus.dictionary.idx2word, embedding_lengths)), key = lambda x: x[1], reverse = True)
    print(', '.join(["{} ({:.2f})".format(w, m.item()) for w, m in sortedpairs[:20]]))
    print('...')
    print(', '.join(["{} ({:.2f})".format(w, m.item()) for w, m in sortedpairs[-20:]]))

    return total_loss / alltokens, lm_only_loss / alltokens, unk_only_loss / alltokens



def eval_validation_and_save(*, corpusdata, epoch, epoch_start_time, stored_loss, model_lexemes, model_invocab_speller, model_oov_speller, savingmessage = 'Saving!'):
    val_loss, val_lm_only_loss, val_unk_only_loss = evaluate(corpusdata.val_data, corpusdata = corpusdata, model_lexemes = model_lexemes, model_invocab_speller = model_invocab_speller, model_oov_speller = model_oov_speller, i_epoch = epoch)

    if ARGS.speller_mode != "none" and corpusdata.devlexeme_indices is not None:
        devembs = torch.zeros((corpusdata.devlexeme_indices.size(0), ARGS.emsize), device = ARGS.device)

        # TODO print it under both OOV and invocab speller?
        devlexeme_losses = model_oov_speller(devembs, corpusdata.devlexeme_inputs, corpusdata.devlexeme_outputs, corpusdata.devlexeme_lengths)
        dev_speller_loss_per_lexeme = torch.sum(devlexeme_losses).item() / corpusdata.devlexeme_indices.size(0)
        # Normalize over train tokens (to compare to train speller_loss_per_token)
        dev_speller_loss_per_token = dev_speller_loss_per_lexeme * (corpusdata.trainlexeme_lengths.size(0) / (ARGS.tokmode_oversample if ARGS.speller_mode[-3:] == 'tok' else 1) if len(corpusdata.trainlexeme_lengths) > 0 else 1) * ARGS.speller_interval / (corpusdata.train_data.size(0) * corpusdata.train_data.size(1))
    else:
        dev_speller_loss_per_token = 0.0

    dev_bpc_on_original = val_loss * corpusdata.corpus.devwords / (math.log(2) * corpusdata.corpus.devchars * corpusdata.nchars_multiplier_dev)

    logscalar('dev/lm/loss', val_lm_only_loss, global_step = epoch)
    logscalar('dev/bpc/onoriginal', dev_bpc_on_original, global_step = epoch)
    if ARGS.speller_mode != "none":
        logscalar('dev/speller/loss', dev_speller_loss_per_token, global_step = epoch)
    if ARGS.open_vocab:
        logscalar('dev/unk/loss', val_unk_only_loss, global_step = epoch)
        logscalar('dev/lm+unk/loss', val_loss, global_step = epoch)

    print('-' * 173)
    print('| end of epoch {:3d} | time: {:5.2f}s | spell dev loss {:5.2f} | '
            'unk dev loss {:5.2f} | lm dev loss {:5.2f} | joint dev loss {:5.2f} ||| '
            'joint dev: ppl {:5.2f} | bpw {:6.3f} | bpc{} {:5.3f} |'.format(
                epoch, (time.time() - epoch_start_time), dev_speller_loss_per_token,
                val_unk_only_loss, val_lm_only_loss, val_loss,
                math.exp(val_loss), val_loss / math.log(2),
                ' (on original dataset)' if '-char' in ARGS.data or '-bpe' in ARGS.data else '',
                dev_bpc_on_original
            ))
    print('-' * 173)

    # TODO also print invocab speller?
    if ARGS.speller_mode != "none" and len(corpusdata.trainlexeme_lengths) > 0:
        printstring = "WANTED (train): " + str(corpusdata.wantedwords) + "\n"
        for temp in [0.1, 0.5, 1.0]:
            sampledwords = model_oov_speller(model_lexemes.encoder.weight[corpusdata.trainlexeme_indices[corpusdata.desired_idx_into_train]], corpusdata.trainlexeme_inputs[corpusdata.desired_idx_into_train], corpusdata.trainlexeme_outputs[corpusdata.desired_idx_into_train], corpusdata.trainlexeme_lengths[corpusdata.desired_idx_into_train], feed_samples = True, sampling_temp = temp)[1]
            printstring += "GOT (oovspeller, train, {:3.1f}):".format(temp) + str(sampledwords) + "\n"
        printstring = printstring[:-1]
        print(printstring)

        if BOARD is not None:
            BOARD.add_text('samples', printstring, global_step = epoch)

        # print(model_lexemes.encoder.weight[corpus.dictionary.word2idx['model'],desired_idx_into_train].unsqueeze(0))
    else:
        sampledwords = corpusdata.wantedwords

    if savingmessage is not None and val_loss < stored_loss:
        with open(ARGS.save, 'wb') as file:
            torch.save(model_lexemes, file)
        print(savingmessage)
        if ARGS.speller_mode != "none":
            if ARGS.speller_mode.startswith("sep-backoff"):
                with open(ARGS.save + '.charlm_invocab', 'wb') as file:
                    torch.save(model_invocab_speller.state_dict(), file)
                with open(ARGS.save + '.charlm_oov', 'wb') as file:
                    torch.save(model_oov_speller.state_dict(), file)
            else:
                with open(ARGS.save + '.charlm', 'wb') as file:
                    torch.save(model_oov_speller.state_dict(), file)

    return (val_loss, sampledwords)


# pylint: disable=too-many-locals
def final_eval(*, corpusdata, model_lexemes, model_invocab_speller, model_oov_speller, i_epoch):
    # Run numbers on dev and test data
    for (set_name, set_data, nwords, nchars) in [("dev", corpusdata.val_data, corpusdata.corpus.devwords, corpusdata.corpus.devchars * corpusdata.nchars_multiplier_dev), ("test", corpusdata.test_data, corpusdata.corpus.testwords, corpusdata.corpus.testchars * corpusdata.nchars_multiplier_test)]:
        set_loss, set_lm_only_loss, set_unk_only_loss = evaluate(set_data, corpusdata = corpusdata, model_lexemes = model_lexemes, model_invocab_speller = model_invocab_speller, model_oov_speller = model_oov_speller, i_epoch = i_epoch)
        print('-' * 173)
        print('| End of training | {} set | unk loss {:5.2f} | lm loss {:5.2f} | joint loss {:5.2f} ||| '
                'ppl {:5.2f} | total bits {:.2f} | bpw {:6.3f} | bpc (on original dataset) {:5.3f} |'.format(
                    set_name, set_unk_only_loss, set_lm_only_loss, set_loss,
                    math.exp(set_loss), set_loss * nwords / math.log(2), set_loss / math.log(2), set_loss * nwords / (math.log(2) * nchars)))
        print('-' * 173)

    # TODO also print invocab speller?
    if ARGS.speller_mode != "none" and len(corpusdata.trainlexeme_lengths) > 0:
        # Generate some random spellings
        print("First random spellings based on training words!")
        printstring = "WANTED (train): " + str(corpusdata.wantedwords) + "\n"
        for temp in [0.1, 0.5, 1.0]:
            sampledwords = model_oov_speller(model_lexemes.encoder.weight[corpusdata.trainlexeme_indices[corpusdata.desired_idx_into_train]], corpusdata.trainlexeme_inputs[corpusdata.desired_idx_into_train], corpusdata.trainlexeme_outputs[corpusdata.desired_idx_into_train], corpusdata.trainlexeme_lengths[corpusdata.desired_idx_into_train], feed_samples = True, sampling_temp = temp)[1]
            printstring += "GOT (oovspeller, train, {:3.1f}):".format(temp) + str(sampledwords) + "\n"

        # Now generate "interesting spellings"!
        interesting_lexeme_indices = []
        # for wordlist in ['phobias.txt', 'who_drugs.txt', 'hormones.txt', 'chemels.txt']:
        for wordlist in ['chemels.txt']:
            # print('\n'+wordlist)
            with open(wordlist) as file:
                desired = [word.lower() for word in file.read().splitlines()]
                for idx, word in enumerate(corpusdata.corpus.dictionary.idx2word):
                    if word.lower() in desired and idx < ARGS.vocab_size:  # we need to have embeddings for them!
                        interesting_lexeme_indices.append(idx)
        if len(interesting_lexeme_indices) > 0:
            interesting_lexeme_indices = torch.LongTensor(interesting_lexeme_indices, device = ARGS.device)
            printstring += "Now from chemical elements!\n"
            average_embedding = torch.sum(model_lexemes.encoder.weight.detach()[interesting_lexeme_indices], dim = 0)
            average_embedding /= torch.norm(average_embedding, p = 2)
            maxlen = 30
            to_generate = min(ARGS.speller_batchsize, 20)
            ins  = torch.full((to_generate, maxlen + 1), 0, dtype = torch.long, device = ARGS.device)  # 0 -> bow
            outs = torch.full((to_generate, maxlen + 1), 1, dtype = torch.long, device = ARGS.device)  # 1 -> eow
            lens = torch.full((to_generate,), maxlen + 1, dtype = torch.long, device = ARGS.device)
            for temp in [0.1, 0.5, 1.0]:
                sampledwords = model_oov_speller(average_embedding.unsqueeze(0).repeat(to_generate, 1), ins, outs, lens, feed_samples = True, sampling_temp = temp)[1]
                printstring += "OOV, Temp {:3.1f}:".format(temp) + str(sampledwords) + "\n"

        print(printstring)

    # Probe every 50th lexeme for the total probability of an in-vocab spelling given its embedding.
    if ARGS.speller_mode != "none" and ARGS.exclude_in_vocab_spellings_for_unk:
        total_ps = []
        max_ps = []
        for probed_lexeme in corpusdata.trainlexeme_indices[::500]:
            total_p, max_p = get_invocab_marginal_for_embedding(probed_embedding = model_lexemes.encoder.weight[probed_lexeme].detach(), corpusdata = corpusdata, model_marginal_speller = model_invocab_speller)
            total_ps.append(total_p)
            max_ps.append(max_p)
            if total_ps[-1] > 0.2:
                print("Lexeme", int(probed_lexeme), "(", corpusdata.corpus.dictionary.idx2word[probed_lexeme], ") has has total probability", total_ps[-1])
        zero_total_p, _ = get_invocab_marginal_for_embedding(probed_embedding = torch.zeros(ARGS.emsize, device = ARGS.device), corpusdata = corpusdata, model_marginal_speller = model_invocab_speller)
        print("0emb:", zero_total_p)
        with open('total_max_p.txt', 'w') as file:
            print("total\tmax", file = file)
            print('\n'.join([str(p) + '\t' + str(m) for p, m in zip(total_ps, max_ps)]), file = file)

    # Now sample actual sentences!
    if ARGS.sample_words > 0:
        for sampling_temp in [0.1, 0.75, 0.875, 1.0, 1.25]:
            print('Example sentence for T =', sampling_temp)
            idx0 = corpusdata.corpus.dictionary.word2idx[data.EOSSYM]
            hc0 = None
            # Sample words
            idxs, hc0, top_hs = model_lexemes.sample_sequence(idx0 = idx0, hc0 = hc0, sampling_temp = sampling_temp, n_items = ARGS.sample_words)
            # Sample UNK spellings
            unk_hs = [h.detach() for (idx, h) in zip(idxs, top_hs) if idx == 0]
            if unk_hs != []:
                unk_hs = torch.cat(unk_hs, dim = 1).squeeze(0)
                idx0 = idxs[-1]

                maxlen = 30
                to_generate = unk_hs.size(0)
                ins  = torch.full((to_generate, maxlen + 1), 0, dtype = torch.long, device = ARGS.device)  # 0 -> bow
                outs = torch.full((to_generate, maxlen + 1), 1, dtype = torch.long, device = ARGS.device)  # 1 -> eow
                lens = torch.full((to_generate,), maxlen + 1, dtype = torch.long, device = ARGS.device)
                sampledwords = model_oov_speller(unk_hs, ins, outs, lens, feed_samples = True, sampling_temp = sampling_temp)[1]
            else:
                sampledwords = []
            # Write them out
            sampledwords = iter(sampledwords)
            print(' '.join([corpusdata.corpus.dictionary.idx2word[idx] if idx != 0 else '\033[3m' + next(sampledwords) + '\033[23m' for idx, h in zip(idxs, top_hs)]))


def main():
    ###############################################################################
    # Load data
    ###############################################################################

    # First set vocab size
    corpus = data.Corpus(ARGS.data, ARGS.char_min_count, ARGS.bootstrap_with_seed, ARGS.bootstrap_training_data)
    ARGS.vocab_size = min(ARGS.vocab_size, corpus.vocab_size_train)  # len(corpus.dictionary.idx2word)
    print("Vocab size from", corpus.vocab_size_train, "/", corpus.vocab_size_train_and_dev, "/", len(corpus.dictionary.idx2word), "(train/+dev/+test) -->", ARGS.vocab_size)

    # print(corpus.devwords, corpus.devchars)
    # print(corpus.testwords, corpus.testchars)
    # exit(0)

    # If we are running in character level mode, we also want the word frequencies in the training set for the bpc binning, so load that now
    wordtraincount = None
    if '-char' in ARGS.data or '-bpe' in ARGS.data:
        word_corpus = data.Corpus(ARGS.data.replace('-char', '').replace('-bpe', ''), ARGS.char_min_count, ARGS.bootstrap_with_seed, ARGS.bootstrap_training_data)
        wordtraincount = {word_corpus.dictionary.idx2word[wordidx]: count for wordidx, count in enumerate(word_corpus.idx2traincount)}

    original_data = ARGS.data.replace('-char', '').replace('-bpe', '').replace('-tok', '')
    if ARGS.data != original_data:
        word_corpus = data.Corpus(original_data, ARGS.char_min_count, ARGS.bootstrap_with_seed, ARGS.bootstrap_training_data)
        nchars_multiplier_dev = word_corpus.devchars / corpus.devchars
        nchars_multiplier_test = word_corpus.testchars / corpus.testchars
    else:
        nchars_multiplier_dev = 1.0
        nchars_multiplier_test = 1.0

    # (then we're ready to write out the args)
    logtext('args', str(ARGS))

    # If EOS is OOV... things break.
    if corpus.dictionary.word2idx[data.EOSSYM] >= ARGS.vocab_size:
        raise Exception("EOS is an OOV (word id " + str(corpus.dictionary.word2idx[data.EOSSYM]) + ") - this breaks assumptions (namely that all UNKs shall be spelled out)")

    # Set the second bin breakpoint to the count at which OOVs start!
    if '-bpe' in ARGS.data or corpus.idx2traincount.size(0) <= ARGS.vocab_size + 1:
        eval_binning_breakpoints = (1, 1, 100)
    else:
        breakpoint = corpus.idx2traincount[ARGS.vocab_size + 1].item()
        eval_binning_breakpoints = (1, breakpoint, 100)

    # Batchify all data
    train_data = batchify(corpus.train, ARGS.batch_size).to(ARGS.device)
    val_data = batchify(corpus.valid, VALID_BATCH_SIZE).to(ARGS.device)
    if not ARGS.per_line:
        test_data = batchify(corpus.test, TEST_BATCH_SIZE).to(ARGS.device)
    else:
        # Make it a list so we can tell that this is to be evaluated line-wise
        test_data = list(sentence_batchify(corpus.test, ARGS.batch_size, corpus.dictionary.word2idx[data.EOSSYM]))
        test_data[0] = [b.to(ARGS.device) for b in test_data[0]]

    # Get lexeme (type) tensors
    if ARGS.speller_mode != "none":
        # trainvocab: lexemes for which we maintain embeddings
        trainvocab_indices = detshuffle(list(range(ARGS.vocab_size)))
        if ARGS.speller_mode[-3:] == 'typ':
            trainlexeme_lengths, trainlexeme_indices, trainlexeme_inputs, trainlexeme_outputs = \
                to_device(corpus.dictionary.build_wordtensors(vocab_indices = trainvocab_indices, max_type_length = ARGS.max_type_length))
        elif ARGS.speller_mode[-3:] == 'tok':
            # TODO can i resample these anew every epoch or so? then i wouldnt feel bad about the -tok baselines anymore!
            trainlexeme_lengths, trainlexeme_indices, trainlexeme_inputs, trainlexeme_outputs = \
                to_device(corpus.dictionary.build_wordtensors(vocab_indices = trainvocab_indices, max_type_length = ARGS.max_type_length,
                sample_by_frequency_total = ARGS.tokmode_oversample * (ARGS.vocab_size - 2), traincounts = corpus.idx2traincount))
        else:
            raise Exception("Unknown speller mode " + ARGS.speller_mode)
        # devvocab: a subset (only one batch!) of those for which we do not
        devvocab_indices = detshuffle(list(range(ARGS.vocab_size, min(corpus.vocab_size_train_and_dev, ARGS.vocab_size + ARGS.speller_batchsize))))
        if devvocab_indices == []:
            devlexeme_lengths = devlexeme_indices = devlexeme_inputs = devlexeme_outputs = None
        else:
            devlexeme_lengths, devlexeme_indices, devlexeme_inputs, devlexeme_outputs = \
                to_device(corpus.dictionary.build_wordtensors(vocab_indices = devvocab_indices, max_type_length = ARGS.max_type_length))

        # Need all wanted words to put the into the log file header
        n_sampled_words = 10
        desired_idx_into_train = (trainlexeme_indices.size(0) // n_sampled_words) * torch.arange(n_sampled_words, dtype = torch.long, device = ARGS.device) if len(trainlexeme_indices) > 0 else torch.LongTensor([], device = ARGS.device)
        wantedwords = [corpus.dictionary.idx2word[i] for i in trainlexeme_indices[desired_idx_into_train]] if len(trainlexeme_indices) > 0 else []

        # Make sure the UNKs are actually "generated from" UNK
        trainlexeme_indices[trainlexeme_indices >= ARGS.vocab_size] = corpus.dictionary.word2idx['<unk>']
    else:
        wantedwords = []

    corpusdata = namedtuple('CorpusData', [s + 'lexeme_' + x for s in ['train', 'dev'] for x in ['indices', 'inputs', 'outputs', 'lengths']] + ['train_data', 'val_data', 'test_data', 'eval_binning_breakpoints', 'wordtraincount', 'nchars_multiplier_dev', 'nchars_multiplier_test', 'desired_idx_into_train', 'wantedwords', 'corpus'])(
        train_data = train_data,
        val_data = val_data,
        test_data = test_data,
        trainlexeme_indices = trainlexeme_indices if ARGS.speller_mode != "none" else None,
        trainlexeme_inputs = trainlexeme_inputs if ARGS.speller_mode != "none" else None,
        trainlexeme_outputs = trainlexeme_outputs if ARGS.speller_mode != "none" else None,
        trainlexeme_lengths = trainlexeme_lengths if ARGS.speller_mode != "none" else None,
        devlexeme_indices = devlexeme_indices if ARGS.speller_mode != "none" else None,
        devlexeme_inputs = devlexeme_inputs if ARGS.speller_mode != "none" else None,
        devlexeme_outputs = devlexeme_outputs if ARGS.speller_mode != "none" else None,
        devlexeme_lengths = devlexeme_lengths if ARGS.speller_mode != "none" else None,
        eval_binning_breakpoints = eval_binning_breakpoints,
        wordtraincount = wordtraincount,
        nchars_multiplier_dev = nchars_multiplier_dev,
        nchars_multiplier_test = nchars_multiplier_test,
        desired_idx_into_train = desired_idx_into_train if ARGS.speller_mode != "none" else None,
        wantedwords = wantedwords if ARGS.speller_mode != "none" else [],
        corpus = corpus)

    ###############################################################################
    # Build the model
    ###############################################################################

    model_lexemes = model_module.RNNModel(ARGS.model, ARGS.vocab_size, ARGS.emsize, ARGS.nhid, ARGS.nlayers, dropout = ARGS.dropout, dropouth = ARGS.dropouth, dropouti = ARGS.dropouti, dropoute = ARGS.dropoute, wdrop = ARGS.wdrop, tie_weights = not ARGS.not_tied)
    model_lexemes = model_lexemes.to(ARGS.device)
    total_params = sum(len(x.view(-1)) for x in model_lexemes.parameters())
    print('Args:', ARGS)
    print('Model total parameters:', total_params)

    with open(ARGS.save + '.config', 'w', encoding = 'utf-8') as file:
        for key, val in vars(ARGS).items():
            print(key, val, sep = '\t', file = file)
        print("total_params", total_params, sep = '\t', file = file)

    with open(ARGS.save + '.losses', 'w', encoding = 'utf-8') as file:
        print("epoch", "LM train loss", "LM val loss", "speller train loss", *wantedwords, sep = '\t', file = file)

    if ARGS.speller_mode != "none":
        model_invocab_speller = ConditionedLSTM(
            vocab = corpus.dictionary.charset,
            token_dim = ARGS.speller_char_dim,
            conditioner_size = ARGS.emsize,
            hidden_units = ARGS.speller_hidden,
            dropout_p = ARGS.speller_dropout,
            conditioner_dropout_p = ARGS.speller_conditioner_dropout,
            feed_n_characters = ARGS.speller_feed_n_characters,
            num_layers = ARGS.speller_num_layers
        )

        if ARGS.speller_mode.startswith("sep-backoff"):
            model_oov_speller = ConditionedLSTM(
                vocab = corpus.dictionary.charset,
                token_dim = ARGS.speller_char_dim,
                conditioner_size = ARGS.emsize,
                hidden_units = ARGS.speller_hidden,
                dropout_p = ARGS.speller_dropout,
                conditioner_dropout_p = ARGS.speller_conditioner_dropout,
                feed_n_characters = ARGS.speller_feed_n_characters,
                num_layers = ARGS.speller_num_layers
            )
        else:
            model_oov_speller = model_invocab_speller

        # Set the conditioning input matrix to 0 if it wasn't wanted.
        if ARGS.speller_nuclear_regularization == float('inf'):
            model_invocab_speller.get_conditioner_inputweights()[0].zero_()
            model_oov_speller.get_conditioner_inputweights()[0].zero_()
        # Reset hidden-to-hidden and input-to-hidden if it wasn't wanted (i.e. unigram-only speller) through unsafe .data
        if ARGS.speller_mode[:5] == '1gram':
            model_invocab_speller.rnn.weight_hh_l0.data.zero_()
            model_invocab_speller.rnn.weight_ih_l0.data.zero_()
            model_oov_speller.rnn.weight_hh_l0.data.zero_()
            model_oov_speller.rnn.weight_ih_l0.data.zero_()

        model_invocab_speller = model_invocab_speller.to(ARGS.device)
        model_oov_speller = model_oov_speller.to(ARGS.device)

    else:
        model_invocab_speller = None
        model_oov_speller = None

    ###############################################################################
    # Run the training loop!
    ###############################################################################

    # Load the best saved model, if wanted
    if ARGS.resume:
        with open(ARGS.save, 'rb') as file:
            model_lexemes = torch.load(file)
        if ARGS.speller_mode != "none":
            if ARGS.speller_mode.startswith("sep-backoff"):
                with open(ARGS.save + '.charlm_invocab', 'rb') as file:
                    model_invocab_speller.load_state_dict(torch.load(file))
                with open(ARGS.save + '.charlm_oov', 'rb') as file:
                    model_oov_speller.load_state_dict(torch.load(file))
            else:
                with open(ARGS.save + '.charlm', 'rb') as file:
                    model_invocab_speller.load_state_dict(torch.load(file))
                model_oov_speller = model_invocab_speller

    # Epoch 0!
    epoch = 0

    # At any point you can hit Ctrl + C to break out of training early.
    best_val_losses = []
    stored_loss = 100000000
    try:
        optimizer_lm = None
        if ARGS.optimizer == 'sgd':
            optimizer_lm = torch.optim.SGD(model_lexemes.parameters(), lr=ARGS.lr_lm, weight_decay=ARGS.wdecay)
        if ARGS.optimizer == 'adam':
            optimizer_lm = torch.optim.Adam(model_lexemes.parameters(), lr=ARGS.lr_lm, weight_decay=ARGS.wdecay)

        if ARGS.speller_mode != "none":
            optimizer_speller = torch.optim.SGD(list(set(list(model_invocab_speller.parameters()) + list(model_oov_speller.parameters()))), lr = ARGS.lr_speller, weight_decay = ARGS.speller_wdecay)
        else:
            optimizer_speller = None
        for epoch in range(1, ARGS.epochs + 1):
            epoch_start_time = time.time()
            train_loss, speller_loss = train_for_epoch(i_epoch = epoch, corpusdata = corpusdata, model_lexemes = model_lexemes, model_invocab_speller = model_invocab_speller, model_oov_speller = model_oov_speller, optimizer_lm = optimizer_lm, optimizer_speller = optimizer_speller)
            if 't0' in optimizer_lm.param_groups[0]:
                tmp = {}
                for prm in model_lexemes.parameters():
                    tmp[prm] = prm.data.clone()
                    prm.data = optimizer_lm.state[prm]['ax'].clone()

                with torch.no_grad():
                    val_loss, sampledwords = eval_validation_and_save(epoch = epoch, epoch_start_time = epoch_start_time, stored_loss = stored_loss, corpusdata = corpusdata, model_lexemes = model_lexemes, model_invocab_speller = model_invocab_speller, model_oov_speller = model_oov_speller, savingmessage = 'Saving Averaged!')
                stored_loss = min(stored_loss, val_loss)

                for prm in model_lexemes.parameters():
                    prm.data = tmp[prm].clone()

            else:
                with torch.no_grad():
                    val_loss, sampledwords = eval_validation_and_save(epoch = epoch, epoch_start_time = epoch_start_time, stored_loss = stored_loss, corpusdata = corpusdata, model_lexemes = model_lexemes, model_invocab_speller = model_invocab_speller, model_oov_speller = model_oov_speller, savingmessage = 'Saving Normal!')
                stored_loss = min(stored_loss, val_loss)

                if ARGS.optimizer == 'sgd' and 't0' not in optimizer_lm.param_groups[0] and (len(best_val_losses) > ARGS.nonmono and val_loss > min(best_val_losses[:-ARGS.nonmono])):
                    print('Switching to ASGD!')
                    optimizer_lm = torch.optim.ASGD(model_lexemes.parameters(), lr=ARGS.lr_lm, t0=0, lambd=0., weight_decay=ARGS.wdecay)
                    # optimizer_lm.param_groups[0]['lr'] /= 2.
                best_val_losses.append(val_loss)

            if epoch % 100 == 0:
                if 't0' in optimizer_lm.param_groups[0]:
                    for prm in model_lexemes.parameters():
                        tmp[prm] = prm.data.clone()
                        prm.data = optimizer_lm.state[prm]['ax'].clone()
                with torch.no_grad():
                    final_eval(corpusdata = corpusdata, model_lexemes = model_lexemes, model_invocab_speller = model_invocab_speller, model_oov_speller = model_oov_speller, i_epoch = epoch)
                if 't0' in optimizer_lm.param_groups[0]:
                    for prm in model_lexemes.parameters():
                        prm.data = tmp[prm].clone()

            if epoch in ARGS.when:
                print('Dividing learning rate by 10')
                optimizer_lm.param_groups[0]['lr'] /= 10.

            with open(ARGS.save + '.losses', 'a', encoding = 'utf-8') as file:
                print(epoch, train_loss, val_loss, speller_loss, *sampledwords, sep = '\t', file = file)

    except KeyboardInterrupt:
        print('-' * 89)
        print('Exiting from training early')

    # Load the best saved model.
    with open(ARGS.save, 'rb') as file:
        model_lexemes = torch.load(file)
    if ARGS.speller_mode != "none":
        if ARGS.speller_mode.startswith("sep-backoff"):
            with open(ARGS.save + '.charlm_invocab', 'rb') as file:
                model_invocab_speller.load_state_dict(torch.load(file))
            with open(ARGS.save + '.charlm_oov', 'rb') as file:
                model_oov_speller.load_state_dict(torch.load(file))
        else:
            with open(ARGS.save + '.charlm', 'rb') as file:
                model_invocab_speller.load_state_dict(torch.load(file))
            model_oov_speller = model_invocab_speller

    with torch.no_grad():
        final_eval(corpusdata = corpusdata, model_lexemes = model_lexemes, model_invocab_speller = model_invocab_speller, model_oov_speller = model_oov_speller, i_epoch = epoch)


if __name__ == "__main__":
    main()
