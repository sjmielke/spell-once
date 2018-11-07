import gzip
import os
from collections import Counter

import torch

# For fairness of char-level eval choose the pilcrow as EOS symbol instead (only 1 char, not occuring in data)
# Caution: this is assumed to be a character for the loss summing of characters into words in main.evaluate
# EOSSYM = '<eos>'
EOSSYM = '¶'
UNKCHAR = '◊'


# Transparent compression handling: pass the uncompressed filename, it will
# append '.gz' automagically and try to open. On failure, open the uncompressed file.
def openfile(filename, *args, **kwargs):
    try:
        return gzip.open(filename + '.gz', *args, **kwargs)
    except FileNotFoundError:
        return open(filename, *args, **kwargs)


class Dictionary(object):
    def __init__(self):
        self.word2idx = {'<unk>': 0}
        self.idx2word = ['<unk>']
        self.counter = Counter()
        self.total = 1
        self.charset = ['<bow>', '<eow>']  # ♯

    def add_word(self, word):
        if word not in self.word2idx:
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word) - 1
            self.charset += sorted(list(set(word) - set(self.charset)))
        token_id = self.word2idx[word]
        self.counter[token_id] += 1
        self.total += 1
        return self.word2idx[word]

    def sort_vocabulary(self):
        eos_id = self.word2idx[EOSSYM]
        new2old = [0, eos_id] + [i for i, _ in self.counter.most_common() if i != 0 and i != eos_id]

        sorted_idx2word = [self.idx2word[old] for old in new2old]
        sorted_word2idx = {w: i for (i, w) in enumerate(sorted_idx2word)}
        sorted_counter = Counter({i: self.counter[new2old[i]] for i in range(len(sorted_idx2word))})

        self.idx2word = sorted_idx2word
        self.word2idx = sorted_word2idx
        self.counter = sorted_counter

    # pylint: disable=too-many-locals
    def build_wordtensors(self, vocab_indices, *, exclude = (EOSSYM, '<unk>'), max_type_length = 99999999999, traincounts = None, sample_by_frequency_total = None):
        for _e in exclude:
            assert _e in self.word2idx

        if traincounts is not None and sample_by_frequency_total is not None:
            indices_and_words = []
            vocab_indices = set(vocab_indices)
            while len(indices_and_words) < sample_by_frequency_total:
                indices = torch.multinomial(traincounts.float(), sample_by_frequency_total + 10000, replacement = True)  # generate a couple more, to be safe
                indices_and_words = [(i.item(), self.idx2word[i.item()]) for i in indices if i.item() in vocab_indices]  # inefficient, but easy
                indices_and_words = [(i, w) for (i, w) in indices_and_words if w not in exclude and len(w) <= max_type_length]
            indices_and_words = indices_and_words[:sample_by_frequency_total]
        else:
            assert(traincounts is None and sample_by_frequency_total is None)

            indices_and_words = [(i, self.idx2word[i]) for i in vocab_indices]
            indices_and_words = [(i, w) for (i, w) in indices_and_words if w not in exclude and len(w) <= max_type_length]

        # def detsh(w):
        #     import random
        #     wl = list(w)
        #     random.seed(w)
        #     random.shuffle(wl)
        #     return ''.join(wl)

        words = [w for (_, w) in indices_and_words]
        indices = torch.LongTensor([i for (i, _) in indices_and_words])
        lengths = torch.LongTensor([len(w) + 1 for w in words])  # plus BOW or EOW
        maxlen = int(torch.max(lengths)) if len(lengths) > 0 else 0

        # Build lookup table
        char2idx = {c: i for (i, c) in enumerate(self.charset)}

        # Construct tensors with indices into charset
        bow, eow = char2idx['<bow>'], char2idx['<eow>']
        inputs =  torch.LongTensor([[bow] + [char2idx[w[i]] if i < len(w) else eow for i in range(maxlen - 1)] for w in words])  # noqa: E222
        outputs = torch.LongTensor([        [char2idx[w[i]] if i < len(w) else eow for i in range(maxlen    )] for w in words])  # noqa: E201,E202,E221

        # print(lengths)
        # print(inputs)
        # print(outputs)
        # print(torch.max(lengths))
        # print(torch.sum(inputs, dim = 0).type(torch.LongTensor))
        # print(torch.sum(outputs, dim = 0).type(torch.LongTensor))

        return (lengths, indices, inputs, outputs)

    def __len__(self):
        return len(self.idx2word)


class Corpus(object):
    def __init__(self, path, min_char_count, bootstrap_with_seed = None, bootstrap_training_data = False):
        self.dictionary = Dictionary()

        # Read in strings, possibly doing bootstrap
        strings = {}
        for section in ['train', 'valid', 'test']:
            with openfile(os.path.join(path, section + '.txt'), 'rt', encoding = 'utf-8') as file:
                strings[section] = file.read()

                if bootstrap_with_seed is not None and (section != 'train' or bootstrap_training_data):
                    # Split WikiText-2 articles
                    assert 'wikitext-2' in path
                    import random
                    import re
                    boundaries = [match.start() for match in re.finditer(r'( \n |\n\n)= [^=]', strings[section])] + [len(strings[section])]
                    articles = [strings[section][boundaries[i]:boundaries[i+1]] for i in range(len(boundaries) - 1)]
                    random.seed(bootstrap_with_seed)
                    strings[section] = ''.join(random.choices(articles, k = len(articles)))

        # First make decision about character vocab
        self.charset = self.get_charset(strings['train'], min_char_count)

        # Only include training vocab for reordering and making UNKing decision
        self.read_vocab(strings['train'])
        self.dictionary.sort_vocabulary()
        self.traincounter = Counter(self.dictionary.counter)  # is later turned into matrix

        # Add all for BPE baseline
        if '-bpe' in path:
            with openfile(os.path.join(path, 'codes'), 'rt', encoding = 'utf-8') as file:
                vocab = set(self.charset) | set([c + '@@' for c in self.charset])
                for join in [l.replace(' ', '') for l in file.read().splitlines()]:
                    if join[-4:] == '</w>':
                        vocab.add(join[:-4])
                    else:
                        vocab.add(join)
                        vocab.add(join + '@@')
                for word in vocab:
                    if word not in self.dictionary.word2idx:
                        self.dictionary.word2idx[word] = len(self.dictionary.idx2word)
                        self.dictionary.counter[len(self.dictionary.idx2word)] = 0
                        self.dictionary.idx2word.append(word)
                        self.dictionary.charset += sorted(list(set(word) - set(self.dictionary.charset)))

        # Estimate these distributions for the baselines
        ttyp = 0
        ttok = 0
        ttyplen = 0
        ttoklen = 0
        self.typcharcounter = {count: 0 for count in list(self.charset) + [UNKCHAR]}
        self.tokcharcounter = {count: 0 for count in list(self.charset) + [UNKCHAR]}
        for typ_i, count in self.dictionary.counter.items():
            ttyp += 1
            ttok += count
            word = self.dictionary.idx2word[typ_i]
            ttyplen += len(word)
            ttoklen += len(word) * count
            for char in word:
                if char not in self.charset:
                    char = UNKCHAR
                self.typcharcounter[char] += 1
                self.tokcharcounter[char] += 1

        self.avg_typlen = ttyplen / ttyp
        self.avg_toklen = ttoklen / ttok
        # print(self.avg_typlen, self.avg_toklen)
        # print(self.typcharcounter, self.tokcharcounter)
        # exit(0)

        # Then read in rest and count characters
        self.vocab_size_train = len(self.dictionary.idx2word)
        self.devchars, self.devwords = self.read_vocab(strings['valid'])
        self.vocab_size_train_and_dev = len(self.dictionary.idx2word)
        self.testchars, self.testwords = self.read_vocab(strings['test'])

        # Now matrixize the counter
        self.idx2traincount = torch.zeros(len(self.dictionary)).long()
        for (typ_i, count) in self.traincounter.items():
            self.idx2traincount[typ_i] = count

        self.train = self.tokenize(strings['train'])
        self.valid = self.tokenize(strings['valid'])
        self.test  = self.tokenize(strings['test'])  # noqa: E221

        if '-bpe' in path:
            assert [self.dictionary.idx2word[i] for i in torch.cat([self.train[self.train >= self.vocab_size_train], self.valid[self.valid >= self.vocab_size_train], self.test[self.test >= self.vocab_size_train]])] == []

    # pylint: disable=no-self-use
    def get_charset(self, string, min_char_count):
        charcounter = Counter(string)
        charset = [char for (char, count) in charcounter.most_common() if count >= min_char_count]
        # assert all((char not in charset for char in EOSSYM + UNKCHAR))
        print("Using", len(charset), "of", len(charcounter), "chars (plus", EOSSYM, "as EOS, " + UNKCHAR + " as UNK, and <bow>/<eow> 'chars'):", "".join(charset))
        # print(charset)
        charset = sorted(list(set(charset + ['\n', ' '] + [char for char in EOSSYM + UNKCHAR])))

        return charset

    def read_vocab(self, string):
        # Add words to the dictionary
        totalchars = 0
        tokens = 0
        for line in string.splitlines():
            unked_line = "".join((char if char in self.charset else '◊' for char in line))
            totalchars += len(unked_line) + 1
            words = unked_line.split() + [EOSSYM]
            tokens += len(words)
            for word in words:
                self.dictionary.add_word(word)
        return totalchars, tokens

    def tokenize(self, string):
        """Tokenizes a text file."""
        # Count for allocation
        tokens = sum([len("".join((c if c in self.charset else '◊' for c in line)).split()) + 1 for line in string.splitlines()])

        # Tokenize file content
        ids = torch.LongTensor(tokens)
        token_i = 0
        for line in string.splitlines():
            line = "".join((c if c in self.charset else '◊' for c in line))
            words = line.split() + [EOSSYM]
            for word in words:
                ids[token_i] = self.dictionary.word2idx[word] if word in self.dictionary.word2idx else 0
                token_i += 1
        return ids
