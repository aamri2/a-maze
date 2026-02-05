import os
import torch
import pickle



EOS = "<EOS>"

def load_vocab_with_eos(path, eos):
    with open(os.path.join(path, "segm_filtered.vocab"), "rb") as fi:
        vocab = pickle.load(fi)

    assert EOS not in vocab["tok2idx"]
    vocab["tok2idx"][EOS] = len(vocab["idx2tok"])
    vocab["idx2tok"].append(EOS)

    return vocab


class Corpus(object):
    def __init__(self, path):
        self.vocab = load_vocab_with_eos(path, EOS)
        # self.train = self.tokenize(os.path.join(path, 'train.txt'))
        # self.valid = self.tokenize(os.path.join(path, 'dev.txt'))
        # self.test = self.tokenize(os.path.join(path, 'test.txt'))

    def tokenize(self, path):
        """Tokenizes a text file."""
        assert os.path.exists(path)

        eos_idx = self.vocab["tok2idx"][EOS]

        # Tokenize file content
        with open(path, 'r', encoding="utf8") as f:
            idss = []
            for line in f:
                ids = list(map(int, line.strip().split()))
                ids.append(eos_idx)
                idss.append(torch.tensor(ids).type(torch.int64))
            ids = torch.cat(idss)

        return ids

    def tokenize_strs(self, words):
        """Tokenizes a list of strings."""

        eos_idx = self.vocab["tok2idx"][EOS]

        # Tokenize file content
        idss = []
        for word in words:
            ids = list(map(int, word.strip().split()))
            ids.append(eos_idx)
            idss.append(torch.tensor(ids).type(torch.int64))
        ids = torch.cat(idss)

        return ids