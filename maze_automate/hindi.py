import torch
import logging
from hindi_code import data
from lang_model import lang_model
import utils
import sys  # because pickle is dumb and can't deal with things in other locations
import string
from wordfreq_distractor import get_frequency

sys.path.insert(0, './hindi_code')


class hindi_model(lang_model):
    """Wrapping class for hindi model"""

    def __init__(self):
        """Do whatever set-up it takes"""
        with open("hindi_data/fortyepochs.pt", 'rb') as f:
            print("Loading the model")
            # to convert model trained on cuda to cpu model
            self.model = torch.load(f, map_location=lambda storage, loc: storage, weights_only=False)
        self.model.eval()  # put model in eval mode
        self.model.cpu()  # put it on cpu
        self.device = torch.device("cpu")  # what sort of data model is stored on
        self.dictionary = data.Corpus("hindi_code/vocabdata")  # load a dictionary
        self.ntokens = self.dictionary.vocab['idx2tok'].__len__()  # length of dictionary

    def tokenize(self, word):
        """ returns a list of tokens according to the models desired tokenization scheme"""
        words = word.split()
        for i, word in enumerate(words):
            words[i] = word.strip(string.punctuation + '।')
        return words

    def empty_sentence(self):
        """Initialize a new sentence -- starter hidden state etc"""
        hidden = self.model.init_hidden(1)  # sets initial values on hidden layer
        return hidden

    def update(self, hidden, word):
        """Given the model representation (=hidden state) and the next word (not tokenized)
        returns new hidden state (at end of adding word)
        and probability distribution of next words at end of addition"""
        input_word = torch.randint(self.ntokens, (1, 1), dtype=torch.long).to(self.device)  # make a word placeholder
        parts = self.tokenize(word)  # get list of tokens
        for part in parts:
            if part not in self.dictionary.vocab['tok2idx']:
                logging.warning('%s is not in the Gulordava model vocabulary.', part)
            token = self.dictionary.vocab['tok2idx'][part]  # get id of token
            input_word.fill_(torch.tensor(token))  # fill with value of token
            output, hidden = self.model(input_word, hidden)  # do the model thing
            word_weights = output.squeeze().div(1.0).exp().cpu()  # process output into weights
            word_surprisals = -1 * torch.log2(word_weights / sum(word_weights))  # turn into surprisals
        return hidden, word_surprisals

    def get_surprisal(self, surprisals, word):
        """Given a probability distribution, and a word
        Return its surprisal (bits), or use something as unknown code"""
        word = self.tokenize(word)[0]
        token = self.dictionary.vocab['tok2idx'][word]
        return surprisals[token].item()  # numeric value of word's surprisal



def get_thresholds(words, lang='hi'):
    """given words, returns min and max length to use"""
    lengths = []
    freqs = []
    for word in words:
        stripped = word.strip(string.punctuation + '।')
        lengths.append(len(stripped))
        freqs.append(get_frequency(stripped, lang=lang))
    min_length = min(min(lengths), 15)
    max_length = max(max(lengths), 4)
    min_freq = min(min(freqs), 11)
    max_freq = max(max(freqs), 3)
    return min_length, max_length, min_freq, max_freq
