import nltk
import pandas as pd
from collections import Counter, defaultdict
import numpy as np
from constants import OFFSET

def tokenize_and_downcase(string,vocab=None):
    """for a given string, corresponding to a document:
    - tokenize first by sentences and then by word
    - downcase each token
    - return a Counter of tokens and frequencies.

    :param string: input document
    :returns: counter of tokens and frequencies
    :rtype: Counter

    """
    bow = Counter()

    # If we have a vocabulary, add all words to counter
    if vocab:
        for token in vocab:
            downcase = token.lower()
            bow[downcase] = 0

    sentences = nltk.sent_tokenize(string)
    for sentence in sentences:
        words = nltk.word_tokenize(sentence)
        for word in words:
            downcase = word.lower()
            bow[downcase] += 1

    return bow


### Helper code

def read_data(csvfile,labelname,preprocessor=lambda x : x):
    # note that use of utf-8 encoding to read the file
    df = pd.read_csv(csvfile,encoding='utf-8')
    return df[labelname].values,[preprocessor(string) for string in df['text'].values]


def get_vocab_counts(x_tr, x_te):
	counts = Counter()
	for bows in [x_tr, x_te]:
		for bow in bows:
			for k, v in bow.iteritems():
				counts[k] += v

	return counts


def create_vocab(x_tr, x_te):
	"""
	input: training data, a list of bag-of-words (python Counter objects)
	output: the set of all possible tokens (vocab) in our input vectors
	"""

	vocab = set()
	for bows in [x_tr, x_te]:
		for bow in bows:
			# add all tokens in bow to the vocab set
			vocab.update(bow.iterkeys())


	vocab.add(OFFSET)  # this should improve accuracies later
	return vocab


def create_mapping(vocab, K):
	"""
	input: vocab, K = size of vocab
	output: a defaultdict mapping distinct tokens (keys) to indices in a feature array
	"""

	# use a defaultdict to prevent crashing code if we somehow try to get a
	# mapping for a word not in the original vocab

	mapping_temp = {token : idx for idx, token in enumerate(vocab)}
	# any word not in vocab maps to Kth index
	mapping = defaultdict(lambda: K, mapping_temp)

	return mapping


def feature_vector(bow, vocab, K, mapping):
	vec_size = len(vocab)
	vec = np.zeros(vec_size)
	for word, count in bow.iteritems():
		idx = mapping[word]
		if idx != K:
			vec[idx] += count

	return vec
