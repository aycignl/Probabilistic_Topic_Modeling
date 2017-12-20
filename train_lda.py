from nltk.tokenize import RegexpTokenizer
from stop_words import get_stop_words
from nltk.stem.porter import PorterStemmer
from gensim import corpora, models
import gensim

import sys
import os


def train_lda(train_filename):
	train_file = open(train_filename, encoding="utf-8")
	train_data = train_file.readlines()
	train_kmers = []
	for prot in train_data:
		info = prot.split(",")
		kmers = info[2:]
		train_kmers.append(kmer)

	dictionary = corpora.Dictionary(train_kmers)
	corpus = [dictionary.doc2bow(text) for text in train_kmers]

	ldamodel = gensim.models.ldamodel.LdaModel(corpus, num_topics=10000, id2word = dictionary, passes=20) #topic number selection???


