from gensim import corpora, models
import gensim

import sys
import os


def train_lda(train_filename):
	train_file = open(train_filename, encoding="utf-8")
	train_data = train_file.readlines()
	train_kmers = []
	for prot in train_data:
		print(prot)
		info = prot.split()
		kmers = info[2:]
		train_kmers.append(kmers)

	print(len(train_kmers))
	print(train_kmers[2])
	dictionary = corpora.Dictionary(train_kmers)

	corpus = [dictionary.doc2bow(text) for text in train_kmers]

	print(len(corpus))
	ldamodel = gensim.models.ldamodel.LdaModel(corpus, num_topics=10, id2word = dictionary, passes=20) #topic number selection??? according to family?

	result_dict = {}
	result_dict['ldamodel'] = ldamodel

	doc_lda = ldamodel[corpus]

	result_dict['train_topics'] = doc_lda

	return result_dict