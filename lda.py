from gensim import corpora, models
import gensim

import sys
import os

global family_count

def classify(train_filename,test_filename):

	train_result = train(train_filename)
	ldamodel = train_result['ldamodel']
	topic_results_train = train_result['train_topics']
	train_family_list = train_result['family_list']
	topic_dict = getFamilyTopicPairs(topic_results_train, train_family_list)

	test_result = test(test_filename,ldamodel)

	topic_results_test = test_result['test_topics']
	test_family_list = test_result['family_list']
	predicted_labels = getPredictedLabels(topic_results_test, topic_dict)

	result = {}
	result['reallabels'] = test_family_list
	result['predictedlabels'] = predicted_labels

	result['test_topics'] = topic_results_test
	result['train_topics'] = topic_results_train
	result['train_family_list'] = train_family_list
	result['test_family_list'] = test_family_list
	result['top_topics'] = topic_dict.keys()


	return result


def test(test_filename,ldamodel):
	test_file = open(test_filename, encoding="utf-8")
	test_data = test_file.readlines()
	test_kmers = []
	family_list = []
	data_count = 0

	for prot in test_data:

		data_count = data_count + 1
		
		if data_count < 2:
			continue
		if data_count > 20 * 100:
			break
		
		prot = prot.replace('"','')
		prot = prot.strip()
		info = prot.split(',')
		family_list.append(info[1])
		kmers = info[3:]
		test_kmers.append(kmers)

	
	dictionary = corpora.Dictionary(test_kmers)

	corpus = [dictionary.doc2bow(text) for text in test_kmers]

	

	result_dict = {}
	
	doc_lda = ldamodel[corpus]

	result_dict['test_topics'] = doc_lda
	result_dict['family_list'] = family_list

	return result_dict


def train(train_filename):
	train_file = open(train_filename, encoding="utf-8")
	train_data = train_file.readlines()
	train_kmers = []
	family_list = []
	data_count = 0

	for prot in train_data:

		data_count = data_count + 1

		if data_count < 2:
			continue
		if data_count > 20 * 100 * 4:
			break
		
		prot = prot.replace('"','')
		prot = prot.strip()
		info = prot.split(',')
		family_list.append(info[1])
		kmers = info[3:]
		train_kmers.append(kmers)

	dictionary = corpora.Dictionary(train_kmers)

	corpus = [dictionary.doc2bow(text) for text in train_kmers]

	ldamodel = gensim.models.ldamodel.LdaModel(corpus, num_topics= 40, id2word = dictionary, passes=1)

	result_dict = {}
	result_dict['ldamodel'] = ldamodel

	doc_lda = ldamodel[corpus]

	result_dict['train_topics'] = doc_lda
	result_dict['family_list'] = family_list

	return result_dict



def getFamilyTopicPairs(topicResults, familyList):
	topic_family_dict = {}
	intermediate_dict = {}
	index = 0
	for item in topicResults:
		family_name = familyList[index]
		max_topic =  max(item, key=lambda x: x[1])
		if max_topic[0] in intermediate_dict:
			temp_dict = intermediate_dict[max_topic[0]]
			if family_name in temp_dict:
				score = temp_dict[family_name]
				score = score + 1
				temp_dict[family_name] = score
			else:
				temp_dict[family_name] = 1

			intermediate_dict[max_topic[0]] = temp_dict
		else:
			temp_dict = {}
			temp_dict[family_name] = 1
			intermediate_dict[max_topic[0]] = temp_dict

		index = index + 1

	for topicKey in intermediate_dict.keys():
		topicValue = intermediate_dict[topicKey]
		top_scored_family = max(topicValue.keys(), key=(lambda key: topicValue[key]))
		topic_family_dict[topicKey] = top_scored_family

	return topic_family_dict

def getPredictedLabels(topicResults, topicDict):
	predictedLabels = []
	for item in topicResults:
		max_topic =  max(item, key=lambda x: x[1])
		if max_topic[0] in topicDict:
			predictedLabels.append(topicDict[max_topic[0]])
		else:
			predictedLabels.append(-1)

	return predictedLabels