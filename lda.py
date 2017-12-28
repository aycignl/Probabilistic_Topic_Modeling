from gensim import corpora, models
import gensim

import sys
import os

global family_count

#This class trains and perform classifying using lda
#the most probable topic is assigned to each sequence
#the most frequent topic in a family is assigned to families
def classify(train_filename,test_filename):
	#train the lda model and get topic probabilities for sequences
	train_result = train(train_filename)
	ldamodel = train_result['ldamodel']
	topic_results_train = train_result['train_topics']
	train_family_list = train_result['family_list']
	train_kmers = train_result['train_kmers']
	#match topics with families
	topic_dict = getFamilyTopicPairs(topic_results_train, train_family_list)

	#get topic probabilities for test data with the lda model
	test_result = test(test_filename,ldamodel)
	test_kmers = test_result['test_kmers']

	topic_results_test = test_result['test_topics']
	test_family_list = test_result['family_list']
	#get prediction using the topic-family dictionary
	predicted_labels = getPredictedLabels(topic_results_test, topic_dict)

	result = {}
	result['reallabels'] = test_family_list
	result['predictedlabels'] = predicted_labels

	result['test_topics'] = topic_results_test
	result['train_topics'] = topic_results_train
	result['train_family_list'] = train_family_list
	result['test_family_list'] = test_family_list
	result['top_topics'] = topic_dict.keys()
	result['train_kmers'] = train_kmers
	result['test_kmers'] = test_kmers

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
		if data_count > 20 * 100: #write number of families you want to test as the first item in product e.g. 20
			break
		
		prot = prot.replace('"','')
		prot = prot.strip()
		info = prot.split(',')
		family_list.append(info[1])
		kmers = info[3:]
		test_kmers.append(kmers)

	#create dictonary from the test_kmers so that we can create the corpus.
	dictionary = corpora.Dictionary(test_kmers)

	corpus = [dictionary.doc2bow(text) for text in test_kmers]

	

	result_dict = {}
	#get topic probability distribution of the test data
	doc_lda = ldamodel[corpus]

	result_dict['test_topics'] = doc_lda
	result_dict['family_list'] = family_list
	result_dict['test_kmers'] = test_kmers

	return result_dict


def train(train_filename):
	train_file = open(train_filename, encoding="utf-8")
	train_data = train_file.readlines()
	train_kmers = []
	family_list = []
	data_count = 0

	for prot in train_data:

		data_count = data_count + 1

		#if data_count < 2:
		#	continue
		if data_count > 20 * 100 * 4: #write number of families you want to test as the first item in product e.g. 20
			break
		
		prot = prot.replace('"','')
		prot = prot.strip()
		info = prot.split(',')
		family_list.append(info[1])
		kmers = info[3:]
		train_kmers.append(kmers)

	dictionary = corpora.Dictionary(train_kmers)

	corpus = [dictionary.doc2bow(text) for text in train_kmers]

	#create lda model with the corpus we created by the kmers
	ldamodel = gensim.models.ldamodel.LdaModel(corpus, num_topics= 100, id2word = dictionary, passes=1)

	result_dict = {}
	result_dict['ldamodel'] = ldamodel

	#get topic probabilities for the train data with the corpus.
	doc_lda = ldamodel[corpus]

	result_dict['train_topics'] = doc_lda
	result_dict['family_list'] = family_list
	result_dict['train_kmers'] = train_kmers

	return result_dict


#this class matches topics with families so that we can predict topics for the test data.
#the matching is done like this;
#	for each sequence, assign the most probable topic to it
# 	for each family, count the number of sequences with their topics
# 	find the topic that represents the family as the most frequent topic with sequences
def getFamilyTopicPairs(topicResults, familyList):
	topic_family_dict = {} #resulting dictionary topic -> family t1 -> f3 according to below example.
	intermediate_dict = {} #intermediate dictionary that keeps family frequences for topic. t1 -> {f1->4,f2->2,f3->10}
	index = 0
	for item in topicResults: #for every sequence
		family_name = familyList[index] # get the family name
		max_topic =  max(item, key=lambda x: x[1]) # get the most probable topic for sequence
		if max_topic[0] in intermediate_dict: #increase family score of the topic
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

	for topicKey in intermediate_dict.keys(): #get only the most frequent family for topic
		topicValue = intermediate_dict[topicKey]
		top_scored_family = max(topicValue.keys(), key=(lambda key: topicValue[key]))
		topic_family_dict[topicKey] = top_scored_family

	return topic_family_dict

#gets the predicted labels for sequence using the topic family dictionary.
def getPredictedLabels(topicResults, topicDict):
	predictedLabels = []
	for item in topicResults:
		max_topic =  max(item, key=lambda x: x[1])
		if max_topic[0] in topicDict:
			predictedLabels.append(topicDict[max_topic[0]])
		else:
			predictedLabels.append(-1)

	return predictedLabels