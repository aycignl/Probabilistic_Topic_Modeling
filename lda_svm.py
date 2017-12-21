

import sys
import os

from sklearn import svm

def classify(results):
	topic_results_train = results['train_topics']
	topic_results_test = results['test_topics']
	train_family_list = results['train_family_list']
	test_family_list = results['test_family_list']
	top_topics = results['top_topics']
	

	train_vectors = createFeatureVectors(topic_results_train,top_topics)
	test_vectors = createFeatureVectors(topic_results_test,top_topics)

	clf = svm.SVC(kernel='linear', C = 1.0)
	clf.fit(train_vectors, train_family_list)

	predicted_labels = list(clf.predict(test_vectors))

	result = {}
	result['reallabels'] = test_family_list
	result['predictedlabels'] = predicted_labels

	return result


def createFeatureVectors(topic_lda,top_topics): #reduce the number of topics if the feature vector is too big. or the number of families.
	feature_vector = []
	for item in topic_lda:
		item_vector = []
		print(item)
		item_topic =  [i[0] for i in item]
		for topic in top_topics:
			print(topic)
			if topic in item_topic:
				index = item_topic.index(topic)
				item_vector.append(item[index][1])
			else:
				item_vector.append(0)
		
		feature_vector.append(item_vector)

	return feature_vector