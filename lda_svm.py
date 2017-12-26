

import sys
import os

from sklearn import svm
from sklearn.model_selection import cross_val_score

def classify(results):
	topic_results_train = results['train_topics']
	topic_results_test = results['test_topics']
	train_family_list = results['train_family_list']
	test_family_list = results['test_family_list']
	top_topics = results['top_topics']
	

	train_vectors = createFeatureVectors(topic_results_train,top_topics)
	test_vectors = createFeatureVectors(topic_results_test,top_topics)

	best_C = findBestCValue(train_vectors,train_family_list)
	clf = svm.SVC(kernel='linear', C = best_C)

	clf.fit(train_vectors, train_family_list)

	predicted_labels = list(clf.predict(test_vectors))

	result = {}
	result['reallabels'] = test_family_list
	result['predictedlabels'] = predicted_labels

	return result


def findBestCValue(train_vectors,train_family_list):
	c_list = [0.5,1,2,10,100]

	best_score = 0
	best_c = -1
	for c in c_list:
		clf = svm.SVC(kernel='linear', C = c)
		scores = cross_val_score(clf, train_vectors, train_family_list, cv=5)
		accuracy = scores.mean()
		if accuracy > best_score:
			best_score = accuracy
			best_c = c

	print("Best c for this data is "+str(best_c))
	return best_c
#since topic number is 200 we can use everyone not only top topics.
def createFeatureVectors(topic_lda,top_topics):
	feature_vector = []

	for item in topic_lda:
		item_vector = []
		item_topic_index =  [i[0] for i in item]
		item_topic_score =  [i[1] for i in item]
		for number in range(200):
			if number in item_topic_index:
				index = item_topic_index.index(number)
				item_vector.append(item_topic_score[index])
			else:
				item_vector.append(0)
		
		feature_vector.append(item_vector)

	return feature_vector
	# feature_vector = []
	# for item in topic_lda:
	# 	item_vector = []
	# 	item_topic =  [i[0] for i in item]
	# 	for topic in top_topics:
	# 		if topic in item_topic:
	# 			index = item_topic.index(topic)
	# 			item_vector.append(item[index][1])
	# 		else:
	# 			item_vector.append(0)
		
	# 	feature_vector.append(item_vector)

	# return feature_vector