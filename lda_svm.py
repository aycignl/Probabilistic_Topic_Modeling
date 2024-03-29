

import sys
import os

from sklearn import svm
from sklearn.model_selection import cross_val_score

from sklearn import decomposition

#Different from the lda.py which used only most frequent
#topic to represent a sequence. We used a feature vector
#that either uses all probability distributions in the ldamodel
#or probability distribution over the best topic for each family.
def classify(results):
	topic_results_train = results['train_topics']
	topic_results_test = results['test_topics']
	train_family_list = results['train_family_list']
	test_family_list = results['test_family_list']
	top_topics = results['top_topics']
	
	#create feature vectors.
	train_vectors = createFeatureVectors(topic_results_train,top_topics)
	test_vectors = createFeatureVectors(topic_results_test,top_topics)

	#pca if the feature vector is too big
	#we tried this but did not perform better than not using pca.
	#pca = decomposition.PCA(n_components=20)
	#pca.fit(train_vectors)
	#train_vectors = pca.transform(train_vectors)

	#test_vectors = pca.transform(test_vectors)

	#We find the best c value for the data by using the training data
	#and doing 5-fold cross validation
	best_C = findBestCValue(train_vectors,train_family_list)
	clf = svm.SVC(kernel='linear', C = best_C)

	clf.fit(train_vectors, train_family_list)

	#svm prediction
	predicted_labels = list(clf.predict(test_vectors))

	result = {}
	result['reallabels'] = test_family_list
	result['predictedlabels'] = predicted_labels

	return result

#5-fold cross validation
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
#you can use the best topics by commenting out the line from the for number in range(100)
# and open the code below it.
def createFeatureVectors(topic_lda,top_topics):
	feature_vector = []

	for item in topic_lda:
		item_vector = []
		item_topic_index =  [i[0] for i in item]
		item_topic_score =  [i[1] for i in item]
		for number in range(100):
			if number in item_topic_index:
				index = item_topic_index.index(number)
				item_vector.append(item_topic_score[index])
			else:
				item_vector.append(0)

	# 	for topic in top_topics:
	# 		if topic in item_topic:
	# 			index = item_topic.index(topic)
	# 			item_vector.append(item[index][1])
	# 		else:
	# 			item_vector.append(0)
		feature_vector.append(item_vector)

	return feature_vector
	