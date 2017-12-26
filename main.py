from gensim import corpora, models
import gensim

import sys
import os

global trainFile
global testFile

import lda
import lda_svm as ldasvm

from sklearn.metrics import confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
import itertools



def main():

	global count
	count = 0

	results = {}

	#if mode == 1: # "mostProbableWins": default
	print("Start classifying with lda.")
	results = lda.classify(trainFile, testFile)
	showConfusionMatrix(results,"Best Topic Assigned LDA")

	print("Start classifying with lda svm.")
	#best half topics for sentence
	results = ldasvm.classify(results)

	real_labels = results['reallabels']
	predicted_labels = results['predictedlabels']
	class_names = list(set(real_labels))
	write_to_files(real_labels, predicted_labels,class_names)
	
	showConfusionMatrix(results,"Probable Topic Assigned LDA - SVM")

def showConfusionMatrix(results,caption):
	real_labels = results['reallabels']
	predicted_labels = results['predictedlabels']
	class_names = list(set(real_labels))
	write_to_files(real_labels, predicted_labels,class_names)
	cnf_matrix = confusion_matrix(real_labels, predicted_labels, labels = class_names)

	acc = np.asarray(cnf_matrix)
	print("Accuracy : "+str(np.trace(acc)/np.sum(acc)))

	np.set_printoptions(precision=2)
	plt.figure()
	plot_confusion_matrix(cnf_matrix, classes=class_names,
                      title=caption)
	plt.show()


def write_to_files(real_labels, predicted_labels, class_names):

	global count
	output_file = open("results_"+str(count)+".txt", "w+", encoding="utf-8")
	output_file.write("[ ")
	for item in real_labels:
		output_file.write(item)
		output_file.write(" ")
	output_file.write("]")
	output_file.write("\n")
	output_file.write("[ ")
	for item in predicted_labels:
		output_file.write(item)
		output_file.write(" ")
	output_file.write("]")
	output_file.write("\n")
	output_file.write("[ ")
	for item in class_names:
		output_file.write(item)
		output_file.write(" ")
	output_file.write("]")
	
	count = count + 1

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')



if __name__ == '__main__':
	trainFile = sys.argv[1]
	testFile = sys.argv[2]
	mode = sys.argv[3]
	global count
	main()