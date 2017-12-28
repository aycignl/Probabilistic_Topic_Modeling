# probabilisticTopicModeling
Probabilistic Topics Modeling project for Cmpe59h - Bioinformatics Course @[Bogazici University](http://www.boun.edu.tr/en_US).<br>
**Participants:** [Gonul Aycı](https://www.cmpe.boun.edu.tr/~gonul.ayci/), [Dilara Keküllüoğlu](https://dilara91.github.io/). <br> **Instructor:** [Assoc. Dr. Arzucan Özgür](https://www.cmpe.boun.edu.tr/~ozgur/)

In this project, we use popular Asgari Word to vec(W2V) protein embeddings and the [dataset](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/JMFHTN) they used. We implement our project using with **Python**. <br> 

In this project, we used this paper as reference: "La Rosa, Massimo, et al. [Probabilistic topic modeling for the analysis and classification of genomic sequences.](https://bmcbioinformatics.biomedcentral.com/articles/10.1186/1471-2105-16-S6-S2) BMC bioinformatics 16.6 (2015): S2." <br>

Our repository consists of two parts. The first part is a data preparation. The second part includes algorithms and main parts of this project. <br>

To run this project, <br>
  python main.py "train-data" "test-data" <br>
  python main.py classification_train_3.csv classification_test_3.csv <br>
  
Implementations: <br>
* **data-preparation.ipynb**: In this notebook, you can find how to merge data, and apply 3, 5, and 8 -mers to this data. <br>
* **analysis.ipynb**: In this notebook, you can find how to select 100 families which have maximum number of proteins among Asgari dataset, and how do we split train and test data. <br>
* **lda.py**: Creates Latent Dirichlet Allocation (lda) model from the train data and then extracts topic - family dictionary from the probability distributions. Classification is done by assigning the family of the most probable topic to each sequence.
* **lda_svm.py**: Uses the lda model created in lda.py and creates feature vectors with the probability distributions using this model. Classification is done by svm of sklearn library.
* **lda_svm_w2v.py**: Adds Asgari's word2vec embeddings to the lda_svm.py's feature vectors. Classification is done with svm.
* **main.py**: Called with the train and test data file paths. Executes lda, lda_svm and lda_svm_w2v consecutively and shows confusion matrices using matplotlib. Need to close the corresponding confusion matrix for code to continue executing.<br>
  
Our csv files formats are in **Asgari ID, Family ID, SwissProt Accession ID, Sequences**.
