# Object Detection Literature Survey

2020-2021 Fall CSE4084 - Multimedia Systems



This [report]() and [presentation]() are prepared for the final project of CSE4084 Multimedia Systems course in Marmara University and aiming to cover foundational papers of object detection literature. Mainly classical approaches, deep learning detectors along with their design tricks are explained by using original papers and web blogs. 

Main accomplishments of this project are learning and examining the architectures, algorithms of design tricks / heuristics and gaining knowledge on state-of-art results. It can be pointed out that working this project was very helpful on introducing object detection domain and can be expanded in to video object detection, image segmentation, object tracking, 3D scenes and even more on autonomous driving domains as future work.

---

#### Online Notebooks:

1. [EDA](https://www.kaggle.com/aysenur95/text-classification-1-eda)

2. [Text cleaning, hand-made features, label encoding, train-test split](https://www.kaggle.com/hakkoz/ctr-2-read-data)

3. a. [Text representations (bow and 1-hot)](https://www.kaggle.com/aysenur95/text-classification-3-1-text-representation)
   
   b. [Text representations (tf-idf) and the analysis of most correlated words](https://www.kaggle.com/aysenur95/text-classification-3-2-text-representation)

4. a. [ML classification with 3 text representations (Naive-Bayes)](https://www.kaggle.com/aysenur95/text-classification-4-1-naive-bayes)
   
   b. [ML classification with 3 text representations (Random Forest)](https://www.kaggle.com/aysenur95/text-classification-4-2-rf)
   
   c. [ML classification with 3 text representations (Logistic Regression)](https://www.kaggle.com/aysenur95/text-classification-4-3-logistic-regression)
   
   d. [ML classification with 3 text representations (Support Vector Machines)](https://www.kaggle.com/aysenur95/text-classification-4-4-svm)
   
   e. [ML classification with 3 text representations (XGBoost Classifier)](https://www.kaggle.com/aysenur95/text-classification-4-5-xgboost-classifier)
   
   f. [Training word embeddings and a vanilla neural network](https://www.kaggle.com/aysenur95/text-classification-4-6-word-embedding-nnvanilla)
   
   g. [Fine-tuning Glove(6B_50d) embeddings and CNN](https://www.kaggle.com/aysenur95/text-classification-4-7-cnn-glove-6b-50d)

5. a. [Comparing models and deciding the best one](https://www.kaggle.com/aysenur95/text-classification-5-results)
   
   b. [Topic modeling with LDA, WordClouds and PCA](https://www.kaggle.com/aysenur95/text-classification-5-topic-modelling)

6. [Utility functions for training models](https://www.kaggle.com/aysenur95/text-classification-utility-functions)

---

#### Repo Content and Implementation Steps:

[**1.EDA**](https://github.com/mustafahakkoz/Text_Classification_ML-DL/tree/master/1.EDA)

- Exploratory Data Analysis for gaining insights about the data. We have examined the dataset using univariate and bivariate analysis. 
- Also we determined on which attribute will be the target.

[**2.feature engineering**](https://github.com/mustafahakkoz/Text_Classification_ML-DL/tree/master/2.feature%20engineering)

- We clean punctuations signs, special characters, numbers and possessive pronouns.

- We convert all letters to lowercase.

- All reviews are processed by wordnet lemmatizer of NLTK library, to convert words into their dictionary forms. 

- Stopwords are also removed by using stopword list of NLTK library again.

- Some text-based features are calculated such as *word count, unique word count, letter count, punctuation count, count of upper-cased words, number count, stopword count, average length of words*. Also bivariate analysis and correlation matrix of these features are implemented. However, we did not choose to use them as training features since we  would like to make predictions semantically.

- And finally, label encoding of the target attribute and .75/.25 train-test split are done.

[**3.text representation**](https://github.com/mustafahakkoz/Text_Classification_ML-DL/tree/master/3.text%20representation)

- We implemented three different techniques for text representation (one-hot, bow, tf-idf vectorizers of ScikitLearn) in 2 different notebooks.

- Most correlated unigrams and bigrams are calculated by chi2 method for each label.

[**4.a.ML classification algorithms**](https://github.com/mustafahakkoz/Text_Classification_ML-DL/tree/master/4.a.ML%20classification%20algorithms)

- 5 ML models (*Naive Bayes, Logistic Regression, SVM, Random Forest, XGBoost*) are implemented for each of these 3 text representations with 5-fold stratified cross validation and gridsearch.

[**4.b.DL classification algorithms**](https://github.com/mustafahakkoz/Text_Classification_ML-DL/tree/master/4.b.DL%20classification%20algorithms)

- We also tried word embeddings using deep learning models. For both of the experiments,  we used ​ *relu*​ as an activation function for all hidden layers.​​ Two different approaches are used on GPU notebooks of Kaggle: Embedding Layer in a vanilla NN to train our own vectors and pretrained *Glove* vectors + CNN.

- First, we trained our own word embedding using the Embedding layer (dimension=50) of Keras. In this experiment we used vanilla neural network GlobalMaxPool layer + DropOut (0.5) layer + Dense layer with 50 nodes + SoftMax layer with 10 nodes.

- As a second experiment, we used the pre-trained word embedding (Glove 6B 50d) with a CNN model. The architecture of CNN model is Embedding (trainable=True for fine tuning) layer + Conv1D (filters=50, kernel size=3) + MaxPooling1D (3) layer + DropOut (0.5) layer + Dense layer with 50 nodes + SoftMax layer with 10 nodes.

[**5.a.classification results**](https://github.com/mustafahakkoz/Text_Classification_ML-DL/tree/master/5.a.classification%20results)

<img title="" src="https://github.com/mustafahakkoz/Text_Classification_ML-DL/blob/master/images/results.png" alt="" height="300">

- Neural networks are the most successful models on train dataset, but on the test set their accuracy scores are a bit off compared to RF and XGBoost. On the other hand training scores of RF and XGBoost are lower than their test scores which is a weird behaviour, so we choose the Glove CNN model over them in our app.

[**5.b.topic_modeling**](https://github.com/mustafahakkoz/Text_Classification_ML-DL/tree/master/5.b.topic_modeling)

- Latent Dirichlet Allocation (LDA) is an unsupervised example of topic modelling and it is used to cluster the text to particular topics. We ran LDA using the tf-idf method.

- We plot WordClouds to to see the words belonging each topics and to compare the results of LDA.

- And visualize tf-idf vectors in 2D and 3D (plotly) by using PCA dimensionality reduction method. 

[**6.best model predictor**](https://github.com/mustafahakkoz/Text_Classification_ML-DL/tree/master/6.best%20model%20predictor)

- A python script to make predictions with best model (Glove+CNN) for testing purposes. It does preprocessing, converts string to array with pretrained *keras-tokenizer* and runs pretrained *CNN* model on it. And finally prints predictions on screen.

[**7.application deployment**](https://github.com/mustafahakkoz/Text_Classification_ML-DL/tree/master/7.application%20deployment)

- In the deployment phase, *Dash* and *Heroku* used to build and deploy the app. Dash is a framework to create single page basic web applications based on Flask and Plotly. Also, Heroku is a free deployment tool. 

- The input text is converted to a pre-trained word embedding model. After that, a CNN classifier is applied to predict its condition.

[**text-classification-utility-functions.ipynb**](https://github.com/mustafahakkoz/Text_Classification_ML-DL/blob/master/text-classification-utility-functions.ipynb)

- A python script that contains two functions for displaying test scores and tuning hyperparameters.

[**report.pdf**](https://github.com/mustafahakkoz/Text_Classification_ML-DL/blob/master/report.pdf)

- A detailed report on project goals and implementation steps also including explanations on methods used.
