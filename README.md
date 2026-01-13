# Spam-Email-Classification-using-Machine-Learning
Project Description  This project focuses on classifying emails as Spam or Not Spam (Ham) using machine learning techniques. Text data is converted into numerical form using TF-IDF Vectorization, and a Naive Bayes classifier is trained to identify spam messages effectively.
Libraries Used
Library	Purpose
numpy	Numerical operations
pandas	Data handling
sklearn.model_selection	Train-test split
sklearn.feature_extraction.text	Text vectorization (TF-IDF)
sklearn.naive_bayes	Naive Bayes classifier
sklearn.metrics	Model evaluation
Algorithm Used
Multinomial Naive Bayes
This algorithm is well-suited for text classification problems and works based on Bayes’ theorem, assuming feature independence.
The required libraries are imported for data processing, feature extraction, model training, and evaluation.
A small sample dataset containing spam and ham messages is created and converted into a Pandas DataFrame.
The text messages are separated as input features, and the corresponding labels are used as output.
The dataset is divided into training and testing sets using an 80–20 split.
TF-IDF Vectorizer converts text data into numerical values that machine learning algorithms can understand.
A Multinomial Naive Bayes classifier is trained using the transformed training data.
The trained model predicts whether test messages are spam or ham.
The performance of the model is evaluated using accuracy score, classification report, and confusion matrix.
