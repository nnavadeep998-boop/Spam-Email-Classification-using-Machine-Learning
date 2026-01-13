# Spam-Email-Classification-using-Machine-Learning
Project Description  This project focuses on classifying emails as Spam or Not Spam (Ham) using machine learning techniques. Text data is converted into numerical form using TF-IDF Vectorization, and a Naive Bayes classifier is trained to identify spam messages effectively.

| Library                           | Purpose                  |
| --------------------------------- | ------------------------ |
| `numpy`                           | Numerical operations     |
| `pandas`                          | Data handling            |
| `sklearn.model_selection`         | Train-test split         |
| `sklearn.feature_extraction.text` | Text vectorization       |
| `sklearn.naive_bayes`             | Classification algorithm |
| `sklearn.metrics`                 | Model evaluation         |


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



1 Importing Libraries
Required libraries are imported for text processing, model training, and evaluation.

2️ Dataset Creation
A small email dataset is manually created with spam and ham labels.

3️ Data Preprocessing
Text messages are converted into numerical vectors using TF-IDF Vectorizer.

4️ Train-Test Split
Dataset is split into 80% training and 20% testing data.

5️ Model Creation
Multinomial Naive Bayes classifier is used for spam detection.

6️ Model Training
The model learns patterns from the training data.

7️ Prediction
The model predicts whether messages are spam or ham.

8️. Model Evaluation
Performance is measured using:
Accuracy
Classification report
Confusion matrix
