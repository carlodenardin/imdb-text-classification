# Text Classification on IMDB dataset

This report contains the analysis and implementation of Text Classification algorithms. Specifically, there are 3 notebooks:

- exploratory: in which an analysis of the dataset is carried out
- ml: where ML techniques are applied (svm, naive bayes, and logistic regression)
- deep: where DL techniques are applied (bilstm, cnn)

The following implementation has been tested with `python 3.11` and the libraries listed in the `requirements.txt`. Additionally, to test everything, it is necessary to have the [IMDB](https://ai.stanford.edu/%7Eamaas/data/sentiment/) dataset.

Below are the results obtained:

| Model  | Accuracy | F1-Score | Recall | Precision |
| ------------- | ------------- | -------------  | -------------  | -------------  |
| Support Vector Classifier  | <u>**0.9049**</u> | 0.9053| 0.9011 | 0.9096 |
| Naive Bayes  | 0.8768 | 0.8719 | 0.9078 | 0.8388 |
| Logistic Regression  | 0.9011 | 0.9014 | 0.8981 | 0.9048 |

| Model  | Accuracy | 
| ------------- | ------------- |
| BiLSTM | 0.8796 |
| Double BiLSTM  | 0.8836|
| CNN  | <u>**0.8896**</u> |
| CNN + LSTM | 0.8736 |