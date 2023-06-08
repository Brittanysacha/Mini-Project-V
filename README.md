# Duplicate Question Identification Project

## Project Goals
The principal aim of this project is to accurately identify duplicate questions from a given dataset. We've utilised a range of advanced machine learning techniques to reach our goal, from data pre-processing and feature engineering to applying sophisticated models like Neural Networks and Ensemble Models.

### Data

The labeled dataset can be downloaded from [here](https://drive.google.com/file/d/19iWVGLBi7edqybybam56bt2Zy7vpf1Xc/view?usp=sharing).

## Exploratory Data Analysis (EDA)
During the EDA, a human error was detected in the dataset. Specifically, the ground-truth labels contained errors, with duplicate entries in the Q1 and Q2 columns being incorrectly labelled as non-duplicates. A few of these instances were identified in the preliminary analysis of the data frame, indicating potential further inaccuracies.

## Process

### Data Cleaning
The cleaning process consisted of three essential stages:

* **Grammartisation**: Lowercased all questions and removed punctuation.
* **Normalisation**: Stemmed and removed stop words.
* **Tokenisation**: Text was split into individual words or tokens, generating a list of tokens representing the vocabulary or word occurrences.

### Feature Engineering (FE)
A diverse set of features were engineered to provide a comprehensive understanding of the text data:

* **Similarity Measures**: Cosine similarity, Jaccard similarity, and GloVe similarity were applied.
* **Text Length and Structure Analysis**: Features like question length, word count, length difference, and word length ratio were considered.
* **Semantic Representation**: Word2Vec averages and sentiment analysis were implemented.
* **Text Analysis**: Common words and TF-IDF differences were employed.

### Model Building
We applied three models for our task:

1. **Linear Regression**: Used primarily as a preliminary model to test the FE variables.
2. **Neural Network (NN)**: A sequential architecture NN model was built with Keras. The model underwent hyperparameter tuning with Keras HyperModelBuilder using RandomSearch.
3. **Ensemble Model**: An ensemble model was built by leveraging RandomForest, GradientBoosting, and LogisticRegression models using the VotingClassifier.

## Results

* **Linear Regression**: Showed limited capability for handling complex variables.
* **Neural Network**: Achieved an accuracy of 0.75, precision of 0.73, and F1-score of 0.73. The ROC AUC score was 0.83, and the recall was 0.74.
* **Ensemble Model**: Demonstrated good overall performance with an accuracy, precision, recall, and F1-score of 0.74. Cross-validation score was 0.735.

## Challenges
The project encountered challenges due to the incorrectly labelled ground-truth variable, which may have introduced errors in the modelling process. Despite hyperparameter tuning and thorough model testing, a higher error level was anticipated due to these inaccuracies.

## Future Goals
In future work, we intend to:

* Incorporate a BERT classifier for more robust feature engineering.
* Apply Optuna Bayesian optimisation for hyperparameter tuning across all models.
* Conduct a thorough review of a larger portion of the dataset (1000 rows) to ensure proper identification of duplicates, consequently improving the overall quality of the dataset.

For more details on the specific methods and results, please refer to the individual notebooks within this repository.


