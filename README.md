# Domain Classification for Haptik Dataset

* The dataset contains two separate files for training and testing/evaluation of data.
* Each human query/message is categorized into single/multiple domains by human annotators.
    - *T*: message belongs to the corresponding class
    - *F*: message does not belong to the corresponding class (column heading represents the class name)


### Haptik Domain Classification
* Task: train a multi-label classifier to classify "message" into one or more classes/domains
* Suggested evaluation metrics
    - Precision
    - Recall
    - Accuracy per class
    - Overall Accuracy

NOTE:
* Please note that the messages are in raw format, it also contains system specific content which may not be important for identifying domain.
* It is important to preprocess messages and remove such information for better training.


### The Setting
Here is a nice function that will fetch the dataset and format it in the correct way for further processing:


    import pandas as pd
    import numpy as np

    def fetch_grid():
        train = pd.read_csv('./data/domain_classification/train_data.csv')
        test = pd.read_csv('./data/domain_classification/test_data.csv')
        return train, test

    train_haptik, test_haptik = fetch_grid()
    train_haptik.head()
    test_haptik.head()

The column "message" contains the text to be analysed and the following columns contain the information about which category the text falls in.


#### Task 1: Constructing the Datasets

Write a fucntion `feature_target_separator()` with following parameter:

* `df`: non-separated dataset

The function splits the dataset into the features and target dataframes

#### Task 2: Reversing One-Hot

It can be seen that the target variables are One-hot-encoded.

Write a function `reverse_OHE()` with follwing parameters

* `df`: target dataframe

And returns labeled pandas Series


#### Task 3: Word Cloud

Word cloud is a technique that helps us visualise raw data

Write a function `word_cloud()` with follwing parameters

* `X`: pandas series containing text data to be plotted
* `max_font_size`: maximum size of the font desired in the word cloud

And returns the word cloud.

The function should perform following tasks:

* tokenization
* removal of stop words
* Stemming


#### Task 4: Classification

Write a function `classification()` with follwing parameters

* `X_train`: 1 column pandas dataframe with text
* `X_test`: 1 column pandas dataframe with text
* `y_train`: 1 column pandas dataframe with labels
* `y_test`: 1 column pandas dataframe with labels
* `tokenizer`: string
* `stop_words`: string
* `ngram_range`: tuple with (min_val, max_val) format
* `max_df`: ignore terms that appear in more than given fraction of the documents[0,1]
* `min_df`: only keep terms that appear in at least n documents (int)

The function uses `MultinomialNB` classifier and should return accuracy and confusion matrics