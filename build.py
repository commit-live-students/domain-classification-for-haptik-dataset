def feature_target_separator(dataset):
    """
    enter your solution here
    """
    X = dataset.iloc[:,0]
    y = dataset.iloc[:,1:]
    return X,y


def reverse_OHE(df):
    """
    enter your solution here
    """
    df = df.iloc[:,1:].astype(str).replace({'T':1,'F':0})
    df = df.idxmax(axis=1)
    return df

def word_cloud(X, max_font_size):
    """
    enter your solution here
    """
    X = X.apply(lambda row: row.split(' '))
    text = []
    for sentence in X:
        text.extend(sentence)
    textall = " ".join(text)
    wordcloud = WordCloud(max_font_size = max_font_size).generate(textall)
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    plt.show()


def classification(X_train, X_test, y_train, y_test, stop_words = [], ngram_range = (1,2), max_df = 1, min_df = 1, tokenizer = TreebankWordTokenizer()):

    en_stop = get_stop_words('en')
    en_stop.extend(stop_words)
    p_stemmer = PorterStemmer()


    if tokenizer == 'RegexpTokenizer':
        tokenizer = RegexpTokenizer(r'\w+')
    elif tokenizer == 'WordPunctTokenizer':
        tokenizer = WordPunctTokenizer()
    elif tokenizer == 'PunktSentenceTokenizer':
        tokenizer = PunktSentenceTokenizer()

    X_train = X_train.apply(lambda row: row.lower())
    X_train = X_train.apply(lambda row: tokenizer.tokenize(row))
    X_train = X_train.apply(lambda row: [i for i in row if i not in en_stop])
    X_train = X_train.apply(lambda row: [p_stemmer.stem(i) for i in row])

    X_test = X_test.apply(lambda row: row.lower())
    X_test = X_test.apply(lambda row: tokenizer.tokenize(row))
    X_test = X_test.apply(lambda row: [i for i in row if i not in en_stop])
    X_test = X_test.apply(lambda row: [p_stemmer.stem(i) for i in row])

    vectorizer = CountVectorizer(ngram_range = ngram_range, max_df = max_df, min_df = min_df)

    X_train = pd.Series([' '.join(sentence) for  sentence in X_train])
    X_test = pd.Series([' '.join(sentence) for  sentence in X_test])

    y_train = LabelEncoder().fit_transform(y_train)
    y_test = LabelEncoder().fit_transform(y_test)



    vectorizer.fit(X_train)
    train_dtm = vectorizer.transform(X_train)
    test_dtm = vectorizer.transform(X_test)

    model = MultinomialNB(fit_prior = False)
    model.fit(train_dtm, y_train)
    y_predictions = model.predict(test_dtm)

    print classification_report(y_test, y_predictions)
    return accuracy_score(y_test, y_predictions),confusion_matrix(y_test, y_predictions)
