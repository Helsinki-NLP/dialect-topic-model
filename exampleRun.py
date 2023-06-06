import dialectTopicModel

# An example run of the topic model, 5-component model of SKN on bigrams and NMF
# corpus, format, naming pattern, model, number of components, whether to use inverse document frequency, normalization method, whether to use sublinear term frequency
dialectTopicModel.topic_model('skn', 'bigram', 'skn_bigram', 'nmf', 5, use_idf=True, norm='l2', sublinear=True)

# An example run of the topic model, 2-component model of Archimob on words and LDA
dialectTopicModel.topic_model('archimob', 'words', 'archimob_words', 'lda', 2, relevance=True, lambda_=0.2)
