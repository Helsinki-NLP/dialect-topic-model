## Dialect topic modeling
This repo contains scripts and metadata for the paper "Corpus-based dialectometry with topic models". For the original data, see below. 
There are two scripts: one for pre-processing your data to character n-grams, and one for the actual topic modeling. For Morfessor-segmentation, please refer to <https://morfessor.readthedocs.io/en/latest/general.html>.
* ngramming.py assumes your data is stored in a folder as txt files. You can run the script by python3 ngramming.py your-corpus-name. This will result in four json files: words_corpus, bigram_corpus, trigram_corpus and fourgram_corpus.
* dialectTopicModel.py assumes your data is stored in the aforementioned json files. There are several arguments one can change in the running of the model.

## Example runs of the topic model 
* 5-component model of SKN on bigrams and NMF
```bash
dialectTopicModel.topic_model('skn', 'bigram', 'skn_bigram', 'nmf', 5, use_idf=True, norm='l2', sublinear=True)
```
* 2-component model of Archimob on words and LDA
```bash
dialectTopicModel.topic_model('archimob', 'words', 'archimob_words', 'lda', 2, relevance=True, lambda_=0.2)
```

## Original paper data
* Samples of Spoken Finnish: <https://korp.csc.fi/download/SKN/skn-vrt/>
* Norwegian Dialect Corpus: <http://www.tekstlab.uio.no/scandiasyn/download.html>
* Archimob Corpus of Swiss German: <https://spur.uzh.ch/en/departments/research/textgroup/ArchiMob.html>
