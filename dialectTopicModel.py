import matplotlib.pyplot as plt
import json
import re
import numpy as np
import pandas as pd
import os
import glob

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF, LatentDirichletAllocation
from sklearn.metrics.cluster import homogeneity_completeness_v_measure

from sklearn.metrics.pairwise import cosine_similarity

def topic_model(corpus, inputtype, label, modeltype, no_topics, max_df=1.0, use_idf=False, norm=None, sublinear=False, relevance=False, lambda_=False):
    """Runs a topic model from a given corpus and type of input, with additional parameters.
    Prints top items per component while running, and returns a csv file with document-component weights 
    and a bar plot of the top terms for each component.
    
    - Options of a model are 'lda' and 'nmf' corresponding to LatentDirichletAllocation and NMF of the sklearn library.
    - It is assumed that for each corpus and input type, the data is stored in json files labeled inputtype_corpus.
    - label is used solely for saving files so you can add whatever you wish
    - no_topics controls the number of topics produced.
    - max_df is a float stating the upper limit of items appearing in the corpus: 0.90 excludes all items that appear 
      in more than 90% of the documents.
    - use_idf is a boolean stating whether inverse document frequency should be used before modeling. 
      For NMF True produces better results, for LDA False.
    - norm controls the normalization of frequencies, and options are None, 'l1' and 'l2'.
      We have worked with 'l2' for NMF and None for LDA.
    - sublinear is a boolean stating whether sublinear scaling should be used for the frequencies before modeling.
      We have worked with True for NMF and False for LDA.
    - relevance is a metric only relevant to LDA. Relevance is a post-modeling process that makes the output of the top terms
      more understandable. It is described in Sievert & Shirley, 2014.
    - lambda_ controls the weight of the relevance metric. Gets a value between 0 and 1. 0 gives more weight to terms that are highly
      exclusive to a single topic, whereas value of 1 gives weight to the most frequent terms. We have used a value of 0.2.
    """
    
    if corpus == 'ndc':
        tfidf_vectorizer = TfidfVectorizer(encoding='utf-8', analyzer='word', max_df=max_df, min_df=2, 
                                       token_pattern=r'(?u)\b\w+\b', lowercase=False, use_idf=use_idf, 
                                       norm=norm, sublinear_tf=sublinear)
                                               
    else:                                   
        tfidf_vectorizer = TfidfVectorizer(encoding='utf-8', analyzer='word', max_df=max_df, min_df=2, 
                                       token_pattern=r'(?u)\b\w+\b', lowercase=True, use_idf=use_idf, 
                                       norm=norm, sublinear_tf=sublinear)
    
    with open("{}_{}".format(inputtype, corpus), "r", encoding="utf-8") as fp:
        matrix = json.load(fp)
    
    tfidf = tfidf_vectorizer.fit_transform(matrix)
    tfidf_feature_names = tfidf_vectorizer.get_feature_names_out()
    
    if modeltype == 'lda':
        model = LatentDirichletAllocation(n_components=no_topics, random_state=1, max_iter=200).fit(tfidf)
    
    elif modeltype == 'nmf':
        model = NMF(n_components=no_topics, random_state=1).fit(tfidf)
    
    else:
        return
    
    if relevance == True:
        doc_lengths = tfidf.sum(axis=1).getA1()
        term_freqs = tfidf.sum(axis=0).getA1()
        vocab = tfidf_vectorizer.get_feature_names_out()
        
        def _row_norm(dists):
        # row normalization function required
        # for doc_topic_dists and topic_term_dists
            return dists / dists.sum(axis=1)[:, None]
        
        if norm == None:
            doc_topic_dists = _row_norm(model.transform(tfidf))
            topic_term_dists = _row_norm(model.components_)
            
        else:
            doc_topic_dists = model.transform(tfidf)
            topic_term_dists = model.components_
        
        # compute relevance and top terms for each topic
        term_proportion = term_freqs / term_freqs.sum()
        
        log_lift = np.log(pd.eval("topic_term_dists / term_proportion")).astype("float64")
        log_ttd = np.log(pd.eval("topic_term_dists")).astype("float64")
        
        values_ = lambda_ * log_ttd + (1 - lambda_) * log_lift
        
    else:
        values_ = model.components_
    
    n_words = 15
    topic_model_list = []
    for topic_idx, topic in enumerate(values_):
        top_n = [tfidf_feature_names[i]
                for i in topic.argsort()
                [-n_words:]][::-1]
        top_features = ' '.join(top_n)
        topic_model_list.append(f"topic_{'_'.join(top_n[:3])}") 
        print(f"Topic {topic_idx}: {top_features}")
    
    amounts = model.transform(tfidf)
    
    ### Set it up as a dataframe
    if corpus == 'skn':
        metadata = pd.read_csv('paikat_ja_murteet.csv')
        metadata['DocID'] = metadata['Tiedosto'].astype(str)
        metadata['DocID'] = [re.sub('.txt', '', sent) for sent in metadata['DocID']]
    
    elif corpus == 'ndc':
        metadata = pd.read_csv('nor-data-speakers.csv')
        metadata['DocID'] = metadata['SpeakerID'].astype(str)
    
    elif corpus == 'archimob':
        metadata = pd.read_csv('coords43_base_areas.csv')
        metadata['DocID'] = metadata['ID'].astype(str)
    
    else:
        # Find files
        joined_files = os.path.join("*txt")
        joined_list = glob.glob(joined_files)
        metadata = pd.DataFrame(joined_list)
        metadata.rename(columns={metadata.columns[0]: 'DocID'}, inplace=True)
        metadata['DocID'] = [re.sub('.txt', '', sent) for sent in metadata['DocID']]
    
    ### Filenames and dominant topic
    topics = pd.DataFrame(amounts, columns=topic_model_list)
    dominant_topic = np.argmax(topics.values, axis=1)
    topics['dominant_topic'] = dominant_topic
    topics['DocID'] = metadata['DocID'].astype(str)
    
    ### Combine data frames
    model_metadata = pd.merge(metadata, topics, on = "DocID", how = "inner")
    
    ### Save results
    model_metadata.to_csv('{}_{}_{}_{}_relevance{}_idf{}_norm{}_sub{}.csv'.format(modeltype, label, no_topics, max_df, lambda_, use_idf, norm, sublinear))
    
    if relevance == True:
        values_ = np.exp(values_)
        
    ### The plotting function, but save
    def plot_top_words(model, feature_names, n_top_words):
        fig, axes = plt.subplots(1, no_topics, figsize=(25, 15), sharex=True)
        axes = axes.flatten()
        for topic_idx, topic in enumerate(values_):
            top_features_ind = topic.argsort()[:-n_top_words - 1:-1]
            top_features = [feature_names[i] for i in top_features_ind]
            weights = topic[top_features_ind]
            
            ax = axes[topic_idx]
            ax.barh(top_features, weights, height=0.7)
            ax.set_title(f'Component {topic_idx +1}',
                         fontdict={'fontsize': 30})
            ax.invert_yaxis()
            ax.tick_params(axis='both', which='major', labelsize=25)
            for i in 'top right left'.split():
                ax.spines[i].set_visible(False)
        
        plt.subplots_adjust(top=0.90, bottom=0.05, wspace=0.90, hspace=0.3)
        plt.savefig('{}_{}_{}_{}_relevance{}_idf{}_norm{}_sub{}.tiff'.format(modeltype, label, no_topics, max_df, lambda_, use_idf, norm, sublinear), dpi=300, format="tiff", pil_kwargs={"compression": "tiff_lzw"})
        plt.close(fig='all')
    
    n_words = 10
    plot_top_words(model, tfidf_feature_names, n_words)
    
    if corpus == 'skn':
        df2 = pd.read_csv('skn_aluetaso.csv')
        df_merged = pd.merge(model_metadata, df2, on = "Tiedosto", how = "inner")
        
        metric_area = homogeneity_completeness_v_measure(df_merged['Seutukunta'], df_merged['dominant_topic'], beta=1.5)
        metric_group = homogeneity_completeness_v_measure(df_merged['Murrealue'], df_merged['dominant_topic'], beta=1.5)
    
    elif corpus == 'ndc':
        df2 = pd.read_csv('norwegian_dialects.csv')
        df_merged = pd.merge(model_metadata, df2, on = "SpeakerID", how = "inner")
        
        metric_area = homogeneity_completeness_v_measure(df_merged['County'], df_merged['dominant_topic'], beta=1.5)
        metric_group = homogeneity_completeness_v_measure(df_merged['Dialect area'], df_merged['dominant_topic'], beta=1.5)
    
    elif corpus == 'archimob':
        df2 = pd.read_csv('coords43_base_areas.csv')
        df_merged = pd.merge(model_metadata, df2, on = "ID", how = "inner")
        
        metric_area = homogeneity_completeness_v_measure(df_merged['GEOAREA'], df_merged['dominant_topic'], beta=1.5)
        metric_group = homogeneity_completeness_v_measure(df_merged['DIALAREA'], df_merged['dominant_topic'], beta=1.5)
        
    else:
        return
    
    cosine = cosine_similarity(values_)
    np.fill_diagonal(cosine, np.nan)
    cosine = cosine[~np.isnan(cosine)].reshape(cosine.shape[0], cosine.shape[1] - 1)
    max_cosine = cosine.max()
    min_cosine = cosine.min()
    
    print(label, modeltype, no_topics, 'geographical', metric_area)
    print(label, modeltype, no_topics, 'dialect', metric_group)
    print(label, modeltype, no_topics, 'cosine_mean', cosine.mean())
    print(label, modeltype, no_topics, 'cosine similarity_max', max_cosine)
    print(label, modeltype, no_topics, 'cosine similarity_min', min_cosine)
    print(label, modeltype, no_topics, 'cosine difference', max_cosine-min_cosine)
