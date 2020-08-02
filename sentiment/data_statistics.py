import os
import numpy as np
import pandas as pd

# Sklearn Vectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline

import matplotlib.pyplot as plt
import lightgbm as lgb


# Top n-gram correlation
def get_term_frequency(corpus, ngram_range=(1, 1)):
    tokenizer_kwargs = dict(
        analyzer='word',  # for many misspelled words, 'char_wb' is recommended
        ngram_range=ngram_range,  # (1, 1) = unigram, unigram, (1, 2) = unigram, bigram, etc.
        min_df=2,  # if integer, remove word that occurs less than this number
    )
    token_f = CountVectorizer(
        input='content',
        **tokenizer_kwargs,
    )    
    A_tokenized = token_f.fit_transform(corpus)
    
    term_count = np.array(A_tokenized.sum(axis=0)).flatten().tolist()
    term_names = token_f.get_feature_names()
    term_df = pd.DataFrame(list(zip(term_names, term_count)), columns=['name', 'count']).sort_values(by='count', ascending=False)
    term_df = term_df.set_index('name')
    
    return term_df

def plot_side_by_side(first_df, second_df, n_show=50, top=True):
    n_show = n_show
    fig, ax = plt.subplots(1, 2, figsize=(14, n_show/5))
    if top:
        first_df.head(n_show)[::-1].plot(kind='barh', ax=ax[0], legend=None, alpha=0.7)
        second_df.head(n_show)[::-1].plot(kind='barh', ax=ax[1], legend=None, alpha=0.7)
    else:
        first_df.tail(n_show).plot(kind='barh', ax=ax[0], legend=None, alpha=0.7)
        second_df.tail(n_show).plot(kind='barh', ax=ax[1], legend=None, alpha=0.7)
    ax[0].set_title(f'Rating=1 top {n_show} Terms')
    ax[1].set_title(f'Rating=2 top {n_show} Terms')
    ax[0].set_ylabel('')
    ax[1].set_ylabel('')
    plt.tight_layout()
    plt.show()


# get the statistics of word counts in the dataset -> ngram_range(1, 1) means only one word
train_term_df_1 = get_term_frequency(df[df['rating'] == 1]['review'], ngram_range=(1, 1))
train_term_df_2 = get_term_frequency(df[df'rating'] == 2]['review'], ngram_range=(1, 1))
plot_side_by_side(train_term_df_1, train_term_df_2)

# two words
train_term_df_1 = get_term_frequency(df[df['rating'] == 1]['review'], ngram_range=(2, 2))
train_term_df_2 = get_term_frequency(df[df'rating'] == 2]['review'], ngram_range=(2, 2))
plot_side_by_side(train_term_df_1, train_term_df_2)