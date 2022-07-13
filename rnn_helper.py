# -*- coding: utf-8 -*-
"""
Created on Tue Jun  7 22:53:07 2022

@author: user
"""

from nltk.corpus import treebank
from nltk.corpus import brown
from nltk.corpus import conll2000
import pandas as pd
from tqdm import tqdm

treebank_corpus = treebank.tagged_sents(tagset='universal')
brown_corpus = brown.tagged_sents(tagset='universal')
conll_corpus = conll2000.tagged_sents(tagset='universal')


tagged_sentences = treebank_corpus + brown_corpus + conll_corpus


# len(tagged_sentences)
# tagged_sentences[2]

df = pd.DataFrame(columns=['x', 'y'])

for i in tqdm(range(len(tagged_sentences))):
    x_value, y_value = [], []

    for j in range(len(tagged_sentences[i])):
        x_value.append(tagged_sentences[i][j][0])
        y_value.append(tagged_sentences[i][j][1])
    df.at[i, 'x'] = " ".join(x_value)
    df.at[i, 'y'] = " ".join(y_value)

df['y'].apply(replace())


df.to_csv("dataset.csv", index=False)
