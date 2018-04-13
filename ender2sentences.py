# produce sentences from ender text

import nltk
import codecs
import _pickle as pickle
import keras
from keras.preprocessing.text import Tokenizer
from gensim.models import word2vec as w2v
import numpy as np
import re
import string

max_sentences = 10000
max_num_words = 40000


nltk.download("punkt")
nltk.download("stopwords")

corpus_raw = u""
with codecs.open('C:\\Users\\Rey\\Projects\\PersonalDevelopment\\Tutorials\\EnderNLP\\Ender.txt', "r", "utf-8") as book_file:
        corpus_raw += book_file.read()

tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
raw_sentences = tokenizer.tokenize(corpus_raw)
raw_sentences = raw_sentences[:max_sentences]
raw_sentences = [h.lower() for h in raw_sentences]
# clean
split_sentences = []
for raw_sentence in raw_sentences:
    if len(raw_sentence) > 0:
        clean = re.sub("[^a-zA-Z]", " ", raw_sentence)
        words = clean.split()
        split_sentences.append(words)
sentences = [' '.join(s) for s in split_sentences]
# process with keras
prep = keras.preprocessing.text.Tokenizer(num_words=max_num_words,
                                          filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n',
                                          split=" ", char_level=False)
prep.fit_on_texts(sentences)
sequences = prep.texts_to_sequences(sentences)
# create dictionaries of words and their indices
word2idx = prep.word_index
idx2word = dict((idx,word) for word,idx in word2idx.items())
# process with word2vec to make word embeddings
embedding_dim = 300
min_word_count = 1
context_size = 5
downsampling = 1e-3
seed = 1
# 4.2 define model
endersentences2vec = w2v.Word2Vec(
    sg=1,
    seed=seed,
    size=embedding_dim,
    min_count=min_word_count,
    window=context_size,
    sample=downsampling
)
# 4.3 create embeddings
endersentences2vec.build_vocab(split_sentences)
endersentences2vec.train(split_sentences,
                         total_examples=endersentences2vec.corpus_count, epochs=10)
# save most common words in and embedding matrix
vocab_size = min(max_num_words, len(endersentences2vec.wv.vocab))
embedding_shape = (vocab_size, embedding_dim)
embedding = np.zeros(embedding_shape)
for i in range(1, vocab_size):
    embedding[i, :] = endersentences2vec.wv.word_vec(idx2word[i])
# extract some random numeric features
numeric_feature = [len(h) for h in split_sentences]

# save data for LSTM
with open('C:\\Users\\Rey\\Projects\\PersonalDevelopment\\Tutorials\\RNN\\TextSummerization\\EnderSentences.pkl',
          'wb') as fp:
    pickle.dump((idx2word, word2idx, embedding, sentences, sequences, numeric_feature), fp, -1)

