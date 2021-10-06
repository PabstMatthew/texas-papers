import sys

import nltk
import numpy as np

sys.path.append('.')
from utils.utils import *
from corpus.corpus import corpuses
from corpus import corpus_dictionary, corpus_lookup

HYPERPARAMS = {
        'ignore_stopwords': True,
        'ignore_proper_nouns': True,
        'ignore_non_content_words': True,
        'lemmatize': True,
        'window_size': 4,
}

def preprocess_tagged_word(item):
    word = item[0]
    tag = item[1]
    stopwords = set(nltk.corpus.stopwords.words('english'))
    if not word.isalphanum():
        # Ignore punctutation.
        return None
    if HYPERPARAMS['ignore_stopwords'] and word in stopwords:
        # Ignore stopwords.
        return None
    if HYPERPARAMS['ignore_proper_nouns'] and tag.startswith('NNP'):
        # Ignore proper nouns.
        return None
    if tag.startswith('NNP'):
        # Don't modify the capitalization of proper nouns.
        return word

    # Lowercase all words besides proper nouns.
    word = word.lower()
    if HYPERPARAMS['lemmatize']:
        # Map Penn tag to WordNet tag for lemmatization.
        if tag.startswith('NN'):
            wntag = nltk.corpus.wordnet.NOUN
        elif tag.startswith('VB'):
            wntag = nltk.corpus.wordnet.VERB
        elif tag.startswith('JJ'):
            wntag = nltk.corpus.wordnet.ADJ
        elif tag.startswith('RB'):
            wntag = nltk.corpus.wordnet.ADV
        else: 
            if HYPERPARAMS['ignore_non_content_words']:
                # Not a content word, so just ignore it.
                return None
            else:
                return word
        # Lemmatize and return the word.
        return lemmatizer.lemmatize(word)
    return word

def context_window(i, words):
    window_size = HYPERPARAMS['window_size']
    for j in range(i-window_size, i+window_size+1):
        if j > 0 and j < len(words) and j != i:
            yield words[j]

def train_svd(txt):
    # Build the co-occurrence counts matrix.
    sentences = nltk.sent_tokenize(txt)
    count_matrix = np.zeros((len(corpus_dictionary), len(corpus_dictionary)))
    for sentence in sentences:
        # Preprocess the words in the sentence.
        words = list(
                    filter(lambda word: not word is None, 
                        map(lambda word_pos: preprocess_tagged_word,
                            nltk.pos_tag(nltk.word_tokenize(sentence)))))
        # Iterate over all the words in the sentence, counting co-occurrences.
        for word in words:
            if not word in corpus_lookup:
                # This word isn't in our dictionary, so just ignore it.
                continue
            target_idx = corpus_lookup[word]
            for context_word in context_window:
                if context_word in corpus_lookup:
                    context_idx = corpus_lookup[context_word]
                    count_matrix[target_idx][context_idx] += 1
    # TODO Add some nearest neighbors debug output here, and maybe an evaluation of some kind.
    # TODO Do the SVD.

def train_model(name, txt):
    MODEL_TYPE = 'svd'
    #MODEL_TYPE = 'sgn'
    scope = 'Model'
    name += '-'+MODEL_TYPE
    cached_model = cache_read(scope, name)
    info('Training model "{}" ...'.format(name))
    if cached_model:
        dbg('Loaded cached model with hyperparameters: {}'.format(str(cached_model[0])))
        return cached_model[1]
    if MODEL_TYPE == 'svd':
        model = train_svd(txt)
    elif MODEL_TYPE == 'sgn':
        model = train_sgn(txt)
    else:
        err('Unsupported model type "{}"!'.format(MODEL_TYPE))
    if not model is None: 
        cache_write(scope, name, (HYPERPARAMS, model))
    return model

if __name__ == '__main__':
    for name, txt in corpuses():
        model = train_model(name, txt)
