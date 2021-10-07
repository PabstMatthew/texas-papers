import sys

import nltk
from nltk.probability import FreqDist
import numpy as np
import gensim
from gensim.models import Word2Vec

sys.path.append('.')
from utils.utils import *
import corpus.corpus

MODEL_TYPES = ['ppmi', 'svd', 'sgn']

# The hyperparameters used to train models.
HYPERPARAMS = {
        # Removes stopwords from the text.
        'ignore_stopwords': True,
        # Removes proper nouns from the text.
        # (might mess up nearest neighbors because proper nouns will remain in the dictionary)
        'ignore_proper_nouns': False,
        # Removes non-content (not a noun, verb, adjective, or adverb) words from the text.
        # (might mess up nearest neighbors because some unused words will remain in the dictionary)
        'ignore_non_content_words': False,
        # Attempts to lemmatize words.
        'lemmatize': False,
        # The size of the context window to the left and right of each target word.
        'window_size': 4,
        # The threshold for word frequency. 
        # Only words that occur at least this many times in all corpora will be included.
        'freq_threshold': 20,
        # Number of dimensions for lower-dimensionality models (SVD, SGN).
        'embedding_size': 300,
}

'''
    Loads a dictionary to be used for all corpora, based on the specified word frequency threshold.
        returns: a set of strings (words).
'''
def load_dictionary():
    scope = 'Dictionary'
    freq_threshold = HYPERPARAMS['freq_threshold']
    name = 'Threshold-{}'.format(freq_threshold)
    if HYPERPARAMS['ignore_stopwords']:
        name += '-NoStopwords'
    cached_dict = cache_read(scope, name)
    if cached_dict:
        return cached_dict
    else:
        dictionary = build_dictionary()
        cache_write(scope, name, dictionary)
        return dictionary

'''
    Creates a dictionary of words that appear frequently in all corpora.
        returns: a set of strings (words).
'''
def build_dictionary():
    # Only keep words that appear at least <threshold> times in all corpora.
    dbg('Building a dictionary for the available corpora ...')
    freq_threshold = HYPERPARAMS['freq_threshold']
    dictionary = None
    for name, txt in corpus.corpora():
        words = nltk.word_tokenize(txt)
        dist = FreqDist(map(lambda word: word.lower(), 
                    filter(lambda word: word.isalpha(), words)))
        for word, count in list(dist.items()):
            if count < freq_threshold:
                del dist[word]
        if dictionary:
            dictionary = dictionary.intersection(set(dist.keys()))
        else:
            dictionary = set(dist.keys())

    # Get the user to decide on words that are questionably allowable.
    known_wordset = set(map(lambda w: w.lower(), nltk.corpus.words.words()))
    known_wordset.update(map(lambda w: w.lower(), nltk.corpus.brown.words()))
    cnt = 0
    for word in list(dictionary):
        cnt += 1
        if not word in known_wordset:
            info('Is "{}" a word? Press <Enter> for yes. {}/{} words completed.'.format(word, cnt, len(dictionary)))
            resp = input()
            if resp != '':
                info('Rejected "{}".'.format(word))
                dictionary.remove(word)
    if HYPERPARAMS['ignore_stopwords']:
        stopwords = set(nltk.corpus.stopwords.words('english'))
        dictionary = dictionary.difference(stopwords)
    return dictionary

# The dictionary to be used for every model.
DICTIONARY = load_dictionary()
dbg('Corpus dictionary: {}'.format(str(DICTIONARY)))
dbg('Loaded corpus dictionary of size {}'.format(len(DICTIONARY)))
# A list form of the dictionary used for reverse lookup.
WORD_LIST = list(DICTIONARY)
# A dictionary mapping words to their indices in the word list.
WORD_LOOKUP = dict((word, i) for i, word in enumerate(WORD_LIST))

'''
    Pre-processes a word with its PoS tag based on the hyperparameters.
        item: a tuple, whose first entry is the word, and whose second entry is the PoS tag
        returns: either None if the word should be excluded, or a string representing the processed word.
'''
def preprocess_tagged_word(item):
    word = item[0]
    tag = item[1]
    stopwords = set(nltk.corpus.stopwords.words('english'))
    if not word.isalnum():
        # Ignore punctutation.
        return None
    # Lowercase all words.
    word = word.lower()
    if HYPERPARAMS['ignore_stopwords'] and word in stopwords:
        # Ignore stopwords.
        return None
    if HYPERPARAMS['ignore_proper_nouns'] and tag.startswith('NNP'):
        # Ignore proper nouns.
        return None

    if not tag.startswith('NNP') and HYPERPARAMS['lemmatize']:
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
        lemmatizer = nltk.stem.WordNetLemmatizer()
        return lemmatizer.lemmatize(word)
    return word

'''
    Generator that yields all the words in the context window around a word in a sentence.
        i: the current index (the middle of the window).
        words: a list of words (the sentence) to create the window from.
        returns: a generator of the words in the context window.
'''
def context_window(i, words):
    window_size = HYPERPARAMS['window_size']
    for j in range(max(0, i-window_size), min(i+window_size+1, len(words))):
        if j != i:
            yield words[j]

'''
    Builds a list of sentences for a corpus after pre-processing all words.
        txt: a string containing all the text in the corpus.
        returns: a list of lists of strings representing all sentences in the corpus.
'''
def build_sentences(name, txt):
    scope = 'Sentences'
    cached_sentences = cache_read(scope, name)
    if cached_sentences:
        return cached_sentences
    sentences = []
    for sentence in nltk.sent_tokenize(txt):
        # Preprocess the words in the sentence.
        words = list(
                    filter(lambda word: not word is None, 
                        map(preprocess_tagged_word,
                            nltk.pos_tag(nltk.word_tokenize(sentence)))))
        sentences.append(words)
    cache_write(scope, name, sentences)
    return sentences

'''
    Computes a co-occurrence matrix for a corpus.
        sentences: a list of lists of pre-processed words.
        returns: a matrix of dimension (len(DICTIONARY), len(DICTIONARY)) whose rows correspond 
                 to target words, and whose columns correspond to context words. Each value  
                 contains the number of co-occurrences seen in the corpus.
'''
def build_co_occurrence_count_matrix(sentences):
    # Build the co-occurrence counts matrix.
    count_matrix = np.zeros((len(DICTIONARY), len(DICTIONARY)))
    for sentence in sentences:
        # Iterate over all the words in the sentence, counting co-occurrences.
        for i, word in enumerate(sentence):
            if not word in DICTIONARY:
                # This word isn't in our dictionary, so just ignore it.
                continue
            target_idx = WORD_LOOKUP[word]
            for context_word in context_window(i, sentence):
                if context_word in DICTIONARY:
                    context_idx = WORD_LOOKUP[context_word]
                    count_matrix[target_idx][context_idx] += 1
    return count_matrix

'''
    Computes a positive pointwise mutual-information matrix from a co-occurrence matrix.
        count_matrix: a co-occurrence matrix.
        returns: a matrix of the same dimension where values represent PMI.
'''
def build_ppmi_matrix(count_matrix):
    # First, calculate the probability of each context word. (P(b))
    near_zero = 10e-4
    count_all = count_matrix.sum()
    col_totals = count_matrix.sum(axis=0).astype(float)
    col_totals[col_totals == 0.0] = near_zero
    p_c = col_totals / count_all

    # Then, calculate probability of a context word appearing given a target word. (P(b|a))
    row_totals = count_matrix.sum(axis=1).astype(float)
    row_totals[row_totals == 0.0] = near_zero
    p_c_given_t = (count_matrix.T / row_totals).T

    # Finally, calculate PMI. log(P(b|a)/P(b))
    pct_divided_by_pc = p_c_given_t / p_c
    pct_divided_by_pc[pct_divided_by_pc == 0.0] = near_zero
    pmi_matrix = np.log(pct_divided_by_pc)

    # Replace all negative values with 0.
    pmi_matrix[pmi_matrix < 0] = 0
    return pmi_matrix

'''
    Reduces the dimension of matrix by using SVD. Dimensionality is specified in HYPERPARAMS.
        matrix: the matrix to be reduced with SVD.
        returns: a new matrix with the same number of rows, with reduced column dimensionality.
'''
def build_svd_matrix(matrix):
    # Compute SVD.
    U, s, _ = np.linalg.svd(matrix)
    # Reduce dimensionality.
    embedding_size = HYPERPARAMS['embedding_size']
    U = U[:, :embedding_size]
    S = np.diag(s[:embedding_size])
    return np.dot(U, S)

'''
    An interactive function to investigate the nearest neighbors of words in a meaning space.
        space: a meaning space.
        n: the number of neighbors to print.
        sgn: whether space is an SGN model. Otherwise, space is assumed to be a matrix with 
             len(DICTIONARY) rows.
'''
def nn(space, n=10, sgn=False):
    if sgn:
        def print_nn(word):
            if not word in DICTIONARY:
                print('"{}" does not exist in the space!'.format(word))
                return
            nns = space.wv.most_similar(word, topn=n)
            print('{} nearest neighbors of "{}":'.format(n, word))
            for i in range(n):
                print('    {} ({:.2f})'.format(nns[i][0], nns[i][1]))
    else:
        import scipy
        from sklearn.neighbors import NearestNeighbors
        nn_counts = NearestNeighbors(n_neighbors=n, metric=scipy.spatial.distance.cosine)
        nn_counts.fit(space)
        def print_nn(word):
            if not word in DICTIONARY:
                print('"{}" does not exist in the space!'.format(word))
                return
            idx = WORD_LOOKUP[word]
            if np.all((space[idx] == 0)):
                print('"{}" has no occurrences!'.format(word))
            print('{} nearest neighbors of "{}":'.format(n, word))
            dists, indices = nn_counts.kneighbors([space[idx]])
            for i in range(n):
                print('  {} ({:.2f})'.format(WORD_LIST[indices[0][i]], dists[0][i]))
    info('Enter a word to see its nearest neighbors.')
    q = input()
    while len(q) > 0:
        print_nn(q)
        q = input()

def train_ppmi(sentences):
    count_matrix = build_co_occurrence_count_matrix(sentences)
    ppmi_matrix = build_ppmi_matrix(count_matrix)
    return ppmi_matrix

def train_svd(sentences):
    ppmi_matrix = train_ppmi(count_matrix)
    svd_matrix = build_svd_matrix(ppmi_matrix)
    return svd_matrix

def train_sgn(sentences):
    embedding_size = HYPERPARAMS['embedding_size']
    sgn = Word2Vec(sentences, vector_size=embedding_size, sg=1)
    return sgn

def train_model(name, sentences, model_type):
    scope = 'Model'
    name += '-'+model_type
    cached_model = cache_read(scope, name)
    info('Training model "{}" ...'.format(name))
    if cached_model:
        dbg('Loaded cached model with hyperparameters: {}'.format(str(cached_model[0])))
        return cached_model[1]
    if model_type == 'ppmi':
        model = train_ppmi(sentences)
    elif model_type == 'svd':
        model = train_svd(sentences)
    elif model_type == 'sgn':
        model = train_sgn(sentences)
    else:
        err('Unsupported model type "{}"!'.format(model_type))
    if not model is None: 
        cache_write(scope, name, (HYPERPARAMS, model))
    return model

def models(model_type):
    for name, txt in corpus.corpora():
        sentences = build_sentences(name, txt)
        model = train_model(name, sentences, model_type)
        yield name, model

if __name__ == '__main__':
    # If this script is called, just build every model so they're cached.
    for model_type in MODEL_TYPES:
        for name, model in models(model_type):
            pass

