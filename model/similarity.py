import sys
import math

import numpy as np
import scipy
import sklearn
import sklearn.manifold
import matplotlib.pyplot as plt

try:
    from model import model
except ImportError:
    import model
sys.path.append('.')
from utils.utils import *
from corpus import corpus

def frobenius_norm(A, B):
    frobenius_norm = np.linalg.norm(A-B)
    return frobenius_norm

def compute_similarities(model_type):
    scope = 'ModelSimilarity'
    cached_result = cache_read(scope, model_type)
    if cached_result:
        return cached_result
    names = []
    models = []
    for name, m in model.models(model_type):
        models.append(m)
        names.append(name)
    num_models = len(models)
    dbg_start('Computing model similarity for model type "{}"'.format(model_type))
    # Stores the Frobenius norm between each pair of models.
    distance_matrix = np.zeros((num_models, num_models))    
    # Maps words to their summed Euclidean norms over each model's representation.
    word_variance = {word: 0.0 for word in model.DICTIONARY}
    for i in range(num_models):
        base_word_dist = corpus.corpus_word_distribution(names[i])
        base_model = models[i]
        distance_matrix[i, i] = 0
        for j in range(i+1, num_models):
            comp_word_dist = corpus.corpus_word_distribution(names[j])
            comp_model = models[j]
            # If necessary, align the spaces as best as possible.
            if model_type != 'ppmi':
                R, _ = scipy.linalg.orthogonal_procrustes(comp_model, base_model)
                comp_model = np.dot(comp_model, R)
            # Compute the Frobenius norm between each pair of models.
            distance_matrix[i, j] = frobenius_norm(base_model, comp_model)
            distance_matrix[j, i] = distance_matrix[i, j]
            # Add the Euclidean norm between models for each word.
            for word, idx in model.WORD_LOOKUP.items():
                base_rep = base_model[idx]
                comp_rep = comp_model[idx]
                min_word_freq = min(base_word_dist[word], comp_word_dist[word])
                word_variance[word] += frobenius_norm(base_rep, comp_rep)
    dbg_end()
    result = (names, distance_matrix, word_variance)
    cache_write(scope, model_type, result)
    return result

if __name__ == '__main__':
    for model_type in model.MODEL_TYPES:
        info('{} model:'.format(model_type.upper()))
        results = compute_similarities(model_type)
        names = results[0]
        distance_matrix = results[1]
        word_variance = results[2]

        # Print the similarity matrix.
        info('\nSimilarity matrix:')
        for i in range(len(names)):
            print('\t{:24} : {}'.format(names[i], list(distance_matrix[i])))
        # Print the top K most variable words.
        K = 50
        info('\nTop {} highest variance words:'.format(K))
        for word, var in sorted(word_variance.items(), key=lambda item: item[1], reverse=True)[:K]:
            print('\t{}: {}'.format(word, var))
        '''
        embedded_pts = sklearn.manifold.TSNE().fit_transform(distance_matrix)
        fig, ax = plt.subplots()
        for i, pt in enumerate(embedded_pts):
            ax.scatter(pt[0], pt[1], label=names[i])
        ax.legend()
        ax.grid(True)
        plt.show()
        '''

