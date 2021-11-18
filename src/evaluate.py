import sys
import math

import numpy as np
import scipy
import sklearn
import sklearn.manifold
import matplotlib.pyplot as plt

from utils import *
import corpus
import model

def frobenius_norm(A, B):
    frobenius_norm = np.linalg.norm(A-B)
    return frobenius_norm

def cosine_dist(A, B):
    return np.dot(A, B) / (np.linalg.norm(A)*np.linalg.norm(B))

def tsne_embedding(name, model, model_type):
    scope = 'tsneEmbedding-'+model_type
    cached_result = cache_read(scope, name)
    if cached_result:
        return cached_result
    tsne_embedding = sklearn.manifold.TSNE().fit_transform(m)
    cache_write(scope, name, tsne_embedding)
    return tsne_embedding

def word_variances(model_type):
    scope = 'WordVariance'
    cached_result = cache_read(scope, model_type)
    if not cached_result is None:
        return cached_result
    names = []
    models = []
    for name, m in model.models(model_type):
        models.append(m)
        names.append(name)
    num_models = len(models)
    dbg_start('Computing word variances for model type "{}"'.format(model_type))
    # Maps words to their summed Euclidean norms over each model's representation.
    word_variance = dict((word, 0.0) for word in model.DICTIONARY)
    for i in range(num_models):
        base_model = models[i]
        for j in range(i+1, num_models):
            comp_model = models[j]
            # If necessary, align the spaces as best as possible.
            if model_type != 'ppmi':
                R, _ = scipy.linalg.orthogonal_procrustes(comp_model, base_model)
                comp_model = np.dot(comp_model, R)
            # Add the Euclidean norm between models for each word.
            for word, idx in sorted(model.WORD_LOOKUP.items()):
                base_rep = base_model[idx]
                comp_rep = comp_model[idx]
                word_variance[word] += cosine_dist(base_rep, comp_rep)
    dbg_end()
    cache_write(scope, model_type, word_variance)
    return word_variance

def model_similarities(model_type):
    scope = 'ModelSimilarity'
    cached_result = cache_read(scope, model_type)
    if not cached_result is None:
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
    for i in range(num_models):
        base_model = models[i]
        distance_matrix[i, i] = 0
        for j in range(i+1, num_models):
            comp_model = models[j]
            # If necessary, align the spaces as best as possible.
            if model_type != 'ppmi':
                R, _ = scipy.linalg.orthogonal_procrustes(comp_model, base_model)
                comp_model = np.dot(comp_model, R)
            # Compute the Frobenius norm between each pair of models.
            distance_matrix[i, j] = frobenius_norm(base_model, comp_model)
            distance_matrix[j, i] = distance_matrix[i, j]
    dbg_end()
    cache_write(scope, model_type, distance_matrix)
    return distance_matrix

def plot_word_embedding_all_corpora(model_type, word):
    # Build a plot showing t-SNE embeddings of a word for all corpora.
    dbg_start('Plotting word embeddings across all corpora for word "{}" and model type "{}"'.format(word, model_type))
    idx = model.WORD_LOOKUP[word]
    word_embeddings = []
    for name, m in model.models(model_type):
        lowdim_embedding = sklearn.manifold.TSNE().fit_transform(m)
        word_embeddings.append(lowdim_embedding[idx])
    plt.figure()
    names = [name for name in corpus.corpus_info.keys()]
    for i, pt in enumerate(word_embeddings):
        plt.scatter(pt[0], pt[1], label=names[i])
    plt.legend()
    plt.title('Word embeddings of "{}" for model type {}'.format(word, model_type))
    plt.show()
    dbg_end()

def plot_word_embedding_nn(model_type, name, word):
    # Build a plot showing a word's embedding with its nearest neighbors.
    dbg_start('Plotting nearest neighbor embeddings for word "{}" in corpus {} and model type {}'.format(word, name, model_type))
    idx = model.WORD_LOOKUP[word]
    for n, m in model.models(model_type):
        if n == name:
            lowdim_embedding = sklearn.manifold.TSNE().fit_transform(m)
            indices, _ = model.nn(name, m, word)
            words = [model.WORD_LIST[idx] for idx in indices]
            word_embeddings = [lowdim_embedding[idx] for idx in indices]
            plt.figure()
            x = [embed[0] for embed in word_embeddings]
            y = [embed[1] for embed in word_embeddings]
            for i, embedding in enumerate(word_embeddings):
                plt.scatter(embedding[0], embedding[1], label=words[i])
            plt.legend()
            plt.title('Nearest neighbors of "{}" for {}-{}'.format(word, name, model_type))
            plt.show()
            dbg_end()
            return
    dbg_end()
    warn('No model "{}" of type "{}" found!'.format(name, model_type))

if __name__ == '__main__':
    for model_type in model.MODEL_TYPES:
        if model_type != 'sgn':
            continue
        info('{} model:'.format(model_type.upper()))
        names = [name for name in corpus.corpus_info.keys()]
        # Print the similarity matrix.
        word_variance = word_variances(model_type)
        distance_matrix = model_similarities(model_type)
        info('Similarity matrix:')
        for i in range(len(names)):
            vals = ['{:.2f}'.format(val) for val in distance_matrix[i]]
            print('\t{:24} : {}'.format(names[i], ' | '.join(vals)))
        # Print the top K most variable words.
        K = 50
        info('Top {} highest variance words:'.format(K))
        for word, var in sorted(word_variance.items(), key=lambda item: item[1], reverse=True)[:K]:
            print('\t{}: {:.2f}'.format(word, var))
        word = 'arrested'
        plot_word_embedding_all_corpora(model_type, word)
        for name in names:
            plot_word_embedding_nn(model_type, name, word)

