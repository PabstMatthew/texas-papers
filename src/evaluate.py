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

def tsne_embedding(name, space, model_type):
    scope = 'tsneEmbedding-'+model_type
    cached_result = cache_read(scope, name)
    if not cached_result is None:
        return cached_result
    tsne_embedding = sklearn.manifold.TSNE().fit_transform(space)
    cache_write(scope, name, tsne_embedding)
    return tsne_embedding

def word_variances(model_type, spaces, names):
    scope = 'WordVariance'
    cached_result = cache_read(scope, model_type)
    if not cached_result is None:
        return cached_result
    num_models = len(spaces)
    dbg_start('Computing word variances for model type "{}"'.format(model_type))
    # Maps words to their summed Euclidean norms over each model's representation.
    word_variance = dict((word, 0.0) for word in model.DICTIONARY)
    for i in range(num_models):
        base_model = spaces[i]
        for j in range(i+1, num_models):
            comp_model = spaces[j]
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

def model_similarities(model_type, spaces, names):
    scope = 'ModelSimilarity'
    cached_result = cache_read(scope, model_type)
    if not cached_result is None:
        return cached_result
    num_models = len(spaces)
    dbg_start('Computing model similarity for model type "{}"'.format(model_type))
    # Stores the Frobenius norm between each pair of models.
    distance_matrix = np.zeros((num_models, num_models))    
    for i in range(num_models):
        base_model = spaces[i]
        distance_matrix[i, i] = 0
        for j in range(i+1, num_models):
            comp_model = spaces[j]
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

FIG_PATH = 'figures'
def create_word_figures(model_type, spaces, names, word):
    # Setup directories.
    if not os.path.exists(FIG_PATH):
        os.mkdir(FIG_PATH)
    mtype_path = os.path.join(FIG_PATH, model_type)
    if not os.path.exists(mtype_path):
        os.mkdir(mtype_path)
    word_path = os.path.join(mtype_path, word)
    if not os.path.exists(word_path):
        os.mkdir(word_path)
    # Create figures.
    fpath = os.path.join(word_path, 'embeddings.png')
    if not os.path.exists(fpath):
        plot_word_embedding_all_corpora(model_type, spaces, names, word, fpath=fpath)
    for i in range(len(spaces)):
        fpath = os.path.join(word_path, 'nn-{}.png'.format(names[i]))
        if not os.path.exists(fpath):
            plot_word_embedding_nn(model_type, names[i], spaces[i], word, fpath=fpath)

def plot_word_embedding_all_corpora(model_type, spaces, names, word, fpath=None):
    # Build a plot showing t-SNE embeddings of a word for all corpora.
    dbg_start('Plotting word embeddings across all corpora for word "{}" and model type "{}"'.format(word, model_type))
    idx = model.WORD_LOOKUP[word]
    word_embeddings = []
    for i, space in enumerate(spaces):
        word_embeddings.append(tsne_embedding(names[i], space, model_type))
    plt.figure()
    names = [name for name in corpus.corpus_info.keys()]
    for i, pt in enumerate(word_embeddings):
        plt.scatter(pt[0], pt[1], label=names[i])
    plt.legend()
    plt.title('Word embeddings of "{}" for model type {}'.format(word, model_type))
    dbg_end()
    if fpath:
        plt.savefig(fpath, dpi=300)
        plt.close()
    else:
        plt.show()

def plot_word_embedding_nn(model_type, name, space, word, fpath=None):
    # Build a plot showing a word's embedding with its nearest neighbors.
    dbg_start('Plotting nearest neighbor embeddings for word "{}" in corpus {} and model type {}'.format(word, name, model_type))
    idx = model.WORD_LOOKUP[word]
    lowdim_embedding = tsne_embedding(name, space, model_type)
    indices, _ = model.nn(name, space, word)
    words = [model.WORD_LIST[idx] for idx in indices]
    word_embeddings = [lowdim_embedding[idx] for idx in indices]
    plt.figure()
    x = [embed[0] for embed in word_embeddings]
    y = [embed[1] for embed in word_embeddings]
    for i, embedding in enumerate(word_embeddings):
        plt.scatter(embedding[0], embedding[1], label=words[i])
    plt.legend()
    plt.title('Nearest neighbors of "{}" for {}-{}'.format(word, name, model_type))
    dbg_end()
    if fpath:
        plt.savefig(fpath, dpi=300)
        plt.close()
    else:
        plt.show()

def plot_distance_matrix(distance_matrix, normalize=False, method='tsne'):
    names = [name for name in corpus.corpus_info.keys()]
    # Normalize distance matrix to z-scores
    if normalize:
        dists = [distance_matrix[i][j] for i in range(1, len(names)) for j in range(i+1, len(names))]
        mean = np.mean(dists)
        std = np.std(dists)
        for i in range(len(names)):
            for j in range(len(names)):
                if i == j:
                    continue
                distance_matrix[i][j] -= mean
                distance_matrix[i][j] /= std
    if method == 'tsne':
        embedding = sklearn.manifold.TSNE().fit_transform(distance_matrix)
    elif method == 'pca':
        pca = sklearn.decomposition.PCA(n_components=2)
        pca.fit(distance_matrix)
        embedding = np.transpose(pca.components_)
    plt.figure()
    for i, pt in enumerate(embedding):
        plt.scatter(pt[0], pt[1], label=names[i])
    plt.legend()
    plt.title('tSNE Visualization of Model Differences')
    plt.show()

if __name__ == '__main__':
    for model_type in model.MODEL_TYPES:
        info('{} model:'.format(model_type.upper()))
        names = []
        spaces = []
        for name, space in model.models(model_type):
            names.append(name)
            if len(spaces) > 0:
                # If necessary, align all spaces to the first space.
                if model_type != 'ppmi':
                    R, _ = scipy.linalg.orthogonal_procrustes(space, spaces[0])
                    space = np.dot(space, R)
            spaces.append(space)
        names = [name for name in corpus.corpus_info.keys()]
        # Print the similarity matrix.
        word_variance = word_variances(model_type, spaces, names)
        distance_matrix = model_similarities(model_type, spaces, names)
        info('Similarity matrix:')
        for i in range(len(names)):
            vals = ['{:.2f}'.format(val) for val in distance_matrix[i]]
            #print('\t{:24} : {}'.format(names[i], ' | '.join(vals)))
            # Print each city's closest neighbors in order.
            print('{}:'.format(names[i]))
            for j, val in enumerate([(x, y) for y, x in sorted(zip(distance_matrix[i], names))]):
                print('  {}. {} ({:.0f})'.format(j, val[0], val[1]))
        plot_distance_matrix(distance_matrix, normalize=True, method='tsne')

        # Print the top K most variable words.
        K = 30
        info('Top {} highest variance words:'.format(K))
        for word, var in sorted(word_variance.items(), key=lambda item: item[1], reverse=True)[:K]:
            print('\t{}: {:.2f}'.format(word, var))
            #create_word_figures(model_type, spaces, names, word)
        '''
        word = 'court'
        plot_word_embedding_all_corpora(model_type, spaces, names, word)
        for name, space in model.models(model_type):
            plot_word_embedding_nn(model_type, name, space, word)
        plt.show()
        '''

