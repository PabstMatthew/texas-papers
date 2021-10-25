import sys

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

def space_similarity(A, B, align=True):
    # Align the spaces first.
    if align:
        R, _ = scipy.linalg.orthogonal_procrustes(A, B)
        A = np.dot(A, R)
    # Frobenius norm is used as the similarity metric.
    frobenius_norm = np.linalg.norm(A-B)
    return frobenius_norm

def compute_similarities():
    for model_type in model.MODEL_TYPES:
        names = []
        models = []
        for name, m in model.models(model_type):
            models.append(m)
            names.append(name)
        num_models = len(models)
        info('{} model similarities:'.format(model_type.upper()))
        distance_matrix = np.zeros((num_models, num_models))
        for i in range(num_models):
            distance_matrix[i, i] = 0
            for j in range(i+1, num_models):
                distance_matrix[i, j] = space_similarity(models[i], models[j], align=(model_type != 'ppmi'))
                distance_matrix[j, i] = distance_matrix[i, j]
        for i in range(num_models):
            print('{:24} : {}'.format(names[i], list(distance_matrix[i])))
        '''
        embedded_pts = sklearn.manifold.TSNE().fit_transform(distance_matrix)
        fig, ax = plt.subplots()
        for i, pt in enumerate(embedded_pts):
            ax.scatter(pt[0], pt[1], label=names[i])
        ax.legend()
        ax.grid(True)
        plt.show()
        '''

if __name__ == '__main__':
    compute_similarities()
