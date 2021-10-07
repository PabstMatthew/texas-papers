import numpy as np
import scipy

try:
    import model.model
except ImportError:
    import model
sys.path.append('.')
from utils.utils import *

def space_similarity(A, B):
    # Align the spaces first.
    R, _ = scipy.linalg.orthogonal_procrustes(A, B)
    A = np.dot(A, R)
    # Frobenius norm is used as the similarity metric.
    frobenius_norm = np.linalg.norm(A-B)
    return frobenius_norm

def compute_similarities():
    for model_type in model.MODEL_TYPES:
        info('{} model similarities:'.format(model_type.upper()))
        names = []
        models = []
        for name, model in model.models():
            models.append(model)
            names.append(name)
        num_models = len(models)
        distance_matrix = np.zeros((num_models, num_models))
        for i in range(num_models):
            for j in range(num_models):
                distance_matrix[i, j] = space_similarity(models[i], models[j])
        for i in range(num_models):
            print('{:20} : {}'.format(names[i], distance_matrix[i]))

