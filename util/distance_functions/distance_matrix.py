import numpy as np
import scipy.spatial.distance
import torch
import torch.nn.functional as F

from util.distance_functions.distance_functions import hyperbolic_distance, mahalanobis_distance, hyperbolic_euclidean, hyperbolic_distance_curv, weighted_euclidean_distance, hyperbolic_distance_fixed, hyperbolic_euclidean_v2, hyperbolic_mahalanobis_distance


def euclidean_matrix(enc_reference, enc_query, args):

    distances = torch.cdist(enc_reference, enc_query)
    return distances


def square_matrix(enc_reference, enc_query, args):

    d = euclidean_matrix(enc_reference, enc_query, args)
    return d * d


def manhattan_matrix(enc_reference, enc_query, args):

    distances = torch.cdist(enc_reference, enc_query, p=1)
    return distances


def cosine_matrix(enc_reference, enc_query, args):

    (N, D) = enc_reference.shape
    (M, D) = enc_query.shape
    cosine_sim = torch.zeros((N, M), device=enc_reference.device)
    for j in range(M):
        cosine_sim[:, j] = F.cosine_similarity(enc_reference, enc_query[j:j + 1].repeat(N, 1))
    return 1 - cosine_sim


def hyperbolic_matrix(enc_reference, enc_query, args):

    (N, D) = enc_reference.shape
    (M, D) = enc_query.shape
    d = torch.zeros((N, M), device=enc_reference.device)
    for j in range(M):
        d[:, j] = hyperbolic_distance(enc_reference, enc_query[j:j+1].repeat(N, 1), args)

    return d

def mahalanobis_matrix(enc_reference, enc_query, args):

    (N, D) = enc_reference.shape
    (M, D) = enc_query.shape
    d = torch.zeros((N, M), device=enc_reference.device)
    for j in range(M):
        d[:, j] = mahalanobis_distance(enc_reference, enc_query[j:j+1].repeat(N, 1), args)

    return d

def hyperbolic_euclidean_matrix(enc_reference, enc_query, args):

    (N, D) = enc_reference.shape
    (M, D) = enc_query.shape
    d = torch.zeros((N, M), device=enc_reference.device)
    for j in range(M):
        d[:, j] = hyperbolic_euclidean(enc_reference, enc_query[j:j+1].repeat(N, 1), args)

    return d

def hyperbolic_euclidean_v2_matrix(enc_reference_hyp, enc_query_hyp, enc_reference_euc, enc_query_euc, args):

    (N, D) = enc_reference_hyp.shape
    (M, D) = enc_query_hyp.shape
    d = torch.zeros((N, M), device=enc_reference_hyp.device)
    for j in range(M):
        d[:, j] = hyperbolic_euclidean_v2(enc_reference_hyp, enc_query_hyp[j:j+1].repeat(N, 1), enc_reference_euc, enc_query_euc[j:j+1].repeat(N, 1), args)

    return d

def hyperbolic_mahalanobis_matrix(enc_reference_hyp, enc_query_hyp, enc_reference_euc, enc_query_euc, args):

    (N, D) = enc_reference_hyp.shape
    (M, D) = enc_query_hyp.shape
    d = torch.zeros((N, M), device=enc_reference_hyp.device)
    for j in range(M):
        d[:, j] = hyperbolic_mahalanobis_distance(enc_reference_hyp, enc_query_hyp[j:j+1].repeat(N, 1), enc_reference_euc, enc_query_euc[j:j+1].repeat(N, 1), args)

    return d


def hyperbolic_matrix_numpy(u, v, eps=1e-9):
    m = scipy.spatial.distance.cdist(u, v) ** 2
    u_sqr = np.sum(u ** 2, axis=1)
    v_sqr = np.sum(v ** 2, axis=1)
    divisor = np.maximum(1. - np.expand_dims(u_sqr, axis=1), eps) * np.maximum(1. - np.expand_dims(v_sqr, axis=0), eps)
    D_ij = np.arccosh(1 + 2 * m / divisor)
    return D_ij

def hyperbolic_curv_matrix(enc_reference, enc_query, args):
    (N, D) = enc_reference.shape
    (M, D) = enc_query.shape
    d = torch.zeros((N, M), device=enc_reference.device)
    for j in range(M):
        d[:, j] = hyperbolic_distance_curv(enc_reference, enc_query[j:j+1].repeat(N, 1), args)
    return d

def weighted_euclidean_matrix(enc_reference, enc_query, args):
    (N, D) = enc_reference.shape
    (M, D) = enc_query.shape
    d = torch.zeros((N, M), device=enc_reference.device)
    for j in range(M):
        d[:, j] = weighted_euclidean_distance(enc_reference, enc_query[j:j+1].repeat(N, 1), args)
    return d

def hyperbolic_fixed_matrix(enc_reference, enc_query, args):
    (N, D) = enc_reference.shape
    (M, D) = enc_query.shape
    d = torch.zeros((N, M), device=enc_reference.device)
    for j in range(M):
        d[:, j] = hyperbolic_distance_fixed(enc_reference, enc_query[j:j+1].repeat(N, 1), args)
    return d

DISTANCE_MATRIX = {
    'euclidean': euclidean_matrix,
    'square': square_matrix,
    'manhattan': manhattan_matrix,
    'cosine': cosine_matrix,
    'hyperbolic': hyperbolic_matrix,
    'mahalanobis': mahalanobis_matrix,
    'hyperbolic_euclidean': hyperbolic_euclidean_matrix,
    'hyperbolic_curv': hyperbolic_curv_matrix,
    'weighted_euclidean' : weighted_euclidean_matrix,
    'weighted_hyperbolic' : hyperbolic_matrix,
    'hyperbolic_fixed': hyperbolic_fixed_matrix,
    'hyperbolic_euclidean_v2': hyperbolic_euclidean_v2_matrix,
    'hyperbolic_mahalanobis' : hyperbolic_mahalanobis_matrix
}
