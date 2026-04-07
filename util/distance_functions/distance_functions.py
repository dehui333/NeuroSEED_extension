import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F


def square_distance(t1_emb, t2_emb, args):
    scaling = args.get('scaling', None)

    D = t1_emb - t2_emb
    d = torch.sum(D * D, dim=-1)

    if scaling is not None:
        d = d * scaling
    return d


def euclidean_distance(t1_emb, t2_emb, args):
    scaling = args.get('scaling', None)


    D = t1_emb - t2_emb
    d = torch.norm(D, dim=-1)

    
    if scaling is not None:
        scaling = scaling.to(d.device)
        d = d * scaling
    return d

def euclidean_distance_v2(t1_emb, t2_emb, args):
    scaling = args.get('scaling2', None)


    D = t1_emb - t2_emb
    d = torch.norm(D, dim=-1)

    
    if scaling is not None:
        scaling = scaling.to(d.device)
        d = d * scaling
    return d


def cosine_distance(t1_emb, t2_emb, args):
    scaling = args.get('scaling', None)

    d = 1 - nn.functional.cosine_similarity(t1_emb, t2_emb, dim=-1, eps=1e-6)

    if scaling is not None:
        d = d * scaling
    return d


def manhattan_distance(t1_emb, t2_emb, args):
    scaling = args.get('scaling', None)

    D = t1_emb - t2_emb
    d = torch.sum(torch.abs(D), dim=-1)
    
    if scaling is not None:
        d = d * scaling
    return d


def hyperbolic_distance(u, v, args, epsilon=1e-7):  
    scaling = args.get('scaling', None)

    sqdist = torch.sum((u - v) ** 2, dim=-1)
    squnorm = torch.sum(u ** 2, dim=-1)
    sqvnorm = torch.sum(v ** 2, dim=-1)
    x = 1 + 2 * sqdist / ((1 - squnorm) * (1 - sqvnorm)) + epsilon
    z = torch.sqrt(x ** 2 - 1)

    d = torch.log(x + z)
    
    if scaling is not None:
        scaling = scaling.to(d.device)
        d = d * scaling
    return d

def hyperbolic_distance_v2(u, v, args, epsilon=1e-7):  
    scaling = args.get('scaling', None)

    sqdist = torch.sum((u - v) ** 2, dim=-1)
    squnorm = torch.sum(u ** 2, dim=-1)
    sqvnorm = torch.sum(v ** 2, dim=-1)
    x = 1 + 2 * sqdist / ((1 - squnorm) * (1 - sqvnorm)) + epsilon
    z = torch.sqrt(x ** 2 - 1)

    d = torch.log(x + z)
    
    if scaling is not None:
        scaling = scaling.to(d.device)
        d = d * scaling
    return d


def hyperbolic_distance_numpy(u, v, args, epsilon=1e-9):
    scaling = args.get('scaling', None)


    sqdist = np.sum((u - v) ** 2, axis=-1)
    squnorm = np.sum(u ** 2, axis=-1)
    sqvnorm = np.sum(v ** 2, axis=-1)
    x = 1 + 2 * sqdist / ((1 - squnorm) * (1 - sqvnorm)) + epsilon
    z = np.sqrt(x ** 2 - 1)

    d = np.log(x + z)

    if scaling is not None:
        d = d * scaling
    return d

def mahalanobis_distance(t1_emb, t2_emb, args):
    """
    t1_emb, t2_emb: tensors of shape (..., dim)
    B: covariance factor matrix of shape (dim, dim), where M = B^T B
    """
    scaling = args.get('scaling', None)
    B = args['covariance_factor'] 
    B = B.to(t1_emb.device)

    D = t1_emb - t2_emb           # difference vector
    BD = torch.matmul(D, B.T)     # apply Mahalanobis transform
    d = torch.norm(BD, dim=-1)    # compute norm

    if scaling is not None:
        scaling = scaling.to(t1_emb.device)
        d = d * scaling
    return d

def hyperbolic_euclidean(t1_emb, t2_emb, args):
    w_logit = args['w_logit']
    scaling = args.get('scaling', None)

    # Strip scaling from sub-calls
    local_args = dict(args)
    local_args['scaling'] = None

    hyperbolic_component = hyperbolic_distance(t1_emb, t2_emb, local_args)
    euclidean_component = euclidean_distance(t1_emb, t2_emb, local_args)

    w = torch.sigmoid(w_logit).to(t1_emb.device)
    d = w * hyperbolic_component + (1 - w) * euclidean_component

    # do not scale
    #if scaling is not None:
    #    d = d * scaling.to(d.device)
    return d

def hyperbolic_euclidean_v2(t1_emb_hyp, t2_emb_hyp, t1_emb_euc, t2_emb_euc, args):
    w_logit = args['w_logit']


    hyperbolic_component = hyperbolic_distance(t1_emb_hyp, t2_emb_hyp, args)
    euclidean_component = euclidean_distance(t1_emb_euc, t2_emb_euc, args)

    w = torch.sigmoid(w_logit).to(t1_emb_hyp.device)
    d = w * hyperbolic_component + (1 - w) * euclidean_component

    # do not scale
    #if scaling is not None:
    #    d = d * scaling.to(d.device)
    return d

def hyperbolic_mahalanobis_distance(t1_emb_hyp, t2_emb_hyp, t1_emb_euc, t2_emb_euc, args):
    w_logit = args['w_logit']


    hyperbolic_component = hyperbolic_distance(t1_emb_hyp, t2_emb_hyp, args)
    local_args = dict(args)
    local_args['scaling'] = args['scaling2']
    euclidean_component = mahalanobis_distance(t1_emb_euc, t2_emb_euc, local_args)

    w = torch.sigmoid(w_logit).to(t1_emb_hyp.device)
    d = w * hyperbolic_component + (1 - w) * euclidean_component

    # do not scale
    #if scaling is not None:
    #    d = d * scaling.to(d.device)
    return d



def hyperbolic_distance_curv(u, v, args, epsilon=1e-7):
    scaling = args.get('scaling', None)
    # c is kept positive via softplus in the module
    raw_c = args['curvature']
    c = F.softplus(raw_c).clamp(min=0.1, max=10.0)
    c = c.to(u.device)

    sqdist = torch.sum((u - v) ** 2, dim=-1)
    squnorm = torch.sum(u ** 2, dim=-1)
    sqvnorm = torch.sum(v ** 2, dim=-1)

    x = 1 + 2 * c * sqdist / ((1 - c * squnorm) * (1 - c * sqvnorm)) + epsilon
    z = torch.sqrt(x ** 2 - 1)
    d = torch.log(x + z) / torch.sqrt(c)

    if scaling is not None:
        scaling = scaling.to(d.device)
        d = d * scaling
    return d

def weighted_euclidean_distance(t1_emb, t2_emb, args):
    """
    Per-dimension weighted Euclidean distance (diagonal Mahalanobis).
    weights: learnable vector of shape (dim,), where the effective metric is diag(weights^2).
    """
    scaling = args.get('scaling', None)
    weights = args['dimension_weights']
    weights = weights.to(t1_emb.device)
    D = t1_emb - t2_emb
    WD = D * weights  # per-dimension scaling
    d = torch.norm(WD, dim=-1)
    if scaling is not None:
        scaling = scaling.to(d.device)
        d = d * scaling
    return d

def hyperbolic_distance_fixed(u, v, args, epsilon=1e-7):
    scaling = args.get('scaling', None)
    c = args['curvature'].to(u.device)
    sqdist = torch.sum((u - v) ** 2, dim=-1)
    squnorm = torch.sum(u ** 2, dim=-1)
    sqvnorm = torch.sum(v ** 2, dim=-1)
    x = 1 + 2 * c * sqdist / ((1 - c * squnorm) * (1 - c * sqvnorm))
    x = torch.clamp(x, min=1.0 + 1e-6)
    z = torch.sqrt(x ** 2 - 1)
    d = torch.log(x + z) / torch.sqrt(c)
    if scaling is not None:
        d = d * scaling.to(d.device)
    return d

DISTANCE_TORCH = {
    'square': square_distance,
    'euclidean': euclidean_distance,
    'cosine': cosine_distance,
    'manhattan': manhattan_distance,
    'hyperbolic': hyperbolic_distance,
    'mahalanobis': mahalanobis_distance,
    'hyperbolic_euclidean': hyperbolic_euclidean,
    'hyperbolic_curv': hyperbolic_distance_curv,
    'weighted_euclidean' : weighted_euclidean_distance,
    'weighted_hyperbolic' : hyperbolic_distance,
    'hyperbolic_fixed': hyperbolic_distance_fixed,
    'hyperbolic_euclidean_v2': hyperbolic_euclidean_v2,
    'hyperbolic_mahalanobis' : hyperbolic_mahalanobis_distance,
}
