import torch
import torch.nn as nn
import torch.nn.functional as F

from util.distance_functions.distance_functions import DISTANCE_TORCH


class PairEmbeddingDistance(nn.Module):

    def __init__(self, embedding_model, distance='euclidean', scaling=False, fixed_curvature=1.0, reg_lambda=0.01):
        super(PairEmbeddingDistance, self).__init__()

        self.embedding_model = embedding_model
        self.distance = DISTANCE_TORCH[distance]
        self.distance_str = distance
        self.args = {}

        # doesn't seem very useful. if really need use, should add as argument
        self.reg_lambda = reg_lambda

        embedding_size = self.embedding_model.embedding_size

        self.radius = None
        self.scaling = None
        self.curvature_raw = None
        self.dimension_weights = None
        self.scaling2 = None
        if scaling:
            self.scaling = nn.Parameter(torch.Tensor([1.]), requires_grad=True)
            self.args['scaling'] = self.scaling
            if distance != 'hyperbolic_curv':
                self.radius = nn.Parameter(torch.Tensor([1e-2]), requires_grad=True)
            if distance in ('hyperbolic_euclidean_v2', 'hyperbolic_mahalanobis'):
                self.scaling2 = nn.Parameter(torch.Tensor([1.]), requires_grad=True)
                self.args['scaling2'] = self.scaling2
             
        # for mahalanobis
        self.covariance_factor = None 

        # for weighted sum 
        self.w_logit = None 

        if distance in ('mahalanobis', 'hyperbolic_mahalanobis'):
            factor_init = torch.eye(embedding_size)  # shape (dim, dim)
            self.covariance_factor = nn.Parameter(factor_init, requires_grad=True)
            self.args['covariance_factor'] = self.covariance_factor
        
        if distance in ('hyperbolic_euclidean', 'hyperbolic_euclidean_v2', 'hyperbolic_mahalanobis'):
            self.w_logit = nn.Parameter(torch.tensor(0.0), requires_grad=True)
            self.args['w_logit'] = self.w_logit 

        if distance == 'hyperbolic_curv':
            # Raw parameter; pass through softplus before use to guarantee c > 0
            self.curvature_raw = nn.Parameter(torch.tensor(0.0), requires_grad=True)
            self.args['curvature'] = self.curvature_raw

        if distance in ('weighted_euclidean', 'weighted_hyperbolic'):
            self.dimension_weights = nn.Parameter(torch.ones(embedding_size), requires_grad=True)
            self.args['dimension_weights'] = self.dimension_weights

        if distance == 'hyperbolic_fixed':
            self.register_buffer('curvature', torch.tensor(float(fixed_curvature)))
            self.args['curvature'] = self.curvature
                

    def normalize_embeddings(self, embeddings, distance_str=None):
        """ Project embeddings to an hypersphere of a certain radius """
        if distance_str is None:
            distance_str = self.distance_str

        if distance_str == 'hyperbolic_curv':
            c = F.softplus(self.curvature_raw).clamp(min=0.1, max=10.0)
            max_norm = (1.0 / torch.sqrt(c)) - 1e-5
            norms = embeddings.norm(dim=-1, keepdim=True)
            # Only shrink vectors that exceed the ball radius; leave smaller ones alone
            clamped = norms.clamp(min=1e-7)
            scale = torch.min(torch.ones_like(clamped), max_norm / clamped)
            return embeddings * scale
        if distance_str == 'hyperbolic_fixed':
            c = self.curvature
            max_norm = (1.0 / torch.sqrt(c)) - 1e-5
            norms = embeddings.norm(dim=-1, keepdim=True)
            clamped = norms.clamp(min=1e-7)
            scale = torch.min(torch.ones_like(clamped), max_norm / clamped)
            return embeddings * scale

        min_scale = 1e-7
        if distance_str in ('hyperbolic', 'hyperbolic_euclidean', 'weighted_hyperbolic', 'hyperbolic_eucliean_v2'):
            max_scale = 1 - 1e-3
        else:
            max_scale = 1e10
        
        if distance_str == 'weighted_hyperbolic':
            weights = self.dimension_weights.to(embeddings.device)
            embeddings = embeddings * weights

        return F.normalize(embeddings, p=2, dim=1) * self.radius.clamp_min(min_scale).clamp_max(max_scale)
    
    def encode(self, sequence, skip_normalization=False):
        """ Use embedding model and normalization to encode some sequences. """
        enc_sequence = self.embedding_model(sequence)
        if skip_normalization:
            return enc_sequence
        if self.scaling is not None:
            enc_sequence = self.normalize_embeddings(enc_sequence)
        return enc_sequence
    

    def forward(self, sequence):
        (B, _, N, _) = sequence.shape
        sequence = sequence.reshape(2 * B, N, -1)
        if self.distance_str in ('hyperbolic_euclidean_v2', 'hyperbolic_mahalanobis'):
            enc_sequence = self.encode(sequence, skip_normalization=True)
            
            enc_sequence_hyp = self.normalize_embeddings(enc_sequence, 'hyperbolic')
            enc_sequence_hyp = enc_sequence_hyp.reshape(B, 2, -1)
            
            enc_sequence_euc = self.normalize_embeddings(enc_sequence, 'euclidean')
            enc_sequence_euc = enc_sequence_euc.reshape(B, 2, -1)

            distance = self.distance(enc_sequence_hyp[:, 0], enc_sequence_hyp[:, 1], 
                                     enc_sequence_euc[:, 0], enc_sequence_euc[:, 1], self.args)

        else:
            enc_sequence = self.encode(sequence)
            enc_sequence = enc_sequence.reshape(B, 2, -1)
            distance = self.distance(enc_sequence[:, 0], enc_sequence[:, 1], self.args)
        return distance
    
    def get_param_groups(self, base_lr, all_same=False, covariance_decay=0.0):
        """Return parameter groups with appropriate learning rates."""
        base_params = []
        moderate_params = []
        decay_params = [] 
        slow_params = []

        # Embedding model parameters
        base_params.extend(self.embedding_model.parameters())

        # Scaling parameter
        if self.scaling is not None:
            if self.distance_str == 'hyperbolic_curv':
                moderate_params.append(self.scaling)
            else:
                base_params.append(self.scaling)

        # Radius parameter
        if self.radius is not None:
            base_params.append(self.radius)

        # Blend weight
        if self.w_logit is not None:
            moderate_params.append(self.w_logit)

        # Covariance factor
        if self.covariance_factor is not None:
            decay_params.append(self.covariance_factor)

        # Curvature
        if self.curvature_raw is not None:
            slow_params.append(self.curvature_raw)

        # Dimension weights
        if self.dimension_weights is not None:
            slow_params.append(self.dimension_weights)

        groups = []
        if base_params:
            groups.append({'params': base_params, 'lr': base_lr})
        if moderate_params:
            if all_same:
                groups.append({'params': moderate_params, 'lr': base_lr})
            else:
                groups.append({'params': moderate_params, 'lr': base_lr * 0.5})
        if slow_params:
            if all_same:
                groups.append({'params': slow_params, 'lr': base_lr})
            else:
                groups.append({'params': slow_params, 'lr': base_lr * 0.3})
        if decay_params:
            if all_same:
                groups.append({'params': decay_params, 'lr': base_lr})
            else:
                groups.append({'params': decay_params, 'lr': base_lr * 0.3, 'weight_decay': covariance_decay})

        return groups
    
    def regularization_loss(self):
        """Return auxiliary regularization loss, or 0 if none needed."""
        reg = 0.0
        if self.covariance_factor is not None:
            # Penalize deviation of B^T B from identity
            I = torch.eye(self.covariance_factor.shape[0], device=self.covariance_factor.device)
            M = self.covariance_factor.T @ self.covariance_factor
            reg = reg + ((M - I) ** 2).sum()
        if self.dimension_weights is not None:
            # Penalize deviation of weights from uniform (ones)
            reg = reg + ((self.dimension_weights - 1.0) ** 2).sum()
        return reg