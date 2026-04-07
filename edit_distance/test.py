"""
Standalone testing script for edit distance models.
Loads model architecture from the final export file and weights from an epoch checkpoint,
then runs hierarchical clustering and/or extra dataset evaluation.

Usage examples:
    # Run extra dataset testing only
    python -m edit_distance.test --model_path CNN.pkl --checkpoint_path 42.pkl \
        --extr_data_path ./edit_large_1024.pkl --distance hyperbolic --scaling True

    # Run hierarchical clustering only
    python -m edit_distance.test --model_path CNN.pkl --checkpoint_path 42.pkl \
        --hierarchical_data_path ./hc_1024_large.pkl --distance hyperbolic --scaling True

    # Run both
    python -m edit_distance.test --model_path CNN.pkl --checkpoint_path 42.pkl \
        --extr_data_path ./edit_large_1024.pkl \
        --hierarchical_data_path ./hc_1024_large.pkl \
        --distance weighted_euclidean --scaling True

    # With fixed curvature hyperbolic
    python -m edit_distance.test --model_path CNN.pkl --checkpoint_path 42.pkl \
        --extr_data_path ./edit_large_1024.pkl \
        --distance hyperbolic_fixed --scaling True --curv 2.0
"""

import argparse
import sys
import torch
import torch.nn as nn
import numpy as np

from util.data_handling.data_loader import get_dataloaders
from edit_distance.train import load_edit_distance_dataset
from edit_distance.models.pair_encoder import PairEmbeddingDistance
from edit_distance.train import test, test_and_plot, MAPE
from hierarchical_clustering.unsupervised.unsupervised import hierarchical_clustering_testing


def test_arg_parser():
    parser = argparse.ArgumentParser(description='Standalone testing for edit distance models')

    # Model loading
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to final export file containing model_class and model_args (e.g. CNN.pkl)')
    parser.add_argument('--checkpoint_path', type=str, required=True,
                        help='Path to epoch checkpoint file containing full state dict (e.g. 42.pkl)')

    # Device
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='Disables CUDA (GPU)')
    parser.add_argument('--device', type=int, default=0, help='Cuda device number')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')

    # PairEmbeddingDistance constructor arguments
    parser.add_argument('--distance', type=str, default='hyperbolic',
                        help='Type of distance used during training')
    parser.add_argument('--scaling', type=str, default='False',
                        help='Whether scaling was used during training')
    parser.add_argument('--curv', type=float, default=1.0,
                        help='Fixed curvature value (for hyperbolic_fixed)')

    # Testing arguments
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size')
    parser.add_argument('--workers', type=int, default=0, help='Number of workers')
    parser.add_argument('--loss', type=str, default='mse',
                        help='Loss function to use (mse, mape or mae)')
    parser.add_argument('--plot', action='store_true', default=False,
                        help='Plot real vs predicted distances')
    parser.add_argument('--extr_data_path', type=str, default='',
                        help='Dataset for extra edit distance tests')
    parser.add_argument('--hierarchical_data_path', type=str, default='',
                        help='Dataset for hierarchical clustering')

    return parser


def load_model(model_path, checkpoint_path, distance, scaling, curv, device):
    """
    Reconstruct the full model from:
      - model_path: final export with (model_class, model_args, _, _)
      - checkpoint_path: epoch checkpoint with full PairEmbeddingDistance state dict
    """
    # Load architecture info from final export
    saved = torch.load(model_path, map_location=device, weights_only=False)
    model_class, model_args = saved[0], saved[1]

    # Update device
    model_args.device = device

    # Build embedding model (architecture-agnostic)
    embedding_model = model_class(**vars(model_args))

    # Build pair encoder
    model = PairEmbeddingDistance(
        embedding_model=embedding_model,
        distance=distance,
        scaling=scaling,
        fixed_curvature=curv,
    )
    model.to(device)

    # Load full state dict from epoch checkpoint
    state_dict = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(state_dict)
    model.eval()

    return model


def main():
    parser = test_arg_parser()
    args = parser.parse_args()

    if args.extr_data_path == '' and args.hierarchical_data_path == '':
        print('Error: Please specify at least one of --extr_data_path or --hierarchical_data_path')
        sys.exit(1)

    # Set device
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = f'cuda:{args.device}' if use_cuda else 'cpu'
    print('Using device:', device)

    # Set random seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if use_cuda:
        torch.cuda.manual_seed(args.seed)

    # Parse scaling
    scaling = True if args.scaling == 'True' else False

    # Load model
    print(f'Loading architecture from {args.model_path}')
    print(f'Loading weights from {args.checkpoint_path}')
    model = load_model(args.model_path, args.checkpoint_path,
                       args.distance, scaling, args.curv, device)
    print(f'Model loaded with distance: {args.distance}')

    total_params = sum(p.numel() for p in model.parameters())
    print(f'Total params: {total_params}')

    # Select loss
    loss = None
    if args.loss == 'mse':
        loss = nn.MSELoss()
    elif args.loss == 'mae':
        loss = nn.L1Loss()
    elif args.loss == 'mape':
        loss = MAPE

    # Hierarchical clustering
    if args.hierarchical_data_path != '':
        print('\n--- Hierarchical clustering ---')
        with torch.no_grad():
            hierarchical_clustering_testing(
                encoder_model=model,
                data_path=args.hierarchical_data_path,
                batch_size=args.batch_size,
                device=device,
                distance=args.distance,
            )

    # Extra datasets testing
    if args.extr_data_path != '':
        print('\n--- Extra datasets testing ---')
        extr_datasets = load_edit_distance_dataset(args.extr_data_path)
        loaders = get_dataloaders(extr_datasets, batch_size=max(1, args.batch_size // 8), workers=args.workers)
        with torch.no_grad():
            for dset in loaders.keys():
                if args.plot:
                    avg_loss = test_and_plot(model, loaders[dset], loss, device, dset)
                else:
                    avg_loss = test(model, loaders[dset], loss, device)
                print('Results {}: loss = {:.6f}  MAPE {:.4f}'.format(dset, *avg_loss))


if __name__ == '__main__':
    main()