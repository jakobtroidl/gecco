#!/usr/bin/env python3
"""
Command line tool for running inference with GECCO model on neuron data.
"""

import argparse
import torch
import os
import sys

import gecco_torch
from gecco_torch.scatter import plot, combine_plots
from gecco_torch.data.neurons import TorchNeuronNet
from gecco_torch.utils import apply_transform
from gecco_torch.utils import apply_transform

from torch.utils.data import DataLoader
from cloudvolume import CloudVolume
from gecco_torch.data.interfaces import PoCADuckInterface


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Run inference with GECCO model on neuron data',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        '--config-root',
        type=str,
        default='/groups/turaga/home/troidlj/gecco/gecco-torch/example_configs',
        help='Path to the directory containing model configurations'
    )
    
    parser.add_argument(
        '--config-name',
        type=str,
        default='neuron_conditional.py',
        help='Name of the configuration file'
    )
    
    parser.add_argument(
        '--checkpoint',
        type=str,
        default='lightning_logs/version_38/arxiv/epoch=7-step=40000.ckpt',
        help='Path to model checkpoint (relative to config-root or absolute path)'
    )

    parser.add_argument(
        '--poca-duck-path',
        type=str,
        default='/nrs/turaga/jakob/autoproof_data/flywire_cave_post/points',
        help='Path to the PoCADuck data'
    )

    parser.add_argument(
        '--data-root',
        type=str,
        default='/nrs/turaga/jakob/autoproof_data/flywire_cave_post/cache_t5',
        help='Root directory for neuron data'
    )
    
    parser.add_argument(
        '--split',
        type=str,
        default='train',
        choices=['train', 'val', 'test'],
        help='Data split to use'
    )
    
    parser.add_argument(
        '--n-points',
        type=int,
        default=2048,
        help='Number of points in point cloud'
    )
    
    parser.add_argument(
        '--idx',
        type=int,
        default=276,
        help='Index of the sample to use from the dataset'
    )
    
    parser.add_argument(
        '--n-samples',
        type=int,
        default=2,
        help='Number of samples to generate'
    )
    
    parser.add_argument(
        '--device',
        type=str,
        default='cuda',
        choices=['cuda', 'cpu'],
        help='Device to use for inference'
    )
    
    parser.add_argument(
        '--no-progress-bar',
        action='store_true',
        help='Disable progress bar during sampling'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default="/nrs/turaga/jakob/autoproof_data/flywire_cave_post/debug",
        help='Directory to save plots (if not specified, plots are displayed)'
    )
    
    parser.add_argument(
        '--save-points',
        action='store_true',
        help='Save generated point clouds as .pt files'
    )
    
    parser.add_argument(
        '--batch-size',
        type=int,
        default=1,
        help='Batch size for data loading'
    )

    parser.add_argument(
        '--cloud-volume-path', 
        type=str,
        default='graphene://https://prod.flywire-daf.com/segmentation/1.0/flywire_public/',
        help='Path to the cloud volume data'
    )

    parser.add_argument(
        '--timestamp',
        type=int,
        default=1749932687,
        help='Unix Timestamp for the cloud volume data (if graphene is used)'
    )

    parser.add_argument(
        '--voxel-size',
        type=list[int],
        default=[4, 4, 40],
        help='Voxel size for the cloud volume data (if graphene is used)'
    )

    return parser.parse_args()


def load_model(config_path, checkpoint_path, device):
    """Load the model from config and checkpoint."""
    print(f"Loading config from: {config_path}")
    config = gecco_torch.load_config(config_path)
    model = config.model
    
    print(f"Loading checkpoint from: {checkpoint_path}")
    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict['ema_state_dict'])
    model = model.eval()
    
    return model


def load_data(data_root, split, n_points):
    """Load the neuron dataset."""
    print(f"Loading data from: {data_root}, split: {split}")
    data = TorchNeuronNet(data_root, split, n_points)
    print(f"Dataset loaded with {len(data)} samples")
    return data


def run_inference(model, partial, n_samples, n_points, with_pbar):
    """Run inference to generate samples."""
    print(f"Generating {n_samples} samples...")
    samples = model.sample_stochastic(
        (n_samples, n_points, 3),
        context=partial,
        with_pbar=with_pbar,
    )
    return samples

def save_point_clouds(partial, generated, superset, gt, output_dir, idx):
    """Save point clouds as .pt files."""
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        
        torch.save(partial.cpu(), os.path.join(output_dir, f'partial_{idx}.pt'))
        torch.save(generated.cpu(), os.path.join(output_dir, f'generated_{idx}.pt'))
        torch.save(superset.cpu(), os.path.join(output_dir, f'superset_{idx}.pt'))
        torch.save(gt.cpu(), os.path.join(output_dir, f'gt_{idx}.pt'))

        print(f"Point clouds saved to: {output_dir}")


def main():
    """Main function."""
    args = parse_args()
    
    # Set device
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        device = 'cpu'
    else:
        device = args.device
    
    # Prepare paths
    config_path = os.path.join(args.config_root, args.config_name)
    if os.path.isabs(args.checkpoint):
        checkpoint_path = args.checkpoint
    else:
        checkpoint_path = os.path.join(args.config_root, args.checkpoint)
    
    # Check if files exist
    if not os.path.exists(config_path):
        print(f"Error: Config file not found: {config_path}")
        sys.exit(1)
    
    if not os.path.exists(checkpoint_path):
        print(f"Error: Checkpoint file not found: {checkpoint_path}")
        sys.exit(1)
    
    if not os.path.exists(args.data_root):
        print(f"Error: Data root not found: {args.data_root}")
        sys.exit(1)
    
    try:
        # Load model
        model = load_model(config_path, checkpoint_path, device)
        dataset = load_data(args.data_root, args.split, args.n_points)
        cv = CloudVolume(args.cloud_volume_path, timestamp=args.timestamp, agglomerate=True, mip=0, use_https=True)
        di = PoCADuckInterface(args.poca_duck_path)

        # Create DataLoader for the dataset
        data_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
        
        # Check if index is valid
        if args.idx >= len(dataset):
            print(f"Error: Index {args.idx} is out of range. Dataset has {len(dataset)} samples.")
            sys.exit(1)

        # iterate over the data loader to get the sample
        for i, data in enumerate(data_loader):
            partial = data.partial
            points = data.points
            T_i = data.T_i

            print(f"Processing sample {i}")
            print(f"Partial shape: {partial.shape}")
            print(f"Ground truth shape: {points.shape}")
            
            # Run inference
            out = run_inference(
                model, 
                partial, 
                1, # for now only generate one hypothesis
                args.n_points, 
                with_pbar=not args.no_progress_bar
            )

            out = out.squeeze(0)  # remove batch dimension

            denormalized_out = apply_transform(T_i, out).squeeze()
            denormalized_partial = apply_transform(T_i, partial).squeeze()
            denormalized_gt = apply_transform(T_i, points).squeeze()


            denormalized_samples_list = [(int(x.item()), int(y.item()), int(z.item())) for x, y, z in denormalized_out.cpu().numpy()]
            labels = cv.scattered_points(denormalized_samples_list)
            labels = torch.tensor([int(v.item()) if hasattr(v, 'item') else int(v) for v in labels.values()], dtype=torch.long).to(device)

            # compute histogram of labels
            unique_labels, counts = torch.unique(labels, return_counts=True)

            # remove all labels with less than 20 points
            mask = counts >= 50
            unique_labels = unique_labels[mask]
            counts = counts[mask]

            all_pc = []
            all_pc.append(denormalized_partial.squeeze(0))

            for l in unique_labels.cpu().numpy().astype('uint64').tolist():
                if l != 0:
                    label_int = l
                    fragment_pc, _ = di.load_pc(label_int)
                    all_pc.append(fragment_pc)

            superset = torch.cat(all_pc, dim=0)
            print(f"Generated samples labels: {unique_labels}, counts: {counts}")

            # visualize_results(partial, denormalized_samples, points, T_i, args.output_dir)
            save_point_clouds(denormalized_partial, denormalized_out, superset, denormalized_gt, args.output_dir, i)

        print("Inference completed successfully!")
        
    except Exception as e:
        print(f"Error during inference: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()
