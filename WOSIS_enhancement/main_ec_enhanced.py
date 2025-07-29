import torch
from tabdiff.main import main as tabdiff_main
import argparse
import json
import os

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Training of TabDiff with EC validation')

    # General configs
    parser.add_argument('--dataname', type=str, default='adult', help='Name dataset, one of those in data/ dir')
    parser.add_argument('--mode', type=str, default='train', help='train or test')
    parser.add_argument('--method', type=str, default='tabdiff', help='Currently we only release our model TabDiff. Baselines will be released soon.')
    parser.add_argument('--gpu', type=int, default=0, help='GPU index')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    parser.add_argument('--no_wandb', action='store_true', help='disable wandb')
    parser.add_argument('--exp_name', type=str, default=None, help='Experiment name, used to name log directories and the wandb run name')
    parser.add_argument('--deterministic', action='store_true', help='Whether to make the entire process deterministic, i.e., fix global random seeds')
    
    # Configs for tabdiff
    parser.add_argument('--y_only', action='store_true', help='Train guidance model that only model the target column')
    parser.add_argument('--non_learnable_schedule', action='store_true', help='disable learnable noise schedule')
    
    # Configs for testing tabdiff
    parser.add_argument('--num_samples_to_generate', type=int, default=None, help='Number of samples to be generated while testing')
    parser.add_argument('--ckpt_path', type=str, default=None, help='Path to the model checkpoint to be tested')
    parser.add_argument('--report', action='store_true', help="Report testing mode: this mode sequentially runs <num_runs> test runs and report the avg and std")
    parser.add_argument('--num_runs', type=int, default=20, help="Number of runs to be averaged in the report testing mode")
    
    # Configs for imputation
    parser.add_argument('--impute', action='store_true')
    parser.add_argument('--trial_start', type=int, default=0)
    parser.add_argument('--trial_size', type=int, default=50)
    parser.add_argument('--resample_rounds', type=int, default=1)
    parser.add_argument('--impute_condition', type=str, default="x_t")
    parser.add_argument('--y_only_model_path', type=str, default=None, help="Path to the y_only model checkpoint that will be used as the unconditional guidance model")
    parser.add_argument('--w_num', type=float, default=0.6)
    parser.add_argument('--w_cat', type=float, default=0.6)
    
    # EC validation configs
    parser.add_argument('--ec_config', type=str, default=None, help='Path to EC validation configuration file')
    parser.add_argument('--enable_ec_validation', action='store_true', help='Enable EC validation loss')
    parser.add_argument('--ec_weight', type=float, default=0.1, help='Weight for EC validation loss')
    
    # Multi-GPU configs
    parser.add_argument('--multi_gpu', action='store_true', help='Enable multi-GPU training using DataParallel')
    
    # DataLoader configs to fix worker issues
    parser.add_argument('--num_workers', type=int, default=0, help='Number of DataLoader workers (0 for no multiprocessing)')
    parser.add_argument('--persistent_workers', action='store_true', help='Keep DataLoader workers alive between epochs')

    args = parser.parse_args()

    # check cuda
    if args.gpu != -1 and torch.cuda.is_available():
        args.device = f'cuda:{args.gpu}'
    else:
        args.device = 'cpu'
    
    # Load EC configuration if provided
    if args.ec_config:
        with open(args.ec_config, 'r') as f:
            ec_config = json.load(f)
        
        # Override args with EC config
        args.enable_ec_validation = ec_config.get('enable_ec_validation', False)
        args.ec_weight = ec_config.get('ec_weight', 0.1)
        args.lat_col_idx = ec_config.get('lat_col_idx', None)
        args.lon_col_idx = ec_config.get('lon_col_idx', None)
        args.ec_col_idx = ec_config.get('ec_col_idx', None)
    else:
        # Auto-determine column indices for WOSIS datasets
        if args.enable_ec_validation and args.dataname.startswith('WOSIS'):
            try:
                info_path = f'data/{args.dataname}/info.json'
                with open(info_path, 'r') as f:
                    info = json.load(f)
                
                column_names = info['column_names']
                num_col_idx = info['num_col_idx']
                
                # Find indices in the original data
                lat_col_original = column_names.index('Y')  # Y is latitude (original index 1)
                lon_col_original = column_names.index('X')  # X is longitude (original index 0)
                
                # Map to indices within numerical columns only
                # Note: For regression tasks, target column is moved to the front
                # Original: [X, Y, upper_depth, lower_depth, value_avg]
                # Processed: [value_avg, X, Y, upper_depth, lower_depth]
                # So lat_col_idx and lon_col_idx need to be adjusted by +1
                args.lat_col_idx = num_col_idx.index(lat_col_original) + 1  # Y moved from pos 1 to pos 2
                args.lon_col_idx = num_col_idx.index(lon_col_original) + 1  # X moved from pos 0 to pos 1
                
                # EC is always the target column for WOSIS - simplified approach
                args.ec_col_idx = -1  # Not used in simplified approach
                args.ec_is_target = True  # Always true for WOSIS
                
                print(f"Auto-detected column indices for {args.dataname}:")
                print(f"  Latitude (Y): {args.lat_col_idx}")
                print(f"  Longitude (X): {args.lon_col_idx}")
                print(f"  EC (value_avg): TARGET COLUMN (will be validated against coordinates)")
                
            except Exception as e:
                print(f"Failed to auto-detect column indices: {e}")
                print("Disabling EC validation")
                args.enable_ec_validation = False
                args.lat_col_idx = None
                args.lon_col_idx = None
                args.ec_col_idx = None
                args.ec_is_target = False
        else:
            # Set default values for non-WOSIS datasets
            args.lat_col_idx = None
            args.lon_col_idx = None
            args.ec_col_idx = None
            args.ec_is_target = False
    
    # Call existing tabdiff_main with EC validation support
    tabdiff_main(args) 