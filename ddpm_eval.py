import os
import random
import argparse
import numpy as np
import torch
import yaml
from evaluation.evaluate_lidc_uncertainty import eval_lidc_uncertainty

def set_seeds(seed: int):
    """Function that sets all relevant seeds (by Claudio)
    :param seed: Seed to use
    """
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed % 2**32)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def main():
    parser = argparse.ArgumentParser(description='Evaluate LIDC Uncertainty')
    parser.add_argument('--params', type=str, default='params_eval.yml', 
                       help='Parameters YAML file')
    parser.add_argument('--gpu', type=str, default='1', 
                       help='CUDA visible devices')
    parser.add_argument('--seed', type=int, default=0,
                       help='Random seed')
    args = parser.parse_args()

    # Set GPU
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    
    # Set seeds
    set_seeds(args.seed)

    # Load parameters
    with open(args.params, 'r') as f:
        params = yaml.safe_load(f)
    
    # Execute evaluation based on dataset
    if 'lidc' in params['dataset_file']:
        eval_lidc_uncertainty(params)   
    else:
        raise ValueError("Unknown dataset")

if __name__ == "__main__":
    main()