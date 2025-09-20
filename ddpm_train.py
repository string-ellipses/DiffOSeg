import os
import yaml
import torch
import ignite.distributed as idist
import ddpm
import sys
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--params', type=str, default='params.yml', help='Parameters YAML file')
    parser.add_argument('--gpu', type=str, default='2', help='CUDA visible devices')
    args = parser.parse_args()                                                              

    # set GPU
    if args.gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    # load params
    with open(args.params, 'r') as f:
        params = yaml.safe_load(f)

    # remove SLURM environment variables
    os.environ.pop("SLURM_JOBID", None)
    os.environ['WANDB_MODE'] = params['wandb_mode']

    params['num_gpus'] = torch.cuda.device_count()  

    if params['distributed']:
        # distributed training
        with idist.Parallel(
                backend="nccl",
                nproc_per_node=torch.cuda.device_count(),
                master_addr="127.0.0.1",
                master_port=27182) as parallel:
            parallel.run(ddpm.run_train, params)
    else:
        # single gpu training
        ddpm.run_train(0, params)


if __name__ == "__main__":
    main()
