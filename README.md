# DiffOSeg-Code
👋This repository contains the official pytorch implementation of our MICCAI 2025 paper "[DiffOSeg: Omni Medical Image Segmentation via Multi-Expert Collaboration Diffusion Model](https://arxiv.org/abs/2507.13087)".

[![arXiv](https://img.shields.io/badge/arXiv-2507.13087-b31b1b.svg)](https://arxiv.org/abs/2507.13087)
## Updates
- [2025.08.27]🔥 Our work has been **shortlisted** for the **MICCAI 2025 Best Paper and Young Scientist Awards**, ranking among the top 25 of 1014 accepted papers (from 3447 submissions) !
- [2025.06.18]📩 Our work has been accepted by **MICCAI 2025** !
- [2024.10.23]🥈 We won **2nd** place on both tasks of [MMIS-2024@ACM MM 2024](https://mmis2024.com/) !
## Method
In this study, we propose DiffOSeg, a two-stage diffusion-based framework, which aims to simultaneously achieve both consensus-driven (combining all experts' opinions) and preference-driven (reflecting experts' individual assessments) segmentation. Stage I establishes population consensus through a probabilistic consensus strategy, while Stage II captures expert-specific preference via adaptive prompts. For more details, please refer to [our paper](https://arxiv.org/abs/2507.13087).
<div align="center">
  <img width="892" height="518" alt="image" src="https://github.com/user-attachments/assets/48258cf3-0038-4e42-bd8c-64f4eb25e911" />
</div>

## Usage

### Task-List
- [ ] Add NPC-170 process.
- [ ] Polish code.

### Installation & Data Preparation
See [INSTALL.md](INSTALL.md) for the installation of dependencies and dataset preperation required to run this codebase.

### Training 
Specify parameters such as stage in params.yml
```
python ddpm_train.py --params params.yml --gpu gpu_id
```
### Inference
Specify parameters such as stage in params_eval.yml
```
python ddpm_eval.py  --params params_eval.yml --gpu gpu_id
```

## Citation
If you found this repository useful to you, please consider giving a star ⭐️ and citing our paper:
```
@article{zhang2025diffoseg,
  title={DiffOSeg: Omni Medical Image Segmentation via Multi-Expert Collaboration Diffusion Model},
  author={Zhang, Han and Luo, Xiangde and Chen, Yong and Li, Kang},
  journal={arXiv preprint arXiv:2507.13087},
  year={2025}
}
```
## Acknowledgements
Greatly appreciate the tremendous effort for the following projects: 

[ccdm-stochastic-segmentation](https://github.com/LarsDoorenbos/ccdm-stochastic-segmentation), [D-Persona](https://github.com/ycwu1997/D-Persona), [PromptIR](https://github.com/hellopipu/PromptMR), [PromptMR](https://github.com/hellopipu/PromptMR), [UniSeg](https://github.com/yeerwen/UniSeg)
