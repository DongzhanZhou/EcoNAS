# EcoNAS: Finding Proxies for Economical Neural Architecture Search

Dongzhan Zhou, Xinchi Zhou, Wenwei Zhang, Chen Change Loy, Shuai Yi, Xuesen Zhang, Wanli Ouyang

## Introduction

In this work, we systemically evaluate the behaviors of common reduction factors in neural architecture search and report our findings. We observe that most existing proxies exhibit different behaviors in maintaining the rank consistency among network candidates and some proxies can be reliable and efficient simultaneously. Based on our observations, we propose an economical evolutionary-based NAS (EcoNAS), which achieves an impressive 400x search cost reduction yet getting better results.

## Requirements

```shell
Python >= 3.5, PyTorch >= 0.4.0, torchvision >= 0.2.0
```

The CIFAR-10 dataset will be automatically downloaded by torchvision.

## Usage

### Search

EcoNAS is composed of two parts, i.e., reliable proxy (c4r4s0e60 in our paper) and hierarchical proxy strategy.

```shell
cd EcoNAS/search

# if you want to use both reliable proxy and hierarchical proxy strategy
python main.py

# if you prefer to use reliable proxy only
python main_single.py
```

Please note that EcoNAS executes the search process by generating shell scripts at each search cycle. You need to modify the `train_parallel` functions in the files to produce the correct scripts for your server.

After searching, run scanresult.py to show the top-k models, which will be retrained subsequently.

### Retrain

After searching, move to the retrain folder and train the full architecture from scratch under the normal setting (c0r0s0e600).

```shell
cd EcoNAS/retrain

python train.py --data=YOUR_PATH_TO_CIFAR10 --save=ckpts/MODEL_NAME
```

Note: As the model structure is based on normal and reduction genotypes, please copy the `arch.py` of the target model to the ckpt path (e.g., ckpts/a1) so that the structure can be built correctly. We adopt [CutOut](https://arxiv.org/abs/1708.04552) and [path dropout](https://openreview.net/forum?id=S1VaB4cex) to further enhance the model performance.

## Citation

If you find our work useful, please consider to cite:

```latex
@inproceedings{zhou2020econas,
  title={Econas: Finding proxies for economical neural architecture search},
  author={Zhou, Dongzhan and Zhou, Xinchi and Zhang, Wenwei and Loy, Chen Change and Yi, Shuai and Zhang, Xuesen and Ouyang, Wanli},
  booktitle={Proceedings of the IEEE/CVF Conference on computer vision and pattern recognition},
  pages={11396--11404},
  year={2020}
}
```

## Acknowledgement

This code benefits from the excellent work [DARTS](https://github.com/quark0/darts).
