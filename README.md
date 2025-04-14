# CUA-O3D: Cross-Modal and Uncertainty-Aware Agglomeration for Open-Vocabulary 3D Scene Understanding

<a href="https://yunzeman.github.io/" style="color:blue;">Yunze Man</a> ·
<a href="https://cs.illinois.edu/about/people/department-faculty/lgui" style="color:blue;">Liang-Yan Gui</a> ·
<a href="https://yxw.web.illinois.edu/" style="color:blue;">Yu-Xiong Wang</a>

<a href="https://tyroneli.github.io/" style="color:blue;">Jinlong Li</a> ·
<a href="https://scholar.google.com/citations?user=PID7Z4oAAAAJ" style="color:blue;">Cristiano Saltori</a> ·
<a href=https://fabiopoiesi.github.io/" style="color:blue;">Fabio Poiesi</a>
<a href=hhttps://scholar.google.co.uk/citations?user=stFCYOAAAAAJ&hl=en" style="color:blue;">Nicu Sebe</a>

[:star: **CVPR 2025**] [[`Project Page`](https://tyroneli.github.io/CUA_O3D/)] [[`arXiv`](https://arxiv.org/abs/2503.16707)] [[`BibTeX`](#BibTex)]

[![Framework: PyTorch](https://img.shields.io/badge/Framework-PyTorch-orange.svg)](https://pytorch.org/) [![arXiv](https://img.shields.io/badge/arXiv-2406.07544-A42C25?style=flat&logo=arXiv&logoColor=A42C25)](https://arxiv.org/abs/2406.07544) [![Project](https://img.shields.io/badge/Project-Page-green?style=flat&logo=Google%20chrome&logoColor=green)](https://yunzeman.github.io/situation3d/) [![YouTube](https://img.shields.io/badge/YouTube-Video-F83323?style=flat&logo=youtube&logoColor=F83323)](https://www.youtube.com/watch?v=IvjZXOs0Ozo) [![GitHub](https://img.shields.io/badge/GitHub-Code-black?style=flat&logo=github&logoColor=white)](https://github.com/YunzeMan/Situation3D) [![License](https://img.shields.io/badge/License-MIT-red.svg)](https://opensource.org/licenses/MIT)

This repository contains the official PyTorch implementation of the paper "Situational Awareness Matters in 3D Vision Language Reasoning" (CVPR 2024). The paper is available on [arXiv](https://arxiv.org/abs/2406.07544). The project page is online at [here](https://yunzeman.github.io/situation3d/).


<p align="center"><img src="imgs/overview.png" alt="outline" width="70%"></p>

<p align="left"><img src="imgs/method.png" alt="outline" width="95%"></p>


## About CUA-O3D
The lack of a large-scale 3D-text corpus has led recent works to distill open-vocabulary knowledge from vision-language models (VLMs). However, these methods typically rely on a single VLM to align the feature spaces of 3D models within a common language space, which limits the potential of 3D models to leverage the diverse spatial and semantic capabilities encapsulated in various foundation models. In this paper, we propose Cross-modal and Uncertainty-aware Agglomeration for Open-vocabulary 3D Scene Understanding dubbed CUA-O3D, the first model to integrate multiple foundation models—such as CLIP, DINOv2, and Stable Diffusion—into 3D scene understanding. We further introduce a deterministic uncertainty estimation to adaptively distill and harmonize the heterogeneous 2D feature embeddings from these models. Our method addresses two key challenges: (1) incorporating semantic priors from VLMs alongside the geometric knowledge of spatially-aware vision foundation models, and (2) using a novel deterministic uncertainty estimation to capture model-specific uncertainties across diverse semantic and geometric sensitivities, helping to reconcile heterogeneous representations during training. Extensive experiments on ScanNetV2 and Matterport3D demonstrate that our method not only advances open-vocabulary segmentation but also achieves robust cross-domain alignment and competitive spatial perception capabilities so as to provide state-of-the-art performance in tasks such as:
- Zero-shot 3D semantic segmentation
- Cross-modal zero-shot segmentation
- Linear probing segmentation

Visit the [CUA-O3D website](https://tyroneli.github.io/CUA_O3D) to explore more details about the project, methodology, and results.


## Todo List
- [X] 2D feature extraction release
- [ ] distillation training release

## Installation

Requirements

- Python 3.x
- Pytorch 1.7.1
- CUDA 11.x or higher

The following installation suppose `python=3.8` `pytorch=1.7.1` and `cuda=11.x`.

- Create a conda virtual environment

  ```
  conda create -n CUA_O3D python=3.8
  conda activate CUA_O3D
  ```
- Clone the repository

  ```
  git clone https://github.com/TyroneLi/CUA_O3D
  ```
- Install the dependencies
  
  1. Install environment dependency

     ```
     pip install -r requirements.txt
     ```

  2. Install [Pytorch 1.7.1](https://pytorch.org/)

     ```
     pip install torch==1.7.1+cu110 torchvision==0.8.2+cu110 -f https://download.pytorch.org/whl/torch_stable.html
     ```
  3. Install MinkowskiEngine from scratch
    ```
     
     ```




## Acknowledgments
We extend our gratitude to all contributors and supporters of the CUA-O3D project. Your valuable insights and contributions drive innovation and progress in the field of **3D and language-based AI systems**.

## Contact
For questions, issues, or collaboration opportunities:
- Submit a ticket on the [issues page](https://github.com/tyroneli/CUA-O3D/issues).
- Visit the [CUA-O3D project website](https://tyroneli.github.io/CUA_O3D/).
- Alternatively, reach out via email: [jinlong.li@unitn.it](mailto:jinlong.li@unitn.it).

## Quick Links
- [CUA-O3D Website](https://tyroneli.github.io/CUA-O3D/)
- [CUA-O3D Code Repository](https://github.com/tyroneli/CUA-O3D)
- [CUA-O3D Paper on arXiv](https://arxiv.org/abs/2411.19774)


## Website License

This project is licensed under the **[Creative Commons Attribution-ShareAlike 4.0 International License](http://creativecommons.org/licenses/by-sa/4.0/)**.

[![Creative Commons License](https://i.creativecommons.org/l/by-sa/4.0/88x31.png)](http://creativecommons.org/licenses/by-sa/4.0/)

For more information, visit the [Creative Commons License page](http://creativecommons.org/licenses/by-sa/4.0/).