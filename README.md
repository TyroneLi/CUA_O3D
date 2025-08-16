<p align="center">
    <h1 align="center">CUA-O3D: Cross-Modal and Uncertainty-Aware Agglomeration for Open-Vocabulary 3D Scene Understanding</h1>
    <p align="center">
        <a href="https://tyroneli.github.io/" style="color:blue;">Jinlong Li</a> ·
        <a href="https://scholar.google.com/citations?user=PID7Z4oAAAAJ" style="color:blue;">Cristiano Saltori</a> ·
        <a href="https://fabiopoiesi.github.io/" style="color:blue;">Fabio Poiesi</a> ·
        <a href="https://scholar.google.com/citations?user=stFCYOAAAAAJ&hl=en" style="color:blue;">Nicu Sebe</a>
    </p>
    <h2 align="center">CVPR 2025</h2>
    <h3 align="center">
        <a href="https://arxiv.org/abs/2503.16707">Paper</a> | <a href="https://tyroneli.github.io/CUA_O3D/">Project Page</a>
    </h3>
    <div align="center"></div>
</p>

<p align="left">
This repository contains the official PyTorch implementation of the paper "CUA-O3D: Cross-Modal and Uncertainty-Aware Agglomeration for Open-Vocabulary 3D Scene Understanding" (CVPR 205). The paper is available on <a href="https://arxiv.org/abs/2503.16707">Arxiv</a>. The project page is online at <a href="https://tyroneli.github.io/CUA_O3D/">CUA-O3D</a>.
</p>
<br>

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
- [X] distillation training release
- [X] linear probing training release
- [x] release datas - <a href="https://drive.google.com/drive/folders/19IuSypv6yscbceVNP2VFCL4LTzPqm1Az?usp=sharing">link</a> 

## 1. Installation

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
     conda install openblas-devel -c anaconda
     git clone https://github.com/NVIDIA/MinkowskiEngine.git
     cd MinkowskiEngine
     python setup.py install --blas_include_dirs=${CONDA_PREFIX}/include --blas=openblas
     ```

## 2. Data Preparation

### ScanNet v2 dataset

Download the [ScanNet](http://www.scan-net.org/) v2 dataset.

Put the downloaded `scans` and `scans_test` folder as follows.

```
CUA_O3D
├── data
│   ├── scannet
│   │   ├── scans
│   │   ├── scans_test
```

Pre-process ScanNet data from 2d multi-view images by adopting Lseg, DINOv2 and Stable Diffusion models.

```
cd 2D_feature_extraction/

```

## 3. 2D Feature Embedding Extraction

  (1) For LSeg feature extraction and projection
  ```
  CUDA_VISIBLE_DEVICES=0 python 2D_feature_extraction/embedding_projection/fusion_scannet_lseg.py \
      --data_dir <ScanNetV2_save_path> \
      --output_dir <save_path_for_lseg_projection_embeddings> \
      --save_aligned False \
      --split train \
      --process_id_range 0,1600
  ```

  We recommend to first extract the 2D multi-view feature embeddings from Lseg model first without specifying the previous projected point mask index, then you need to specify the first 2D model's projected mask index so as to maintain the consistent point mask index during the training. See <a href="https://github.com/TyroneLi/CUA_O3D/blob/main/2D_feature_extraction/embedding_projection/fusion_scannet_lseg.py#L125">here</a>.

  (2) For DINOv2 feature extraction and projection
  ```
  CUDA_VISIBLE_DEVICES=0 python 2D_feature_extraction/embedding_projection/fusion_scannet_dinov2.py \
      --data_dir <ScanNetV2_save_path> \
      --output_dir <save_path_for_DINOv2_projection_embeddings> \
      --save_aligned False \
      --split train \
      --process_id_range 0,1600
  ```

  (3) For Stable Diffusion (SD) feature extraction and projection
  ```
  CUDA_VISIBLE_DEVICES=0 python 2D_feature_extraction/embedding_projection/fusion_scannet_sd.py \
      --data_dir <ScanNetV2_save_path> \
      --output_dir <save_path_for_SD_projection_embeddings> \
      --save_aligned False \
      --split train \
      --process_id_range 0,1600
  ```
After that, specify the corresponding 2D projection embedding path to the config:
<a href="https://github.com/TyroneLi/CUA_O3D/blob/main/config_CUA_O3D/scannet/ours_lseg_ep50_lsegCosine_dinov2L1_SDCosine.yaml#L3">data_root_2d_fused_feature</a> 
<a href="https://github.com/TyroneLi/CUA_O3D/blob/main/config_CUA_O3D/scannet/ours_lseg_ep50_lsegCosine_dinov2L1_SDCosine.yaml#L4">data_root_2d_fused_feature_dinov2</a> 
<a href="https://github.com/TyroneLi/CUA_O3D/blob/main/config_CUA_O3D/scannet/ours_lseg_ep50_lsegCosine_dinov2L1_SDCosine.yaml#L5">data_root_2d_fused_feature_sd</a> 

## 4. 3D Distillation Training

Perform Distillation Training
```
bash run/distill_with_dinov2_sd_adaptiveWeightLoss_demean.sh \
    training_testing_logs/CUA_O3D_LSeg_DINOv2_SD \
    config_CUA_O3D/scannet/ours_lseg_ep50_lsegCosine_dinov2L1_SDCosine.yaml
```

## 5. 3D Evaluation

(1) Perform 2D Fusion Evaluation
```
sh run/eval_with_dinov2_sd.sh \
  training_testing_logs/CUA_O3D_LSeg_DINOv2_SD \
  config_CUA_O3D/scannet/ours_lseg_ep50_lsegCosine_dinov2L1_SDCosine.yaml \
  fusion
```
(2) Perform 2D Distillation Evaluation
```
sh run/eval_with_dinov2_sd.sh \
  training_testing_logs/CUA_O3D_LSeg_DINOv2_SD \
  config_CUA_O3D/scannet/ours_lseg_ep50_lsegCosine_dinov2L1_SDCosine.yaml \
  distill
```
(3) Perform 2D Ensemble Evaluation
```
sh run/eval_with_dinov2_sd.sh \
  training_testing_logs/CUA_O3D_LSeg_DINOv2_SD \
  config_CUA_O3D/scannet/ours_lseg_ep50_lsegCosine_dinov2L1_SDCosine.yaml \
  ensemble
```

## 6. Cross-dataset Generalization
```
sh run/eval_with_dinov2_sd.sh \
  training_testing_logs/CUA_O3D_LSeg_DINOv2_SD \
  config_CUA_O3D/matterport/test_21classes.yaml \
  ensemble
```
```
sh run/eval_with_dinov2_sd.sh \
  training_testing_logs/CUA_O3D_LSeg_DINOv2_SD \
  config_CUA_O3D/matterport/test_40classes.yaml \
  ensemble
```
```
sh run/eval_with_dinov2_sd.sh \
  training_testing_logs/CUA_O3D_LSeg_DINOv2_SD \
  config_CUA_O3D/matterport/test_80classes.yaml \
  ensemble
```
```
sh run/eval_with_dinov2_sd.sh \
  training_testing_logs/CUA_O3D_LSeg_DINOv2_SD \
  config_CUA_O3D/matterport/test_160classes.yaml \
  ensemble
```

## 7. 3D Linear Probing
(1) Concatenate Lseg, DINOv2 and SD to perform linear probing
```
sh run/distill_cat_prob_seg_all.sh \
  config_CUA_O3D/scannet/ours_lseg_ep20_seg.yaml \
  <best_model_saved_path_from_distillation_training>
```
(1) Lseg head to perform linear probing
```
sh run/distill_sep_prob_seg_Lseg.sh \
  config_CUA_O3D/scannet/ours_lseg_ep20_seg.yaml \
  <best_model_saved_path_from_distillation_training>
```
(1) DINOv2 head to perform linear probing
```
sh run/distill_sep_prob_seg_DINOv2.sh \
  config_CUA_O3D/scannet/ours_lseg_ep20_seg.yaml \
  <best_model_saved_path_from_distillation_training>
```
(1) SD head to perform linear probing
```
sh run/distill_sep_prob_seg_SD.sh \
  config_CUA_O3D/scannet/ours_lseg_ep20_seg.yaml \
  <best_model_saved_path_from_distillation_training>
```

## BibTeX
If you use our work in your research, please cite our publication:
```bibtex
@inproceedings{li2025cross,
  title={Cross-modal and uncertainty-aware agglomeration for open-vocabulary 3d scene understanding},
  author={Li, Jinlong and Saltori, Cristiano and Poiesi, Fabio and Sebe, Nicu},
  booktitle={Proceedings of the Computer Vision and Pattern Recognition Conference},
  pages={19390--19400},
  year={2025}
}
```

## Acknowledgments
We extend our gratitude to all contributors and supporters of the CUA-O3D project. Your valuable insights and contributions drive innovation and progress in the field of **3D and language-based AI systems**.

## Website License

This project is licensed under the **[Creative Commons Attribution-ShareAlike 4.0 International License](http://creativecommons.org/licenses/by-sa/4.0/)**.

[![Creative Commons License](https://i.creativecommons.org/l/by-sa/4.0/88x31.png)](http://creativecommons.org/licenses/by-sa/4.0/)

For more information, visit the [Creative Commons License page](http://creativecommons.org/licenses/by-sa/4.0/).
