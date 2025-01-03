# ProjectedEx
This is the code repository for the paper:
> ProjectedEx: Enhancing Generation in Explainable AI for Prostate Cancer
> 
> Xuyin Qi, Zeyu Zhang, Aaron Berliano Handoko, Huazhan Zheng, Mingxi Chen, Ta Duc Huy, Vu Minh Hieu Phan, Lei Zhang, Linqi Cheng, Shiyu Jiang, Zhiwei Zhang, Zhibin Liao, Yang Zhao, Minh-Son To
> 
> [arXiv](https://arxiv.org/abs/2501.01392)
## Introduction
Prostate cancer, a growing global health concern,
necessitates precise diagnostic tools, with Magnetic Resonance
Imaging (MRI) offering high-resolution soft tissue imaging that
significantly enhances diagnostic accuracy. Recent advancements
in explainable AI and representation learning have significantly
improved prostate cancer diagnosis by enabling automated and
precise lesion classification. However, existing explainable AI
methods, particularly those based on frameworks like generative
adversarial networks (GANs), are predominantly developed for
natural image generation, and their application to medical
imaging often leads to suboptimal performance due to the unique
characteristics and complexity of medical image. To address
these challenges, our paper introduces three key contributions.
First, we propose ProjectedEx, a generative framework that
provides interpretable, multi-attribute explanations, effectively
linking medical image features to classifier decisions. Second, we
enhance the encoder module by incorporating feature pyramids,
which enables multiscale feedback to refine the latent space and
improves the quality of generated explanations. Additionally, we
conduct comprehensive experiments on both the generator and
classifier, demonstrating the clinical relevance and effectiveness
of ProjectedEx in enhancing interpretability and supporting the
adoption of AI in medical settings.
![Overview of the proposed network.](https://github.com/Richardqiyi/ProjectedEx/blob/main/architecture.png)

## Environment Setup
```
# create a clean conda environment from scratch
conda create --name python=3.10
conda activate dymultidepth
# install pip
conda install ipython
conda install pip
# install required packages
pip install -r requirements.txt
```
## Data
The dataset should be organized as follows:
```
  data
   └── ProstateCa
       ├── train
       │   ├── 1.jpg
       │   ├── 2.jpg
       │   ├── ...
       └── valid
           ├── ...
```
## Training
Modify the parameters and run:
```
python cli.py
```


  


