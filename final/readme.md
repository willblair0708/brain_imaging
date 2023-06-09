# MRI Image Synthesis Project

Find all non-corrupted files in Google Drive: https://drive.google.com/drive/u/0/folders/1ySDDfXsGiV2DQGDblMH0BK3bIGxrK1mt

This project focuses on synthesizing Diffusion Tensor Imaging (DTI) related images, specifically Fractional Anisotropy (FA) and Apparent Diffusion Coefficient (ADC), using T1w and T2w structural images. The project is based on the use of Magnetic Resonance Imaging (MRI) data to study the brain's white matter structure.

## Abstract

Magnetic Resonance Imaging (MRI) plays a crucial role in examining both the structure and function of the brain. Although structural images, such as T1w and T2w, are typically acquired for most brain-related tasks, diffusion MRI (dMRI) is particularly valuable for studying white matter structure but is not routinely collected. The aim of this project is to synthesize two key feature maps, fractional anisotropy (FA) and apparent diffusion coefficient (ADC), from the provided T1w and T2w structural images.

## Keywords

* DTI
* Registration
* Synthesis

## Project Files

Here are the files included in this project:

* `skullstrip.ipynb`: This notebook is for skull stripping in MRI images.
* `normalize.ipynb`: This notebook contains code to normalize the MRI images.
* `affine.ipynb`: This notebook contains methods for affine transformations.
* `bspline.ipynb`: This notebook contains code for B-spline transformations.
* `fa_syn.ipynb`: This notebook is for synthesizing FA images. For the test dataset, the models do not have to be trained.
* `adc_syn.ipynb`: This notebook is for synthesizing FA images. For the test dataset, the models do not have to be trained.
* `resample.py`: This is a Python script for resampling images to the correct resolution.

The files are listed in the order they should be run on the test set.

## Dataset

The dataset for this project consists of MRI data for 200 subjects, each containing four different types of images: T1w, T2w, FA, and ADC. You can download the dataset on OneDrive (link to be provided).

## Steps

The project consists of the following steps:

1. **Skull Stripping**

This step focuses on removing the skull region from the MRI images. HD-BET was used for this purpose. It's important to ensure that the synthesized FA and ADC maps do not contain any skull regions.

2. **Registration**

In this step, all images are aligned with each other. The reference space is the T1w space. It is also important to align the other images (T2w, FA, and ADC) to this space, which we did so using affine registration. Please note that the provided FA and ADC maps have undergone the same geometric distortion, thus, they underwent B-spline deformable registration.

3. **Synthesis**

The last step is the synthesis of the FA and ADC maps from the input T1w and T2w images. We did this by creating a generative adversarial model to synthesize the FA and ADC images. All the previous steps were done in the higher resolution, so a post-processing step for the FA and ADC images was used.

## Trained Models
We created and trained our own generative adversarial models, one for the synthesis of ADC images and one for the synthesis of FA images.

The model weights are in the following files: For the FA model: t1_model_fa.pt, t2_model_fa.pt, generator_fa.pt, discriminator_fa.pt For the ADC model: t1_model_adc.pt, t2_model_adc.pt, generator_adc.pt, discriminator_adc.pt These model weight files will automatically be used in the files to generate images.

Find updated image in google drive as well.

All found in Google Drive: https://drive.google.com/drive/u/0/folders/172T7z1VOD7gAC5uRx7ARjXCyUzUhpN7E

## Contributions

This project is open to contributions. Feel free to propose changes, report issues, or submit pull requests. We look forward to your input!


