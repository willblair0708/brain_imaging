# Methods

For registration, we tried multiple different approaches:
- Simple ITK B-Spline Registration - https://simpleitk.readthedocs.io/en/master/link_ImageRegistrationMethodBSpline1_docs.html 
- Simple ITK Diffeomorphic Demons Registration - https://simpleitk.org/doxygen/latest/html/classitk_1_1simple_1_1DiffeomorphicDemonsRegistrationFilter.html 
- LDDMM (large deformation diffeomorphic metric mapping) - https://github.com/brianlee324/torch-lddmm/tree/master
- Voxelmorph - https://github.com/voxelmorph/voxelmorph

# Workflow

The order of files that should be run on the test set:
1. skullstrip.ipynb - the skull stripping
2. prepare.ipynb - normalizing images
3. registration.ipynb - affine registration
4. registration2.ipynb - deformable B-spline registration, only need to be run on the FA and ADC files
5. finalmodel_fa_hr.ipynb - GAN to synthesize FA images; however, on the test set, the training loop does not need to be run as the model weights will be loaded from the saved files
6. finalmodel_adc_hr.ipynb - GAN to synthesize ADC images; however, on the test set, the training loop does not need to be run as the model weights will be loaded from the saved files

# Trained Models
We created and trained our own generative adversial models, one for the synthesis of ADC images and one for the synthesis of FA images.

The model weights are in the following files:
For the FA model: t1_model_fa.pt, t2_model_fa.pt, generator_fa.pt, discriminator_fa.pt
For the ADC model: t1_model_adc.pt, t2_model_adc.pt, generator_adc.pt, discriminator_adc.pt
These model weight files will automatically be used in the files to generate images.
