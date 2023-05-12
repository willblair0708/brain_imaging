import os
import SimpleITK as sitk

root_dir = os.path.abspath(os.path.join(os.getcwd(), os.pardir))

def register_images(fixed_image_path, moving_image_path, output_image_path):
    fixed_image = sitk.ReadImage(fixed_image_path, sitk.sitkFloat32)
    moving_image = sitk.ReadImage(moving_image_path, sitk.sitkFloat32)

    # Resample the moving image to match the fixed image resolution
    resampler = sitk.ResampleImageFilter()
    resampler.SetSize(fixed_image.GetSize())
    resampler.SetOutputSpacing(fixed_image.GetSpacing())
    resampler.SetOutputDirection(moving_image.GetDirection())
    resampler.SetOutputOrigin(moving_image.GetOrigin())
    resampler.SetTransform(sitk.Transform())
    resampler.SetInterpolator(sitk.sitkLinear)
    moving_image_resampled = resampler.Execute(moving_image)

    # Affine registration
    initial_transform = sitk.CenteredTransformInitializer(fixed_image, moving_image_resampled, sitk.AffineTransform(fixed_image.GetDimension()))
    registration_method = sitk.ImageRegistrationMethod()
    registration_method.SetInitialTransform(initial_transform)
    registration_method.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
    registration_method.SetMetricSamplingStrategy(registration_method.RANDOM)
    registration_method.SetMetricSamplingPercentage(0.01)
    registration_method.SetInterpolator(sitk.sitkLinear)
    registration_method.SetOptimizerAsGradientDescent(learningRate=1.0, numberOfIterations=100)
    registration_method.SetOptimizerScalesFromPhysicalShift()
    registration_method.SetShrinkFactorsPerLevel(shrinkFactors=[4, 2, 1])
    registration_method.SetSmoothingSigmasPerLevel(smoothingSigmas=[2, 1, 0])
    registration_method.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()

    # Execute registration
    final_transform = registration_method.Execute(fixed_image, moving_image_resampled)

    # Apply the transformation to the moving image
    resampled_image = sitk.Resample(moving_image_resampled, fixed_image, final_transform, sitk.sitkLinear, 0.0, moving_image.GetPixelID())

    # Save the registered image
    sitk.WriteImage(resampled_image, output_image_path)

def main():
    # Change the working directory to the "data" folder
    os.chdir('../data/output')

    # Get the list of patient folders
    patient_folders = [folder for folder in os.listdir() if os.path.isdir(folder)]

    # Register images to T1w space using Diffeomorphic Demons
    for patient_folder in patient_folders:
        output_folder = os.path.abspath(patient_folder)

        # Create the output folder for the registered images
        registered_output_folder = os.path.join(output_folder, "registered")
        os.makedirs(registered_output_folder, exist_ok=True)

        t1w_image_path = os.path.join(output_folder, "normalized", "T1w_1mm_normalized.nii.gz")
        t2w_image_path = os.path.join(output_folder, "normalized", "T2w_1mm_normalized.nii.gz")
        fa_image_path = os.path.join(output_folder, "normalized", "FA_1.25mm_normalized.nii.gz")
        adc_image_path = os.path.join(output_folder, "normalized", "ADC_1.25mm_normalized.nii.gz")

        registered_t2w_image_path = os.path.join(registered_output_folder, "T2w_registered.nii.gz")
        registered_fa_image_path = os.path.join(registered_output_folder, "FA_registered.nii.gz")
        registered_adc_image_path = os.path.join(registered_output_folder, "ADC_registered.nii.gz")

        register_images(t1w_image_path, t2w_image_path, registered_t2w_image_path)
        register_images(t1w_image_path, fa_image_path, registered_fa_image_path)
        register_images(t1w_image_path, adc_image_path, registered_adc_image_path)

if __name__ == "__main__":
    main()

