import os
import SimpleITK as sitk

root_dir = os.path.abspath(os.path.join(os.getcwd(), os.pardir))

def register_images(fixed_image_path, moving_image_path, output_image_path):
    try:
        fixed_image = sitk.ReadImage(fixed_image_path, sitk.sitkFloat32)
        moving_image = sitk.ReadImage(moving_image_path, sitk.sitkFloat32)

        # Print the file paths
        print(f"Registering images: {fixed_image_path}, {moving_image_path}")

        # B-spline registration
        transform_domain_mesh_size=[8]*moving_image.GetDimension() 
        initial_transform = sitk.BSplineTransformInitializer(fixed_image, transform_domain_mesh_size)

        registration_method = sitk.ImageRegistrationMethod()
        registration_method.SetInitialTransform(initial_transform, True)
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
        final_transform = registration_method.Execute(fixed_image, moving_image)

        # Apply the transformation to the moving image
        resampled_image = sitk.Resample(moving_image, fixed_image, final_transform, sitk.sitkLinear, 0.0, moving_image.GetPixelID())

        # Save the registered image
        sitk.WriteImage(resampled_image, output_image_path)
    except Exception as e:
        print(f"Error processing files: {fixed_image_path}, {moving_image_path}. Error: {str(e)}")

def main():
    # Change the working directory to the "data" folder
    os.chdir('../data')

    # Limit to patient folders '000' to '070'
    for i in range(110,140):
        patient_folder = f"{i:03d}"  # Format the folder name with leading zeros

        output_folder = os.path.abspath(patient_folder)

        # Create the output folder for the registered images
        registered_output_folder = os.path.join(output_folder, "registered2")
        os.makedirs(registered_output_folder, exist_ok=True)

        t1w_image_path = os.path.join(output_folder, "normalized", "T1w_1mm_normalized.nii.gz")
        fa_image_path = os.path.join(output_folder, "registered", "FA_registered.nii.gz")
        adc_image_path = os.path.join(output_folder, "registered", "ADC_registered.nii.gz")

        registered_fa_image_path = os.path.join(registered_output_folder, "FA_registered.nii.gz")
        registered_adc_image_path = os.path.join(registered_output_folder, "ADC_registered.nii.gz")

        register_images(t1w_image_path, fa_image_path, registered_fa_image_path)
        register_images(t1w_image_path, adc_image_path, registered_adc_image_path)

if __name__ == "__main__":
    main()

