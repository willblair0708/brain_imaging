import os
import nibabel as nib
import numpy as np
from scipy.ndimage import zoom

def resample_image(input_path, output_path, new_shape):
    img = nib.load(input_path)
    data = img.get_fdata()

    # Calculate the zoom factors for each dimension
    zoom_factors = np.divide(new_shape, data.shape)

    # Resample the image to the new shape
    resampled_data = zoom(data, zoom_factors)

    # Update the image header with the new shape
    img.header.set_data_shape(new_shape)

    # Save the resampled image
    resampled_img = nib.Nifti1Image(resampled_data, img.affine, img.header)
    nib.save(resampled_img, output_path)

def main():
    # Change the working directory to the "data" folder
    os.chdir('../data/output8')

    # Limit to patient folders 'E01' to 'E10'
    for i in range(1, 11):
        patient_folder = f"E{i:02d}"  # Format the folder name with leading zeros

        output_folder = os.path.abspath(patient_folder)

        # Create the output folder for the resampled images
        resampled_output_folder = os.path.join(output_folder, "resampled")
        os.makedirs(resampled_output_folder, exist_ok=True)

        fa_image_path = os.path.join(output_folder, "registered2", "FA_align.nii.gz")
        adc_image_path = os.path.join(output_folder, "registered2", "ADC_align.nii.gz")

        resampled_fa_image_path = os.path.join(resampled_output_folder, "FA_align.nii.gz")
        resampled_adc_image_path = os.path.join(resampled_output_folder, "ADC_align.nii.gz")

        new_shape = (145, 174, 145)
        resample_image(fa_image_path, resampled_fa_image_path, new_shape)
        resample_image(adc_image_path, resampled_adc_image_path, new_shape)

if __name__ == "__main__":
    main()
