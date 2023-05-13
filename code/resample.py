import os
import glob
import SimpleITK as sitk

def normalize_image_intensity(image, output_min=0, output_max=1):
    return sitk.RescaleIntensity(image, outputMinimum=output_min, outputMaximum=output_max)

def main():
    # Set the working directory to the "data/output" folder
    os.chdir('../data/output')

    # Get the list of patient folders
    patient_folders = [folder for folder in os.listdir() if os.path.isdir(folder)]

    for patient_folder in patient_folders:
        input_folder = os.path.abspath(patient_folder)

        # Read the input images
        if patient_folder.startswith('E'):
            fa_file = os.path.join(patient_folder, 'FA_deformed.nii.gz')
            adc_file = os.path.join(patient_folder, 'ADC_deformed.nii.gz')

            # Create the output folder for the preprocessed images if it doesn't exist
            output_folder = os.path.join(input_folder, "normalized")
            if not os.path.exists(output_folder):
                os.makedirs(output_folder)

            # Normalize FA
            fa_image = sitk.ReadImage(fa_file)
            normalized_fa = normalize_image_intensity(fa_image)
            sitk.WriteImage(normalized_fa, os.path.join(output_folder, 'FA_deformed_normalized.nii.gz'))

            # Normalize ADC
            adc_image = sitk.ReadImage(adc_file)
            normalized_adc = normalize_image_intensity(adc_image)
            sitk.WriteImage(normalized_adc, os.path.join(output_folder, 'ADC_deformed_normalized.nii.gz'))

if __name__ == "__main__":
    main()
