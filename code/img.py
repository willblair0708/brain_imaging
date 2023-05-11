import os

output_dir = '../data/output'
num_patients = 200

# Create image list files
t1_t2_list = open("t1_t2_image_list.txt", "w")
t1_fa_list = open("t1_fa_image_list.txt", "w")
t1_adc_list = open("t1_adc_image_list.txt", "w")

for patient_number in range(1, num_patients + 1):
    patient_dir = os.path.join(output_dir, f"{patient_number:03d}", "normalized")

    t1_img = os.path.join(patient_dir, "T1w_1mm_normalized.nii.gz")
    t2_img = os.path.join(patient_dir, "T2w_1mm_normalized.nii.gz")
    fa_img = os.path.join(patient_dir, "FA_1.25mm_normalized.nii.gz")
    adc_img = os.path.join(patient_dir, "ADC_1.25mm_normalized.nii.gz")

    t1_t2_list.write(f"{t1_img}\n{t2_img}\n")
    t1_fa_list.write(f"{t1_img}\n{fa_img}\n")
    t1_adc_list.write(f"{t1_img}\n{adc_img}\n")

# Close the image list files
t1_t2_list.close()
t1_fa_list.close()
t1_adc_list.close()
