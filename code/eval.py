import nibabel as nib
import numpy as np
import os.path as path
from pathlib import Path
import pandas as pd
import argparse

def MAE(img1,img2):
    img1 = np.nan_to_num(img1)
    img2 = np.nan_to_num(img2)
    return np.mean(np.abs(img1-img2))


def argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--student_result_dir', type=str, default='')
    parser.add_argument('--ground_truth_dir', type=str, default='')
    return parser.parse_args()

""" test script: 
$python eval.py --student_result_dir ./Team1 --ground_truth_dir ./ground_truth
"""

if __name__ == '__main__':
    args = argparser()
    student_result_dir = Path(args.student_result_dir)
    ground_truth_dir = Path(args.ground_truth_dir)

    T2w_align_maes = []
    FA_align_maes = []
    ADC_align_maes = []
    FA_syn_maes = []
    ADC_syn_maes = []

    # evaluate registration results
    for i in range(1,11):
        # read the student result
        subject = f'E{i:02d}'
        T2w_align = nib.load( student_result_dir / subject/ 'T2w_align.nii.gz').get_fdata()
        FA_align = nib.load( student_result_dir / subject/ 'FA_align.nii.gz').get_fdata()
        ADC_align = nib.load( student_result_dir / subject/ 'ADC_align.nii.gz').get_fdata()
        # read the ground truth
        T2w = nib.load( ground_truth_dir / subject/ 'T2w_1mm_brain.nii.gz').get_fdata()
        FA = nib.load( ground_truth_dir / subject/ 'FA_brain.nii.gz').get_fdata()
        ADC = nib.load( ground_truth_dir / subject/ 'ADC_brain.nii.gz').get_fdata()
        # compute the MAE
        T2w_align_mae = MAE(T2w_align, T2w)
        FA_align_mae = MAE(FA_align, FA)
        ADC_align_mae = MAE(ADC_align, ADC)
        T2w_align_maes.append(T2w_align_mae)
        FA_align_maes.append(FA_align_mae)
        ADC_align_maes.append(ADC_align_mae)

    # evaluate synthesis results
    for i in range(11,31):
        # read the student result
        subject = f'E{i:02d}'
        FA_syn = nib.load( student_result_dir / subject / 'FA_syn.nii.gz').get_fdata()
        ADC_syn = nib.load( student_result_dir / subject / 'ADC_syn.nii.gz').get_fdata()
        # read the ground truth
        FA = nib.load( ground_truth_dir / subject / 'FA_brain.nii.gz').get_fdata()
        ADC = nib.load( ground_truth_dir / subject / 'ADC_brain.nii.gz').get_fdata()
        # compute the MAE
        FA_syn_mae = MAE(FA_syn, FA)
        ADC_syn_mae = MAE(ADC_syn, ADC)
        FA_syn_maes.append(FA_syn_mae)
        ADC_syn_maes.append(ADC_syn_mae)

    rdict = {'T2w_align_mae': np.mean(T2w_align_maes), 'FA_align_mae': np.mean(FA_align_maes), 'ADC_align_mae': np.mean(ADC_align_maes),
                'FA_syn_mae': np.mean(FA_syn_maes), 'ADC_syn_mae': np.mean(ADC_syn_maes)}
    print(rdict)
    df = pd.DataFrame(rdict, index=[0])
    df.to_csv(f'{student_result_dir.name}_results.csv', index=False)
