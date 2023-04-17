%% Step A: Skull stripping

% BET is a part of FSL, so make sure you have it installed on your system
% and set the FSL environment variables correctly.
% Check https://fsl.fmrib.ox.ac.uk/fsl/fslwiki/FslInstallation for details.

% Input image file names
t1w_filename = 'T1w_1mm.nii.gz';
t2w_filename = 'T2w_1mm_noalign.nii.gz';

% Output file names for skull-stripped images
t1w_brain_filename = 'T1w_1mm_brain.nii.gz';
t2w_brain_filename = 'T2w_1mm_brain.nii.gz';

% Skull stripping with BET
system(['bet ' t1w_filename ' ' t1w_brain_filename]);
system(['bet ' t2w_filename ' ' t2w_brain_filename]);

%% Step B: Registration

% Output file names for registered images
t2w_registered_filename = 'T2w_registered.nii.gz';
fa_registered_filename = 'FA_registered.nii.gz';
adc_registered_filename = 'ADC_registered.nii.gz';

% Perform registration
% Here, we assume that you have an FA and ADC image
fa_filename = 'FA_deformed.nii.gz';
adc_filename = 'ADC_deformed.nii.gz';

% Register T2w to T1w
system(['flirt -in ' t2w_brain_filename ' -ref ' t1w_brain_filename ' -out ' t2w_registered_filename]);

% Register FA to T1w
system(['flirt -in ' fa_filename ' -ref ' t1w_brain_filename ' -out ' fa_registered_filename]);

% Register ADC to T1w
system(['flirt -in ' adc_filename ' -ref ' t1w_brain_filename ' -out ' adc_registered_filename]);

%% Step C: Synthesis

% Read the registered images
t1w_brain_image = niftiread(t1w_brain_filename);
t2w_registered_image = niftiread(t2w_registered_filename);

% Synthesize FA and ADC maps
% As an example, we will use a simple linear regression model for synthesis.
% In practice, you may want to use a more sophisticated approach, such as a deep learning model.

% Perform the linear regression on the registered images
X = [t1w_brain_image(:), t2w_registered_image(:)];
Y_FA = X \ fa_registered_filename(:); % Regression for FA synthesis
Y_ADC = X \ adc_registered_filename(:); % Regression for ADC synthesis

% Synthesize FA and ADC maps
FA_synthesized = reshape(X * Y_FA, size(t1w_brain_image));
ADC_synthesized = reshape(X * Y_ADC, size(t1w_brain_image));

% Save the synthesized FA and ADC maps
niftiwrite(FA_synthesized, 'FA_synthesized.nii.gz');
niftiwrite(ADC_synthesized, 'ADC_synthesized.nii.gz');
