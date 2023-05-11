import os
import subprocess
import numpy as np
import tensorflow as tf
assert tf.__version__.startswith('2.'), 'This tutorial assumes Tensorflow 2.0+'

import inspect
if not hasattr(inspect, 'getargspec'):
    inspect.getargspec = inspect.getfullargspec
import voxelmorph as vxm
import neurite as ne
import nibabel as nib
    # initial shape (182, 218, 182)

physical_devices = tf.config.list_physical_devices('GPU')
print("Num GPUs Available: ", len(physical_devices))

os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # Adjust the GPU device index if needed
tf.config.set_visible_devices(physical_devices[0], 'GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

vol_shape = (192, 224, 192)
nb_features = [
    [16, 32, 32, 32],
    [32, 32, 32, 32, 32, 16, 16]
]

vxm_model = vxm.networks.VxmDense(vol_shape, nb_features, int_steps=0)

# losses and loss weights
losses = ['mse', vxm.losses.Grad('l2').loss]
loss_weights = [1, 0.01]

# Initialize arrays for holding image data
t1w_images = []
t2w_images = []

# Loop over patient folders
for i in range(1, 200):
    if i == 163:
        continue

    # Format the patient number with leading zeros
    patient_number = str(i).zfill(3)
    
    # Load T1w image
    img1 = nib.load(f"../data/output/{patient_number}/normalized/T1w_1mm_normalized.nii.gz")
    data1 = img1.get_fdata()
    data1 = np.pad(data1, [(5, 5), (3, 3), (5, 5)], mode="constant")
    t1w_images.append(data1)
    
    # Load T2w image
    img2 = nib.load(f"../data/output/{patient_number}/registered/T2w_registered.nii.gz")
    data2 = img2.get_fdata()
    data2 = np.pad(data2, [(5, 5), (3, 3), (5, 5)], mode="constant")
    t2w_images.append(data2)


# Stack images together
x_train = np.stack((t1w_images, t2w_images), axis=1)

vxm_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4), loss=losses, loss_weights=loss_weights)

def vxm_data_generator(x_data, batch_size=1):
    """
    Generator that takes in data of size [N, 2, H, W, D], and yields data for
    our custom vxm model. Note that we need to provide numpy data for each
    input, and each output.
    
    Here N is the number of patients, H, W, D are the dimensions of the images
    and 2 represents the two modalities (T1w and T2w).

    inputs:  moving image [bs, H, W, D, 1], fixed image [bs, H, W, D, 1]
    outputs: moved image [bs, H, W, D, 1], zero-gradient [bs, H, W, D, 3]
    """

    # preliminary sizing
    vol_shape = x_data.shape[2:5] # extract data shape
    ndims = len(vol_shape)
    
    # prepare a zero array the size of the deformation
    zero_phi = np.zeros([batch_size, *vol_shape, ndims])
    
    # calculate number of batches in an epoch
    num_batches = int(np.ceil(x_data.shape[0] / batch_size))
    
    while True:
        for i in range(num_batches):
            # prepare inputs:
            # images need to be of the size [batch_size, H, W, D, 1]
            moving_images = x_data[i*batch_size:(i+1)*batch_size, 0, ..., np.newaxis]
            fixed_images = x_data[i*batch_size:(i+1)*batch_size, 1, ..., np.newaxis]
            inputs = [moving_images, fixed_images]
            
            # prepare outputs (the 'true' moved image):
            # we don't have this, but we know we want to compare 
            # the resulting moved image with the fixed image. 
            # we also wish to penalize the deformation field. 
            outputs = [fixed_images, zero_phi]
            
            yield (inputs, outputs)

train_generator = vxm_data_generator(x_train, batch_size=8)
in_sample, out_sample = next(train_generator)

# visualize
images = [img[0, :, :, 40] for img in in_sample + out_sample]
titles = ['moving', 'fixed', 'moved ground-truth (fixed)', 'zeros']
ne.plot.slices(images, titles=titles, cmaps=['gray'], do_colorbars=True);

hist = vxm_model.fit(train_generator, epochs=5, steps_per_epoch=5, verbose=2)

vxm_model.save('my_model.h5')

# Load the model
vxm_model.load_weights('my_model.h5')

# Load your T1w and T2w images
t1w_img_path = "path_to_your_t1w_image.nii.gz"
t2w_img_path = "path_to_your_t2w_image.nii.gz"

t1w_img = nib.load(t1w_img_path).get_fdata()
t2w_img = nib.load(t2w_img_path).get_fdata()

# Pad the images
t1w_img = np.pad(t1w_img, [(5, 5), (3, 3), (5, 5)], mode="constant")
t2w_img = np.pad(t2w_img, [(5, 5), (3, 3), (5, 5)], mode="constant")

# Reshape the images to have a channel dimension and batch dimension
t1w_img = t1w_img[np.newaxis, ..., np.newaxis]
t2w_img = t2w_img[np.newaxis, ..., np.newaxis]

# Make prediction
val_pred = vxm_model.predict([t1w_img, t2w_img])

# Get predicted t2w image
fixed_t2w_img_pred = val_pred[0].squeeze()
pred_warp = val_pred[1]

# Plotting the original fixed image and the predicted fixed image
mid_slices_fixed = [np.take(t2w_img.squeeze(), vol_shape[d]//2, axis=d) for d in range(3)]
mid_slices_fixed[1] = np.rot90(mid_slices_fixed[1], 1)
mid_slices_fixed[2] = np.rot90(mid_slices_fixed[2], -1)

mid_slices_pred = [np.take(fixed_t2w_img_pred, vol_shape[d]//2, axis=d) for d in range(3)]
mid_slices_pred[1] = np.rot90(mid_slices_pred[1], 1)
mid_slices_pred[2] = np.rot90(mid_slices_pred[2], -1)
ne.plot.slices(mid_slices_fixed + mid_slices_pred, cmaps=['gray'], do_colorbars=True, grid=[2,3]);


