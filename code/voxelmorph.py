import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils import data
from torch.autograd import Variable
import numpy as np
import os
import SimpleITK as sitk
from sklearn.model_selection import train_test_split
import nibabel as nib  # Added to work with NIfTI images
from skimage.transform import resize
import voxelmorph3d as vm3d
import time
from skimage import io
from skimage.transform import resize
from sklearn.model_selection import train_test_split
from torch.autograd import Variable
import torch.nn.functional as F

use_gpu = torch.cuda.is_available()


class VoxelMorph():
    """
    VoxelMorph Class is a higher level interface for both 2D and 3D
    Voxelmorph classes. It makes training easier and is scalable.
    """

    def __init__(self, input_dims, use_gpu=True):
        self.dims = input_dims
        self.vm = vm3d
        self.voxelmorph = vm3d.VoxelMorph3d(input_dims[0] * 2, use_gpu)
        self.optimizer = optim.SGD(
            self.voxelmorph.parameters(), lr=1e-4, momentum=0.99)
        self.params = {'batch_size': 3,
                       'shuffle': True,
                       'num_workers': 6,
                       'worker_init_fn': np.random.seed(42)
                       }
        self.device = torch.device("cuda:0" if use_gpu else "cpu")

    def check_dims(self, x):
        try:
            if x.shape[1:] == self.dims:
                return
            else:
                raise TypeError
        except TypeError as e:
            print("Invalid Dimension Error. The supposed dimension is ",
                  self.dims, "But the dimension of the input is ", x.shape[1:])

    def forward(self, x):
        self.check_dims(x)
        return self.voxelmorph(x)

    def calculate_loss(self, y, ytrue, n=9, lamda=0.01, is_training=True):
        loss = self.vm.vox_morph_loss(y, ytrue, n, lamda)
        return loss

    def train_model(self, batch_moving, batch_fixed, n=9, lamda=0.01, return_metric_score=True):
        self.optimizer.zero_grad()
        batch_fixed, batch_moving = batch_fixed.to(
            self.device), batch_moving.to(self.device)
        print("Shape of moving_image:", batch_moving.shape)
        print("Shape of fixed_image:", batch_fixed.shape)
        registered_image = self.voxelmorph(batch_moving, batch_fixed)
        train_loss = self.calculate_loss(
            registered_image, batch_fixed, n, lamda)
        train_loss.backward()
        self.optimizer.step()
        if return_metric_score:
            train_dice_score = self.vm.dice_score(
                registered_image, batch_fixed)
            return train_loss, train_dice_score
        return train_loss


    def get_test_loss(self, batch_moving, batch_fixed, n=9, lamda=0.01):
        with torch.set_grad_enabled(False):
            registered_image = self.voxelmorph(batch_moving, batch_fixed)
            val_loss = self.vm.vox_morph_loss(
                registered_image, batch_fixed, n, lamda)
            val_dice_score = self.vm.dice_score(registered_image, batch_fixed)
            return val_loss, val_dice_score

class Dataset(data.Dataset):
    'Characterizes a dataset for PyTorch'

    def __init__(self, list_IDs):
        'Initialization'
        self.list_IDs = list_IDs

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.list_IDs)

    def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        ID = self.list_IDs[index]

        # Load data and get label
        fixed_image = nib.load(os.path.join(DATA_PATH, ID,"normalized","T1w_1mm_normalized.nii.gz"))
        moving_image = nib.load(os.path.join(DATA_PATH, ID,"registered", "T2w_registered.nii.gz"))
        fixed_image = torch.Tensor(resize(fixed_image.get_fdata(), (182, 218, 182)))  # Adjust the dimensions according to your needs
        moving_image = torch.Tensor(resize(moving_image.get_fdata(), (182, 218, 182)))  # Adjust the dimensions according to your needs  # Adjust the dimensions according to your needs

        return fixed_image, moving_image

# Create list of patient IDs
patient_ids = ['{:03d}'.format(i) for i in range(1, 201)]
DATA_PATH = ('../data/output/')

# Create training and validation data loaders
params = {'batch_size': 1,
          'shuffle': True,
          'num_workers': 6,
          'worker_init_fn': np.random.seed(42)}

max_epochs = 2
partition = {}
partition['train'], partition['validation'] = train_test_split(patient_ids, test_size=0.33, random_state=42)

training_set = Dataset(partition['train'])
training_generator = data.DataLoader(training_set, **params)

validation_set = Dataset(partition['validation'])
validation_generator = data.DataLoader(validation_set, **params)

# Create VoxelMorph model
vm = VoxelMorph(input_dims=(1, 182, 218, 182), use_gpu=use_gpu)
output_dir = 'output_images'
os.makedirs(output_dir, exist_ok=True)
#Train the model
for epoch in range(max_epochs):
    start_time = time.time()
    train_loss = 0
    train_dice_score = 0
    val_loss = 0
    val_dice_score = 0
    for i, (batch_fixed, batch_moving) in enumerate(training_generator):
        loss, dice = vm.train_model(batch_moving, batch_fixed)
        val_dice_score += dice.data
        train_loss += loss.data
        
        # Save the registered image
        registered_image_np = vm.voxelmorph(batch_moving.to(vm.device), batch_fixed.to(vm.device)).cpu().detach().numpy().squeeze()
        registered_image_sitk = sitk.GetImageFromArray(registered_image_np)
        sitk.WriteImage(registered_image_sitk, os.path.join(output_dir, f'registered_image_epoch{epoch}_batch{i}.nii.gz'))
    
    print('[', "{0:.2f}".format((time.time() - start_time) / 60), 'mins]', 'After', epoch + 1, 'epochs, the Average training loss is ', train_loss *
          params['batch_size'] / len(training_set), 'and average DICE score is', val_dice_score.data * params['batch_size'] / len(validation_set))


torch.save(vm.voxelmorph.state_dict(), 'model.pth')
