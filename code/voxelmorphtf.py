import os
import subprocess

voxelmorph_script = "voxelmorph/scripts/tf/voxelmorph3d.py"
gpu_id = "0"

# Specify the image list files and the output directories for the models
training_data = [
    {"img_list": "t1_t2_image_list.txt", "model_dir": "models/output_t1_t2"},
    #{"img_list": "t1_fa_image_list.txt", "model_dir": "models/output_t1_fa"},
    #{"img_list": "t1_adc_image_list.txt", "model_dir": "models/output_t1_adc"},
]

for data in training_data:
    img_list = data["img_list"]
    model_dir = data["model_dir"]

    # Ensure the model directory exists
    os.makedirs(model_dir, exist_ok=True)

    # Train the VoxelMorph model
    train_command = [
        "python",
        voxelmorph_script,
        "--img-list", img_list,
        "--model-dir", model_dir,
        "--gpu", gpu_id,
        "--epochs", "6000",
        "--steps-per-epoch", "100",
        "--batch-size", "1",
        "--lr", "1e-4",
    ]
    subprocess.run(train_command)
