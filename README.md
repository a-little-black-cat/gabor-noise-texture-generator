# OffRoad Sim Gabor-Based Noise Analysis and Generation

*To install all requirements, run*
`pip install -r requirements.txt` from root directory

*To run:*
`python .\train_texture_gan.py`from root directory

Flags:
`python .\train_texture_gan.py -f`
where -f is

`--from-save-state` : enabling to continue training from saved state if it gets interrupted

or

`--gen-output-only` : to generate outputs from saved model

Running without flags starts a fresh training run and replaces the existing
`model_weights` directory, as in the original workflow. `--clear` is also
accepted as an explicit request for the same fresh-run behavior.

## NVIDIA GPU training with WSL2

This project uses TensorFlow. Current TensorFlow releases do not use CUDA GPUs
from native Windows, so NVIDIA GPU training should run in WSL2 with Ubuntu.
The repository's requirements install the CUDA-enabled TensorFlow extra on
Linux and the regular Windows package on native Windows.

Install WSL2 from an elevated PowerShell prompt on Windows:

```powershell
wsl --install -d Ubuntu
```

After restarting and completing Ubuntu's first-time setup, run these commands
inside Ubuntu:

```bash
cd /mnt/c/Users/beren/OneDrive/Desktop/gabor-v-texton
sudo apt update
sudo apt install -y python3-venv python3-pip
mkdir -p ~/venvs ~/tmp
python3 -m venv ~/venvs/gabor-texton
source ~/venvs/gabor-texton/bin/activate
python -m pip install --upgrade pip
TMPDIR=~/tmp pip install --no-cache-dir -r requirements.txt
```

Keep the virtual environment in your WSL home directory rather than under
`/mnt/c`; this avoids filling the Windows-mounted temporary filesystem during
the large CUDA dependency install.

Verify that TensorFlow sees the GPU:

```bash
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```

Run training and require GPU execution:

```bash
python train_texture_gan.py --require-gpu
```

Each completed run writes both Gabor and Gaussian-correlated noise outputs:

- `model_weights/grayscale_gabor_noise.png`
- `model_weights/colorized_gabor_noise.png`
- `model_weights/grayscale_gaussian_noise.png`
- `model_weights/colorized_gaussian_noise.png`
- `model_weights/noise_settings.json`

The Gaussian output uses smoothed white Gaussian noise whose mean and standard
deviation are matched to the generated Gabor noise. Adjust the generation
parameters without changing training:

Run this in the **Ubuntu/WSL terminal**, not PowerShell:

```bash
python train_texture_gan.py --gen-output-only --require-gpu --gabor-density-scale 0.1 --gabor-envelope-scale 1.0 --gaussian-blur-sigma 1.0
```

Increase `--gabor-density-scale` for more impulses, increase
`--gabor-envelope-scale` for larger Gabor envelopes, and increase
`--gaussian-blur-sigma` for smoother Gaussian noise.

The Optuna search uses smaller batch-size ranges for larger patches because
GPU memory grows with the patch area. If a trial still exceeds available VRAM,
it is pruned and the search continues instead of stopping the whole run.
Check current GPU usage from another Ubuntu terminal with:

```bash
nvidia-smi
```

If the verification command prints an empty list, check `nvidia-smi` inside
Ubuntu and confirm that the Windows NVIDIA driver is current. Do not install a
separate Linux NVIDIA driver inside WSL2.

Example run for just generating:
`python train_texture_gan.py --gen-output-only --gabor-density-scale 0.1 --gabor-envelope-scale 4.0 --gaussian-blur-sigma 6.0`