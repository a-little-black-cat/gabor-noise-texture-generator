# Running the Texture GAN

This project contains a TensorFlow DCGAN-style model for learning local RGB
texture patches from a Gabor texture or any other RGB image. It does not use
MNIST.

## 1. Open the repository

Open a PowerShell terminal at the repository root.

## 2. Activate the included environment

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
.\noiseComp\Scripts\Activate.ps1
```

Verify TensorFlow:

```powershell
python -c "import tensorflow as tf; print(tf.__version__)"
```

The included environment currently reports TensorFlow `2.21.0`.

## 3. Run the existing Gabor program

From the repository root:

```powershell
python ".\gabor noise\gabor.py"
```

Use **Upload Texture** to select an RGB image. The Gabor program extracts
parameters and stores the uploaded image as an RGB NumPy array with shape
`(height, width, 3)` and values in `[0, 1]`.

## 4. Train the GAN from an image

The simplest first run is a separate Python script. Create a file named
`train_texture_gan.py` in the repository root with this content:

```python
from pathlib import Path

import numpy as np
from PIL import Image

import sys
sys.path.insert(0, str(Path(__file__).parent / "gabor noise" / "Model"))

from training import PatchGAN

source_path = Path("my_texture.png")
output_directory = Path("model_weights")

trainer = PatchGAN(
    source_image=source_path,
    patch_size=64,
    latent_dim=128,
)

history = trainer.train(
    epochs=100,
    batch_size=32,
    stride=32,
    sample_directory=output_directory / "samples",
    sample_interval=10,
    sample_count=8,
)

trainer.save(output_directory)
print("Final generator loss:", history["generator_loss"][-1])
print("Final discriminator loss:", history["discriminator_loss"][-1])
```

Place `my_texture.png` in the repository root, then run:

```powershell
python .\train_texture_gan.py
```

Generated patches will be written into subfolders such as
`model_weights/samples/epoch_0010/` and `model_weights/samples/epoch_0020/`.
The final generator and discriminator weights will be written directly to
`model_weights`.
After training, `model_weights/training_progress.gif` is created with one
frame per saved epoch. Each frame is a grid of that epoch's generated samples.
The training script clears the previous `model_weights` directory at startup,
so every run contains only newly generated samples and weights.
Training on CPU can take a while. Start with `epochs=1` or `epochs=5` to
verify the pipeline before using a longer run.

The training options controlling periodic samples are:

```python
sample_directory=output_directory / "samples"
sample_interval=10  # use 25 for less frequent saves
sample_count=8
```

The same fixed random seed is used for each checkpoint, so `sample_000.png`
at different epochs shows how the generator changes over time.

## 5. Connect it to the Gabor program

The Gabor UI already creates the right input array in `gabor.py`. After an
image has been uploaded, pass that array to the trainer:

```python
from training import PatchGAN

trainer = PatchGAN(source_image=uploaded_image)
history = trainer.train(epochs=100, batch_size=32, stride=32)
generated_patches = trainer.generate(count=8).numpy()
```

`uploaded_image` must be RGB with shape `(height, width, 3)`. Values may be
floats in `[0, 1]` or image bytes in `[0, 255]`; the trainer normalizes either
format.

If the Gabor extractor is called directly, this is also valid:

```python
import numpy as np
from PIL import Image

from training import PatchGAN

image = np.asarray(Image.open("my_texture.png").convert("RGB"), dtype=np.float32)
trainer = PatchGAN(source_image=image)
trainer.train(epochs=100)
```

## 6. Important model behavior

The GAN learns overlapping `64x64` patches, not a complete image-sized canvas.
This is intentional for a single-source texture workflow. `generate()` returns
independent patches with shape `(count, 64, 64, 3)` and values in `[0, 1]`.

To create a full-size texture, the next integration step is to tile or blend
the generated patches into the desired output dimensions. The existing Gabor
synthesis functions can remain responsible for full-size output dimensions.

## 7. Useful API

```python
from training import PatchGAN, extract_patches, load_rgb_image
```

- `load_rgb_image(image)`: loads a path or validates an RGB NumPy array.
- `extract_patches(image, patch_size=64, stride=32)`: returns training patches
  normalized to `[-1, 1]`.
- `PatchGAN.train(...)`: trains the generator and discriminator.
- `PatchGAN.save_samples(directory, count=8, seed=42)`: saves generated RGB
    patches to a directory.
- `create_samples_gif(sample_directory, output_path)`: creates a GIF from the
    saved epoch sample folders.
- `PatchGAN.generate(count=1)`: returns generated RGB patches in `[0, 1]`.
- `PatchGAN.save(directory)`: saves generator and discriminator weights.

## Troubleshooting

### `ModuleNotFoundError: No module named 'tensorflow'`

Activate the included environment and run the command again:

```powershell
.\noiseComp\Scripts\Activate.ps1
```

### `image must have shape (height, width, 3)`

The input is grayscale or has an alpha channel. Convert it to RGB first:

```python
from PIL import Image
image = Image.open("input.png").convert("RGB")
```

### `image must be at least as large as patch_size`

The source image must be at least `64x64` for the default configuration.

### Training appears slow

Reduce the test run to `epochs=1`, use a larger `stride`, or reduce the image
size before training. A single source image generally requires more epochs for
visibly useful variation.
