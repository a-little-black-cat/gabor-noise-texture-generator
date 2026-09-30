# gabor-noise-texture-generator
Interface to create texture maps, uv maps, etc. from scratch or to replicate/create variations of example textures uploaded.
For WPI AVMI Lab use.

## Texture GAN

The TensorFlow GAN in `gabor noise/Model` learns local RGB texture patches. It
does not load MNIST or depend on the extractor's global state.

```python
import sys
import numpy as np

sys.path.append("gabor noise/Model")
from training import PatchGAN

# Use the RGB float image already produced by gabor.py, or load any RGB image.
texture = np.asarray(uploaded_image, dtype=np.float32)  # shape (H, W, 3), values [0, 1]
trainer = PatchGAN(source_image=texture, patch_size=64, latent_dim=128)
history = trainer.train(epochs=100, batch_size=32, stride=32)
generated_patches = trainer.generate(count=8).numpy()  # shape (8, 64, 64, 3), values [0, 1]
trainer.save("model_weights")
```

`extract_patches` can also be imported directly when the Gabor program should
own the training loop. The current GAN generates independent 64x64 patches;
tile or blend those patches in the existing Gabor output path when a full-size
texture is needed.
