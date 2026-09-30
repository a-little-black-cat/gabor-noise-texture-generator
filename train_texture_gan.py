from pathlib import Path
import shutil

import sys
sys.path.insert(0, str(Path(__file__).parent / "gabor noise" / "Model"))
sys.path.insert(0, str(Path(__file__).parent / "gabor noise"))

import gabor_extractor
from training import PatchGAN, create_samples_gif

source_path = Path("my_texture.png")
output_directory = Path("model_weights")
if output_directory.exists():
    shutil.rmtree(output_directory)
output_directory.mkdir(parents=True)

trainer = PatchGAN(
    source_image=source_path,
    gabor_parameters=gabor_extractor.extract_gabor_parameters(source_path),
    patch_size=64,
    latent_dim=128,
)

history = trainer.train(
    epochs=500,
    batch_size=32,
    stride=32,
    sample_directory=output_directory / "samples",
    sample_interval=10,
    sample_count=8,
)

trainer.save(output_directory)
create_samples_gif(
    output_directory / "samples",
    output_directory / "training_progress.gif",
    duration=500,
)
print("Final generator loss:", history["generator_loss"][-1])
print("Final discriminator loss:", history["discriminator_loss"][-1])