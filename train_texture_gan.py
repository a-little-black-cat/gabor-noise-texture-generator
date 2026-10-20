from pathlib import Path
import shutil

from PIL import Image

import sys
sys.path.insert(0, str(Path(__file__).parent / "gabor noise" / "Model"))
sys.path.insert(0, str(Path(__file__).parent / "gabor noise"))
import cv2 as cv
import gabor_extractor
import numpy as np
from training import PatchGAN, create_samples_gif

import optuna # to optimize hyperparam

source_path = Path("my_texture.png")
if not source_path.is_file():
    raise FileNotFoundError(f"Source image not found: {source_path.resolve()}")

with Image.open(source_path) as source_image:
    source_height, source_width = source_image.height, source_image.width

output_directory = Path("model_weights")
if output_directory.exists():
    shutil.rmtree(output_directory)
output_directory.mkdir(parents=True)



#hyperparam range for optuna to experiment with

hyperaparameters = {
    "patch_size": [64, 128, 256, 512],
    "latent_dim": [64, 128, 256, 512],
    "epochs": [100, 200, 300, 400, 500, 1000, 1500, 2000],
    "stride": [16, 32, 64,128],
    "batch_size": [16, 32, 64, 128],
    "learning_rate": [0.0001, 0.001, 0.01, 0.1],
    "sample_interval": [5, 10, 20, 50],
    "sample_count": [4, 8, 16, 32],
}

hyperaparameters["patch_size"] = [
    patch_size
    for patch_size in hyperaparameters["patch_size"]
    if patch_size <= min(source_height, source_width)
]
if not hyperaparameters["patch_size"]:
    raise ValueError(
        f"Source image ({source_width}x{source_height}) is smaller than every configured patch size"
    )

study = optuna.create_study(direction="minimize")
study.optimize(
    lambda trial: PatchGAN(
        source_image=source_path,
        gabor_parameters=gabor_extractor.extract_gabor_parameters(source_path),
        patch_size=trial.suggest_categorical("patch_size", hyperaparameters["patch_size"]),
        latent_dim=trial.suggest_categorical("latent_dim", hyperaparameters["latent_dim"]),
    ).train(
        epochs=trial.suggest_categorical("epochs", hyperaparameters["epochs"]),
        batch_size=trial.suggest_categorical("batch_size", hyperaparameters["batch_size"]),
        stride=trial.suggest_categorical("stride", hyperaparameters["stride"]),
        sample_directory=output_directory / "samples",
        sample_interval=trial.suggest_categorical("sample_interval", hyperaparameters["sample_interval"]),
        sample_count=trial.suggest_categorical("sample_count", hyperaparameters["sample_count"]),
    )["generator_loss"][-1],
    n_trials=10,
)
best_hyperparameters = study.best_params




trainer = PatchGAN(
    source_image=source_path,
    gabor_parameters=gabor_extractor.extract_gabor_parameters(source_path),
    patch_size=best_hyperparameters["patch_size"],
    latent_dim=best_hyperparameters["latent_dim"],
)

history = trainer.train(
    epochs=best_hyperparameters["epochs"],
    batch_size=best_hyperparameters["batch_size"],
    stride=best_hyperparameters["stride"],
    sample_directory=output_directory / "samples",
    sample_interval=best_hyperparameters["sample_interval"],
    sample_count=best_hyperparameters["sample_count"],
)

trainer.save(output_directory)
create_samples_gif(
    output_directory / "samples",
    output_directory / "training_progress.gif",
    duration=500,
)


generated_image = trainer.generate(count=1, seed=42).numpy()[0]
generated_image = np.clip(generated_image * 255.0, 0, 255).astype(np.uint8)


final_gabor_parameters = (
    gabor_extractor.extract_gabor_parameters(generated_image)
)

grayscale_image = gabor_extractor.generate_gabor_noise(
    source_height,
    source_width,
    final_gabor_parameters,
    seed=42,
)

#getting frequency values from the generated image

power_spectrum = gabor_extractor.get_power_spectrum(grayscale_image)
dominant_frequencies = gabor_extractor.estimate_dominant_frequencies(power_spectrum, num_frequencies=5)
print("Dominant frequencies:")
for item in dominant_frequencies:
    frequency = item["frequency"]
    print(f"  {frequency}: {item}")
    print(
        f"  {frequency} (in cycles per pixel): "
        f"{frequency / min(grayscale_image.shape)}"
    )

dominant_values = [
    item["frequency"]
    for item in dominant_frequencies
    if item["frequency"] > 0.0
]


frequency_min = max(0.01, min(dominant_values) * 0.7)

frequency_max = min(0.45, max(dominant_values) * 1.3)
frequencies = np.geomspace(
    frequency_min,
    frequency_max,
    12,
)


generated_rgb = gabor_extractor.colorize_gabor_noise(grayscale_image, final_gabor_parameters)

"""
gabor_parameters       = parameters extracted from the source texture
final_hyperparameters  = best GAN settings from Optuna
final_gabor_parameters = parameters extracted from the trained GAN output
"""

grayscale_image = gabor_extractor.normalize(grayscale_image)
output = np.uint8(grayscale_image * 255)

cv.imwrite(str(output_directory / "grayscale_gabor_noise.png"), output)
cv.imwrite(str(output_directory / "colorized_gabor_noise.png"), generated_rgb)

print("Final generator loss:", history["generator_loss"][-1])
print("Final discriminator loss:", history["discriminator_loss"][-1])
print("Final Gabor parameters of generated image:", final_gabor_parameters)
print("Best hyperparameters found by Optuna:", best_hyperparameters)
