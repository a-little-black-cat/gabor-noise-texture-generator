from pathlib import Path
import shutil

from PIL import Image

import sys
sys.path.insert(0, str(Path(__file__).parent / "gabor noise" / "Model"))
sys.path.insert(0, str(Path(__file__).parent / "gabor noise"))

import gabor_extractor
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


trainer = PatchGAN(
    source_image=source_path,
    gabor_parameters=gabor_extractor.extract_gabor_parameters(source_path),
    patch_size=study.best_params["patch_size"],
    latent_dim=study.best_params["latent_dim"],
)

history = trainer.train(
    epochs=study.best_params["epochs"],
    batch_size=study.best_params["batch_size"],
    stride=study.best_params["stride"],
    sample_directory=output_directory / "samples",
    sample_interval=study.best_params["sample_interval"],
    sample_count=study.best_params["sample_count"],
)

trainer.save(output_directory)
create_samples_gif(
    output_directory / "samples",
    output_directory / "training_progress.gif",
    duration=500,
)
print("Final generator loss:", history["generator_loss"][-1])
print("Final discriminator loss:", history["discriminator_loss"][-1])
print("Best hyperparameters found by Optuna:", study.best_params)