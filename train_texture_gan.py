import argparse
import json
import os
import shutil
import sys
from pathlib import Path

from PIL import Image

os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

sys.path.insert(0, str(Path(__file__).parent / "gabor noise" / "Model"))
sys.path.insert(0, str(Path(__file__).parent / "gabor noise"))
import cv2 as cv
import gabor_extractor
import numpy as np
import optuna
import tensorflow as tf
from optuna.trial import TrialState
from training import PatchGAN, create_samples_gif

tf.config.optimizer.set_experimental_options({"remapping": False})


def parse_args():
    parser = argparse.ArgumentParser(description="Train the conditional texture GAN")
    parser.add_argument(
        "--from-save-state",
        action="store_true",
        help="Resume Optuna and final GAN training from model_weights",
    )
    parser.add_argument(
        "--gen-output-only",
        action="store_true",
        help="Generate outputs from saved model weights without training",
    )
    parser.add_argument(
        "--clear",
        action="store_true",
        help="Explicitly request the default fresh training behavior",
    )
    parser.add_argument(
        "--require-gpu",
        action="store_true",
        help="Stop instead of silently training on CPU when no GPU is detected",
    )
    return parser.parse_args()


def write_state(path, state):
    path.write_text(json.dumps(state, indent=2), encoding="utf-8")


def configure_device(require_gpu=False):
    """Report TensorFlow devices and optionally prevent an accidental CPU run."""
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("TensorFlow GPU(s):", ", ".join(str(gpu) for gpu in gpus))
    else:
        print("TensorFlow GPU(s): none; using CPU")
        if require_gpu:
            raise RuntimeError(
                "No TensorFlow GPU was detected. Run this project in WSL2 with "
                'a CUDA-enabled install (`pip install "tensorflow[and-cuda]"`), '
                "or omit --require-gpu to allow CPU training."
            )


def generate_outputs(trainer, source_height, source_width, output_directory):
    generated_image = trainer.generate(count=1, seed=42).numpy()[0]
    generated_image = np.clip(generated_image * 255.0, 0, 255).astype(np.uint8)
    final_gabor_parameters = gabor_extractor.extract_gabor_parameters(generated_image)
    grayscale_image = gabor_extractor.generate_gabor_noise(
        source_height,
        source_width,
        final_gabor_parameters,
        seed=42,
    )

    power_spectrum = gabor_extractor.get_power_spectrum(grayscale_image)
    dominant_frequencies = gabor_extractor.estimate_dominant_frequencies(
        power_spectrum, num_frequencies=5
    )
    print("Dominant frequencies:")
    for item in dominant_frequencies:
        frequency = item["frequency"]
        print(f"  {frequency}: {item}")
        print(
            f"  {frequency} (in cycles per pixel): "
            f"{frequency / min(grayscale_image.shape)}"
        )

    generated_rgb = gabor_extractor.colorize_gabor_noise(
        grayscale_image, final_gabor_parameters
    )
    cv.imwrite(
        str(output_directory / "grayscale_gabor_noise.png"),
        np.uint8(grayscale_image * 255),
    )
    cv.imwrite(str(output_directory / "colorized_gabor_noise.png"), generated_rgb)
    return final_gabor_parameters


def main():
    args = parse_args()
    configure_device(require_gpu=args.require_gpu)
    if args.gen_output_only and (args.from_save_state or args.clear):
        raise ValueError(
            "--gen-output-only cannot be combined with --from-save-state or --clear"
        )
    if args.from_save_state and args.clear:
        raise ValueError("--from-save-state cannot be combined with --clear")
    source_path = Path("my_texture.png")
    if not source_path.is_file():
        raise FileNotFoundError(f"Source image not found: {source_path.resolve()}")

    print("dLog: Using source image:", source_path.resolve())
    with Image.open(source_path) as source_image:
        source_height, source_width = source_image.height, source_image.width

    output_directory = Path("model_weights")
    state_path = output_directory / "training_state.json"
    if args.gen_output_only:
        if not state_path.is_file():
            raise FileNotFoundError(
                "Cannot generate outputs: model_weights/training_state.json was not found"
            )
        state = json.loads(state_path.read_text(encoding="utf-8"))
        best_hyperparameters = state.get("best_hyperparameters")
        if not best_hyperparameters:
            raise ValueError(
                "Cannot generate outputs: saved best_hyperparameters are missing"
            )
        source_parameters = gabor_extractor.extract_gabor_parameters(source_path)
        trainer = PatchGAN(
            source_image=source_path,
            gabor_parameters=source_parameters,
            patch_size=best_hyperparameters["patch_size"],
            latent_dim=best_hyperparameters["latent_dim"],
            learning_rate=best_hyperparameters["learning_rate"],
        )
        generator_weights = output_directory / "generator.weights.h5"
        if generator_weights.is_file():
            trainer.gan.generator.load_weights(generator_weights)
        else:
            checkpoint_path = tf.train.latest_checkpoint(
                str(output_directory / "checkpoints")
            )
            if checkpoint_path is None:
                raise FileNotFoundError(
                    "Cannot generate outputs: saved generator weights or checkpoints were not found"
                )
            checkpoint = tf.train.Checkpoint(
                generator=trainer.gan.generator,
                discriminator=trainer.gan.discriminator,
                generator_optimizer=trainer.generator_optimizer,
                discriminator_optimizer=trainer.discriminator_optimizer,
            )
            checkpoint.restore(checkpoint_path).expect_partial()
        final_gabor_parameters = generate_outputs(
            trainer,
            source_height,
            source_width,
            output_directory,
        )
        print("Final Gabor parameters of generated image:", final_gabor_parameters)
        print("Generated outputs using saved model state.")
        return

    if args.from_save_state:
        print("Resuming training from saved state...")
        if not output_directory.is_dir() or not state_path.is_file():
            raise FileNotFoundError(
                "Cannot resume: model_weights/training_state.json was not found"
            )
        state = json.loads(state_path.read_text(encoding="utf-8"))
    else:
        # A normal invocation starts a fresh run, matching the original script.
        # --clear is kept as a backwards-compatible explicit spelling of this
        # behavior.
        if output_directory.exists():
            shutil.rmtree(output_directory)
        output_directory.mkdir(parents=True, exist_ok=True)
        state = {"phase": "optuna"}

    source_parameters = gabor_extractor.extract_gabor_parameters(source_path)
    hyperparameters = {
        "patch_size": [64, 128, 256, 512],
        "latent_dim": [64, 128, 256, 512],
        "epochs": [100, 200, 300, 400, 500, 1000],
        "stride": [16, 32, 64, 128],
        "batch_size": [16, 32, 64, 128],
        "learning_rate": [1e-5, 5e-5, 1e-4, 2e-4, 5e-4, 1e-3],
        "sample_interval": [5, 10, 20, 50],
        "sample_count": [4, 8, 16, 32],
    }
    hyperparameters["patch_size"] = [
        patch_size
        for patch_size in hyperparameters["patch_size"]
        if patch_size <= min(source_height, source_width)
    ]
    if not hyperparameters["patch_size"]:
        raise ValueError(
            f"Source image ({source_width}x{source_height}) is smaller than every configured patch size"
        )

    study_path = output_directory / "optuna.db"
    study = optuna.create_study(
        study_name="texture_gan",
        direction="minimize",
        storage=f"sqlite:///{study_path.resolve().as_posix()}",
        load_if_exists=True,
    )
    state.update({"phase": "optuna"})
    write_state(state_path, state)

    def objective(trial):
        trainer = PatchGAN(
            source_image=source_path,
            gabor_parameters=source_parameters,
            patch_size=trial.suggest_categorical(
                "patch_size", hyperparameters["patch_size"]
            ),
            latent_dim=trial.suggest_categorical(
                "latent_dim", hyperparameters["latent_dim"]
            ),
            learning_rate=trial.suggest_categorical(
                "learning_rate", hyperparameters["learning_rate"]
            ),
        )
        return trainer.train(
            epochs=trial.suggest_categorical("epochs", hyperparameters["epochs"]),
            batch_size=trial.suggest_categorical(
                "batch_size", hyperparameters["batch_size"]
            ),
            stride=trial.suggest_categorical("stride", hyperparameters["stride"]),
            sample_directory=output_directory / "optuna_samples",
            sample_interval=trial.suggest_categorical(
                "sample_interval", hyperparameters["sample_interval"]
            ),
            sample_count=trial.suggest_categorical(
                "sample_count", hyperparameters["sample_count"]
            ),
        )["generator_loss"][-1]

    completed_trials = len(study.get_trials(states=[TrialState.COMPLETE]))
    if completed_trials < 10:
        study.optimize(objective, n_trials=10 - completed_trials)

    best_hyperparameters = study.best_params
    state.update(
        {
            "phase": "final_training",
            "best_hyperparameters": best_hyperparameters,
            "source_size": [source_height, source_width],
        }
    )
    write_state(state_path, state)

    trainer = PatchGAN(
        source_image=source_path,
        gabor_parameters=source_parameters,
        patch_size=best_hyperparameters["patch_size"],
        latent_dim=best_hyperparameters["latent_dim"],
        learning_rate=best_hyperparameters["learning_rate"],
    )
    history = trainer.train(
        epochs=best_hyperparameters["epochs"],
        batch_size=best_hyperparameters["batch_size"],
        stride=best_hyperparameters["stride"],
        sample_directory=output_directory / "samples",
        sample_interval=best_hyperparameters["sample_interval"],
        sample_count=best_hyperparameters["sample_count"],
        checkpoint_directory=output_directory / "checkpoints",
        resume=args.from_save_state,
    )
    state["phase"] = "completed"
    write_state(state_path, state)

    trainer.save(output_directory)
    create_samples_gif(
        output_directory / "samples",
        output_directory / "training_progress.gif",
        duration=500,
    )

    final_gabor_parameters = generate_outputs(
        trainer,
        source_height,
        source_width,
        output_directory,
    )

    print("Final generator loss:", history["generator_loss"][-1])
    print("Final discriminator loss:", history["discriminator_loss"][-1])
    print("Final Gabor parameters of generated image:", final_gabor_parameters)
    print("Best hyperparameters found by Optuna:", best_hyperparameters)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print(
            "\nTraining interrupted. Completed epochs are saved in "
            "model_weights/checkpoints. Resume with --from-save-state."
        )
        raise
