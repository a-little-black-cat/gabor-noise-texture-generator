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

# Avoid TensorFlow's NHWC/NCHW layout rewrite failing on GPU dropout graphs.
tf.config.optimizer.set_experimental_options(
    {"remapping": False, "layout_optimizer": False}
)


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
    parser.add_argument(
        "--gabor-density-scale",
        type=float,
        default=0.1,
        help="Multiplier for Gabor impulse density (default: 0.1)",
    )
    parser.add_argument(
        "--gabor-envelope-scale",
        type=float,
        default=1.0,
        help="Multiplier for Gabor envelope size (default: 1.0)",
    )
    parser.add_argument(
        "--gaussian-blur-sigma",
        type=float,
        default=1.0,
        help="Gaussian noise blur sigma in pixels (default: 1.0)",
    )
    return parser.parse_args()


def write_state(path, state):
    path.write_text(json.dumps(state, indent=2), encoding="utf-8")


def batch_sizes_for_patch(patch_size):
    """Keep activation memory bounded as patch area increases."""
    return {
        64: [16, 32, 64, 128],
        128: [8, 16, 32, 64],
        256: [2, 4, 8, 16],
        512: [1, 2, 4],
    }[patch_size]


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


def generate_outputs(
    trainer,
    source_height,
    source_width,
    output_directory,
    gabor_density_scale=0.1,
    gabor_envelope_scale=1.0,
    gaussian_blur_sigma=1.0,
):
    generated_image = trainer.generate(count=1, seed=42).numpy()[0]
    generated_image = np.clip(generated_image * 255.0, 0, 255).astype(np.uint8)
    final_gabor_parameters = gabor_extractor.extract_gabor_parameters(generated_image)
    grayscale_image = gabor_extractor.generate_gabor_noise(
        source_height,
        source_width,
        final_gabor_parameters,
        seed=42,
        density_scale=gabor_density_scale,
        sigma_scale=gabor_envelope_scale,
    )
    gabor_mean = float(np.mean(grayscale_image))
    gabor_std = float(np.std(grayscale_image))
    gaussian_image = gabor_extractor.generate_gaussian_noise(
        source_height,
        source_width,
        seed=42,
        blur_sigma=gaussian_blur_sigma,
        target_mean=gabor_mean,
        target_std=gabor_std,
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
    gaussian_rgb = gabor_extractor.colorize_gabor_noise(
        gaussian_image, final_gabor_parameters
    )
    cv.imwrite(
        str(output_directory / "grayscale_gabor_noise.png"),
        np.uint8(gabor_extractor.normalize(grayscale_image) * 255),
    )
    cv.imwrite(str(output_directory / "colorized_gabor_noise.png"), generated_rgb)
    cv.imwrite(
        str(output_directory / "grayscale_gaussian_noise.png"),
        np.uint8(gabor_extractor.normalize(gaussian_image) * 255),
    )
    cv.imwrite(
        str(output_directory / "colorized_gaussian_noise.png"),
        gaussian_rgb,
    )
    noise_settings = {
        "gabor": {
            "density_scale": gabor_density_scale,
            "envelope_scale": gabor_envelope_scale,
            "seed": 42,
            "source": "sparse Poisson impulses convolved with extracted Gabor components",
            "mean": gabor_mean,
            "std": gabor_std,
        },
        "gaussian": {
            "blur_sigma": gaussian_blur_sigma,
            "seed": 42,
            "source": "Gaussian white noise smoothed with a wrapped Gaussian filter",
            "matched_to_gabor_mean": gabor_mean,
            "matched_to_gabor_std": gabor_std,
        },
    }
    (output_directory / "noise_settings.json").write_text(
        json.dumps(noise_settings, indent=2),
        encoding="utf-8",
    )
    print("Noise outputs:")
    print(f"  Gabor: density_scale={gabor_density_scale}, envelope_scale={gabor_envelope_scale}")
    print(f"    sparse Poisson impulses + extracted Gabor envelopes; seed=42")
    print(f"  Gaussian: blur_sigma={gaussian_blur_sigma}; seed=42")
    print(f"    smoothed Gaussian noise matched to Gabor mean={gabor_mean:.6f}, std={gabor_std:.6f}")
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
                "Cannot generate outputs: training has not completed and saved "
                "best_hyperparameters are missing. Resume with "
                "--from-save-state --require-gpu first."
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
            args.gabor_density_scale,
            args.gabor_envelope_scale,
            args.gaussian_blur_sigma,
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
        study_name="texture_gan_gpu",
        direction="minimize",
        storage=f"sqlite:///{study_path.resolve().as_posix()}",
        load_if_exists=True,
    )
    state.update({"phase": "optuna"})
    write_state(state_path, state)

    def objective(trial):
        patch_size = trial.suggest_categorical(
            "patch_size", hyperparameters["patch_size"]
        )
        latent_dim = trial.suggest_categorical(
            "latent_dim", hyperparameters["latent_dim"]
        )
        batch_size = trial.suggest_categorical(
            f"batch_size_{patch_size}", batch_sizes_for_patch(patch_size)
        )
        epochs = trial.suggest_categorical("epochs", hyperparameters["epochs"])
        print(
            f"Starting Optuna trial {trial.number}: "
            f"patch_size={patch_size}, latent_dim={latent_dim}, "
            f"batch_size={batch_size}, epochs={epochs}",
            flush=True,
        )
        trainer = PatchGAN(
            source_image=source_path,
            gabor_parameters=source_parameters,
            patch_size=patch_size,
            latent_dim=latent_dim,
            learning_rate=trial.suggest_categorical(
                "learning_rate", hyperparameters["learning_rate"]
            ),
        )
        try:
            return trainer.train(
                epochs=epochs,
                batch_size=batch_size,
                stride=trial.suggest_categorical("stride", hyperparameters["stride"]),
                sample_directory=output_directory / "optuna_samples",
                sample_interval=trial.suggest_categorical(
                    "sample_interval", hyperparameters["sample_interval"]
                ),
                sample_count=trial.suggest_categorical(
                    "sample_count", hyperparameters["sample_count"]
                ),
            )["generator_loss"][-1]
        except tf.errors.OpError as error:
            if "out of memory" not in str(error).lower():
                raise
            raise optuna.TrialPruned("GPU ran out of memory for this trial") from error
        finally:
            del trainer
            tf.keras.backend.clear_session()

    completed_trials = len(study.get_trials(states=[TrialState.COMPLETE]))
    if completed_trials < 10:
        study.optimize(objective, n_trials=10 - completed_trials)

    best_hyperparameters = dict(study.best_params)
    patch_size = best_hyperparameters["patch_size"]
    best_hyperparameters["batch_size"] = best_hyperparameters.pop(
        f"batch_size_{patch_size}"
    )
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
        args.gabor_density_scale,
        args.gabor_envelope_scale,
        args.gaussian_blur_sigma,
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
