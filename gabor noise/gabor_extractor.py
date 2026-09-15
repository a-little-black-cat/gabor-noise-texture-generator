"""Feature extraction for creating variations of uploaded texture images."""

import csv

import cv2 as cv
import numpy as np
from scipy import ndimage as ndi
from skimage.filters import gabor_kernel
from pylette import extract_colors

gray_img = None


def export_gabor_kernels_csv(kernels, file_path):
    """Export kernel coefficients as one row per coefficient."""
    with open(file_path, "w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(("kernel", "row", "column", "coefficient"))
        for kernel_index, kernel in enumerate(kernels, start=1):
            for row, column in np.ndindex(kernel.shape):
                writer.writerow((kernel_index, row, column, float(kernel[row, column])))



def extract_gabor_parameters(image, max_components=8):
    """Extract JSON-friendly noise and appearance parameters from an image."""
    global gray_img

    source = cv.imread(image, cv.IMREAD_COLOR) if isinstance(image, str) else image
    if source is None or source.ndim != 3:
        raise ValueError("Could not read a color texture image")

    image_float = source.astype(np.float32) / 255.0
    gray_img = cv.cvtColor(image_float, cv.COLOR_BGR2GRAY)
    gray_img -= gray_img.mean()
    gray_img /= gray_img.std() + 1e-8

    frequencies = np.geomspace(0.02, 0.45, 12)
    orientations = np.arange(8, dtype=np.float32) * np.pi / 8
    sigmas = ((1.0, 1.0), (2.0, 1.0), (3.0, 2.0), (5.0, 2.0), (7.0, 3.0))
    candidates = []
    for frequency in frequencies:
        for theta in orientations:
            for sigma_x, sigma_y in sigmas:
                kernel = gabor_kernel(
                    frequency, theta=theta, sigma_x=sigma_x, sigma_y=sigma_y
                )
                real = ndi.convolve(gray_img, np.real(kernel), mode="wrap")
                imaginary = ndi.convolve(gray_img, np.imag(kernel), mode="wrap")
                energy = float(np.mean(real * real + imaginary * imaginary))
                candidates.append({
                    "frequency": float(frequency),
                    "theta": float(theta),
                    "sigma_x": float(sigma_x),
                    "sigma_y": float(sigma_y),
                    "energy": energy,
                    "response_std": float(np.sqrt(energy)),
                })

    candidates.sort(key=lambda item: item["energy"], reverse=True)
    components = []
    for candidate in candidates:
        if all(
            abs(candidate["frequency"] - selected["frequency"]) > 0.015
            or abs(candidate["theta"] - selected["theta"]) > np.pi / 16
            for selected in components
        ):
            components.append(candidate)
        if len(components) == max_components:
            break

    rgb = cv.cvtColor(image_float, cv.COLOR_BGR2RGB)
    luminance = cv.cvtColor(image_float, cv.COLOR_BGR2GRAY)
    highlights = np.clip(luminance - 0.8, 0.0, 0.2) / 0.2
    primary = components[0]
    palette_image = np.clip(rgb * 255.0, 0.0, 255.0).astype(np.uint8)
    palette = extract_palette_from_image(palette_image, max_num_colors=8)
    power_spectrum = get_power_spectrum()
    



    return {
        "frequency": primary["frequency"],
        "theta": primary["theta"],
        "sigma_x": primary["sigma_x"],
        "sigma_y": primary["sigma_y"],
        "noise_amplitude": primary["response_std"],
        "components": components,
        "color_mean": np.mean(rgb, axis=(0, 1)).tolist(),
        "color_std": np.std(rgb, axis=(0, 1)).tolist(),
        "luminance_mean": float(np.mean(luminance)),
        "luminance_std": float(np.std(luminance)),
        "specular_mean": float(np.mean(highlights)),
        "specular_std": float(np.std(highlights)),
        "image_size": {"height": int(source.shape[0]), "width": int(source.shape[1])},
        "uv_map": None,
        "palette": [
            {
                "rgb": color.rgb,
                "hex": color.hex,
                "hsv": color.hsv,
                "frequency": float(color.frequency),
            }
            for color in palette.colors
        ],
        "power_spectrum": power_spectrum,

    }


def compute_feats(image, kernels):
    """Compute mean and variance responses for a set of Gabor kernels."""
    feats = np.zeros((len(kernels), 2), dtype=np.double)
    for index, kernel in enumerate(kernels):
        filtered = ndi.convolve(image, kernel, mode="wrap")
        feats[index] = filtered.mean(), filtered.var()
    return feats

def extract_palette_from_image(image, max_num_colors=8):
    palette = extract_colors(image, max_num_colors)
    for color in palette.colors:
        print(f"RGB: {color.rgb}")
        print(f"Hex: {color.hex}")
        print(f"HSV: {color.hsv}")
        print(f"Frequency: {color.frequency:.2%}")
    palette.to_json(filename="palette.json", colorspace='hsv')
    return palette

def create_color_variation(image, color_mean=None, color_std=None, palette=None):
    """Create a color variation using extracted palette colors when available."""
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("Input image must be a color image with 3 channels.")

    # Convert to float32 for processing
    image_float = image.astype(np.float32) / 255.0

    if palette:
        palette_rgb = np.asarray(
            [color["rgb"] for color in palette], dtype=np.float32
        ) / 255.0
        palette_luminance = (
            0.2126 * palette_rgb[:, 0]
            + 0.7152 * palette_rgb[:, 1]
            + 0.0722 * palette_rgb[:, 2]
        )
        order = np.argsort(palette_luminance)
        palette_rgb = palette_rgb[order]
        palette_luminance = palette_luminance[order]

        luminance = (
            0.2126 * image_float[:, :, 0]
            + 0.7152 * image_float[:, :, 1]
            + 0.0722 * image_float[:, :, 2]
        )
        luminance_min, luminance_max = luminance.min(), luminance.max()
        if luminance_max - luminance_min < 1e-8:
            normalized_luminance = np.full_like(luminance, 0.5)
        else:
            normalized_luminance = (luminance - luminance_min) / (
                luminance_max - luminance_min
            )
        palette_positions = np.linspace(0.0, 1.0, len(palette_rgb))
        color_indices = np.abs(
            normalized_luminance[..., None] - palette_positions
        ).argmin(axis=2)
        new_image = palette_rgb[color_indices]
        return np.uint8(np.clip(new_image, 0.0, 1.0) * 255.0)

    if color_mean is None or color_std is None:
        raise ValueError("Provide either a palette or color mean and standard deviation.")
    
    # Calculate current mean and std deviation
    current_mean = np.mean(image_float, axis=(0, 1))
    current_std = np.std(image_float, axis=(0, 1))
    
    # Normalize the image
    normalized_image = (image_float - current_mean) / (current_std + 1e-8)
    
    # Apply new mean and std deviation
    new_image = normalized_image * color_std + color_mean
    
    # Clip values to [0, 1] range and convert back to uint8
    new_image_clipped = np.clip(new_image, 0.0, 1.0)
    return (new_image_clipped * 255).astype(np.uint8)


def normalize(array):
    """Normalize an array to the range [0, 1]."""
    min_val = np.min(array)
    max_val = np.max(array)
    if max_val - min_val < 1e-8:
        return np.zeros_like(array)
    return (array - min_val) / (max_val - min_val)

def get_power_spectrum(image=None):
    """Compute the power spectrum of a grayscale image."""
    if image is None:
        if gray_img is None:
            raise ValueError("No grayscale image is available")
        gray = gray_img
    elif image.ndim == 3 and image.shape[2] == 3:
        gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    else:
        gray = image
    f_transform = np.fft.fft2(gray)
    f_shifted = np.fft.fftshift(f_transform)
    power_spectrum = np.abs(f_shifted) ** 2 #Array of power spectrum values
    power_spectrum = normalize(power_spectrum)
    return power_spectrum
