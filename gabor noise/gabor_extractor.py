"""
This class includes the gabor noise feature extraction.
To be called from gabor.py once file upload happens.
"""

from skimage.filters import gabor_kernel
from skimage import img_as_float, data
import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
import csv
from scipy import ndimage as ndi
from pylette import extract_colors

def export_gabor_kernels_csv(kernels, file_path):
    """Export kernel coefficients as one row per coefficient."""
    
    with open(file_path, "w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(("kernel", "row", "column", "coefficient"))
        for kernel_index, kernel in enumerate(kernels, start=1):
            for row, column in np.ndindex(kernel.shape):
                writer.writerow((kernel_index, row, column, float(kernel[row, column])))


def extract_gabor_parameters(image):
    # "frequency, theta, sigma_x, sigma_y, color_mean, color_std, specular_mean, specular_std, uv_map = extractor.extract_gabor_parameters(file_path)" 

    # prepare filter bank kernels using opencv gabor ekrnel
    ksize = 31 # Size of the filter returned. || I want to modify and see how this affects
    sigma = 4.0 # Standard deviation of the gaussian envelope.
    ktype = cv.CV_32F # Type of filter coefficients. It can be CV_32F or CV_64F.
    lambd = 10.0 # Wavelength of the sinusoidal factor.
    theta_range = np.arange(0, np.pi, np.pi / 4) # Orientation of the normal to the parallel stripes of a Gabor function.
    frequency_range = [0.05, 0.25] # Frequency of the sinusoidal factor.
    phase = 0 # Phase offset.

    kernels = []
    for theta in theta_range:
        for frequency in frequency_range:
            kernel = cv.getGaborKernel((ksize, ksize), sigma, theta, 1.0 / frequency, 1.0, phase, ktype=ktype)
            kernels.append(kernel)

    columns = len(frequency_range)
    rows = len(theta_range)
    figure, axes = plt.subplots(rows, columns, figsize=(8, 3 * rows), squeeze=False)
    for kernel_index, (theta, frequency) in enumerate(
        ((theta, frequency) for theta in theta_range for frequency in frequency_range)
    ):
        axis = axes.flat[kernel_index]
        axis.imshow(kernels[kernel_index], cmap="gray")
        axis.set_title(f"theta={np.rad2deg(theta):.0f}, f={frequency:g}")
        axis.axis("off")

    #sigma split into x and y components based on theta
    sigma_x = np.cos(theta) * sigma
    sigma_y = np.sin(theta) * sigma

    for kernel in kernels:
            yield kernel

    color_mean = get_color_mean(image)
    color_std = get_color_std(image)
    specular_mean = get_specular_mean(image)
    specular_std = get_specular_std(image)
    uv_map = get_uv_map(image)

    figure.suptitle("Gabor Filter Kernels")
    figure.tight_layout()
    figure.show()

    yield frequency, theta, sigma_x, sigma_y, color_mean, color_std, specular_mean, specular_std, uv_map

def get_color_mean(image):
    # actually what if i extract a color palette

    #color_mean = np.mean(image, axis=(0, 1))  # Mean color across the image

    palette = extract_colors(image, palette_size=5)  # Extract a palette of 5 colors

    for color in palette:
        print(f"Color: {color},  HSV: {color.hsv}")  # Print each color in RGB and HSV

    palette.to_json(filename="color_palette.json", colorspace="hsv")  # Save the palette to a JSON file

    # return color_mean
    return None

def get_specular_mean(image):

    #return specular_mean
    return None

def get_color_std(image):
    # return color_std
    return None

def get_specular_std(image):
    #return specular_std
    return None

def get_uv_map(image):
    #return uv_map
    return None