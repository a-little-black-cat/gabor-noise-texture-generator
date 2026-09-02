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
import texturize as txt
from scipy import ndimage as ndi
from scipy.optimize import curve_fit

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

    #get image from filepath
    image = cv.imread(image)

    image_float = image.astype(np.float32) / 255.0 # Convert image to float32 and normalize to [0, 1]

    gray_image = make_image_grayscale(image_float) # for easier analysis, convert to grayscale if not already
    power_spectrum = get_power_spectrum(gray_image) # visualize the power spectrum of the image

    #get orientation and frequency from power spectrum
    orientation, frequency, amplitude, offset = extract_sinusoidal_parameters(power_spectrum) # Placeholder for extracting sinusoidal parameters

    # prepare filter bank kernels using opencv gabor ekrnel
    ksize = 31 # Size of the filter returned. || I want to modify and see how this affects
    sigma = 4.0 # Standard deviation of the gaussian envelope.
    ktype = cv.CV_32F # Type of filter coefficients. It can be CV_32F or CV_64F.
    lambd = 10.0 # Wavelength of the sinusoidal factor.
    theta_range = np.arange(0, np.pi, np.pi / 4) # Orientation of the normal to the parallel stripes of a Gabor function.
    frequency_range = [0.05, 0.25] # Frequency of the sinusoidal factor.
    phase = 0 # Phase offset.
    kernels = [] # List to hold the generated kernels

    #


    







    """
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
    """

    color_mean = get_color_mean(image)
    color_std = get_color_std(image)
    specular_mean = get_specular_mean(image)
    specular_std = get_specular_std(image)
    uv_map = get_uv_map(image)

    figure.suptitle("Gabor Filter Kernels")
    figure.tight_layout()
    figure.show()

    yield frequency, theta, sigma_x, sigma_y, color_mean, color_std, specular_mean, specular_std, uv_map

def compute_feats(image, kernels):
    # Compute the features for an image using a set of Gabor kernels.
    feats = np.zeros((len(kernels), 2), dtype=np.double)
    # Compute the features for each kernel
    for k, kernel in enumerate(kernels):
        filtered = ndi.convolve(image, kernel, mode='wrap') 
        feats[k, 0] = filtered.mean() # Mean of the filtered image
        feats[k, 1] = filtered.var() # Variance of the filtered image
    return feats

def make_image_grayscale(image):
    # Convert the image to grayscale if it is not already
    if len(image.shape) == 3 and image.shape[2] == 3:
        return cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    return image


def get_power_spectrum(image):
    #power spectrum is the squared magnitude of the Fourier transform of the image
    f_transform = np.fft.fft2(image)
    f_shift = np.fft.fftshift(f_transform) # Shift the zero frequency component to the center of the spectrum
    power_spectrum = np.abs(f_shift) ** 2

    return power_spectrum

def exract_gaussian_parameters(image):
    # Placeholder for extracting Gaussian parameters from the image.
    # This function can be implemented to analyze the image and extract relevant Gaussian parameters.
    return None

def extract_sinusoidal_parameters(image):
    #extract sinusoidal parameters from the image using curve fitting
    x_data = np.arange(image.shape[1]) # x_data is the column indices of the image
    y_data = np.arange(image.shape[0]) # y_data is the row indices of the image

    # Fit a sinusoidal curve to the data
    params = fit_sinusoidal_curve(x_data, y_data)

    # Extract the parameters
    amplitude, frequency, phase, offset = params

    # Calculate the orientation (in radians)
    orientation = phase


    return orientation, frequency, amplitude, offset

def sinusoidal_function(x, amplitude, frequency, phase, offset):
    func = amplitude * np.sin(2 * np.pi * frequency * x + phase) + offset
    return func

def fit_sinusoidal_curve(x_data, y_data):
    # Fit a sinusoidal curve to the given data using curve fitting.
    initial_guess = [1, 0.1, 0, 0]  # Initial guess for amplitude, frequency, phase, offset

    params, _ = curve_fit(sinusoidal_function, x_data, y_data, p0=initial_guess)
    return params  # Returns the fitted parameters: amplitude, frequency, phase, offset


def get_color_mean(image):
    # actually... what if i extract a color palette

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