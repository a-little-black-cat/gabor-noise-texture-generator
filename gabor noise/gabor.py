"""
1. Implement gabor filter
    1.a: set filter parameters 
    1.b: set type of texture (grass, gravel, brick)
    1.c: set size of generated image
    1.d: set count of generated images
2. Compute features of the image using gabor filter
"""
import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage
from skimage.filters import gabor_kernel

def main(
    height,
    width,
    brick_width,
    brick_height,
    mortar_width,
    image_count,
    textureRotation,
    frequency,
    theta,
    sigma_x,
    sigma_y,
    vertical_frequency,
    vertical_theta,
    vertical_sigma_x,
    vertical_sigma_y,
    horizontal_weight,
    vertical_weight,
    mortar_frequency,
    mortar_theta,
    mortar_sigma_x,
    mortar_sigma_y,
    brick_base,
    brick_contrast,
    mortar_base,
    mortar_contrast,
    seed,
):
    if image_count < 1:
        raise ValueError("image_count must be at least 1")

    textures = []
    for image_index in range(image_count):
        texture = make_brick_texture(
            height=height,
            width=width,
            brick_width=brick_width,
            brick_height=brick_height,
            mortar_width=mortar_width,
            frequency=frequency,
            theta=np.deg2rad(theta + textureRotation),
            sigma_x=sigma_x,
            sigma_y=sigma_y,
            vertical_frequency=vertical_frequency,
            vertical_theta=np.deg2rad(vertical_theta + textureRotation),
            vertical_sigma_x=vertical_sigma_x,
            vertical_sigma_y=vertical_sigma_y,
            horizontal_weight=horizontal_weight,
            vertical_weight=vertical_weight,
            mortar_frequency=mortar_frequency,
            mortar_theta=np.deg2rad(mortar_theta + textureRotation),
            mortar_sigma_x=mortar_sigma_x,
            mortar_sigma_y=mortar_sigma_y,
            brick_base=brick_base,
            brick_contrast=brick_contrast,
            mortar_base=mortar_base,
            mortar_contrast=mortar_contrast,
            seed=seed + image_index,
        )
        textures.append(texture)

        plt.figure(figsize=(8, 8))
        plt.imshow(texture, cmap="gray")
        plt.axis("off")
        plt.tight_layout()

    plt.show()
    return textures
    


def normalize(image):
    image = image - image.min()
    return image / (image.max() + 1e-8)


def make_gabor_noise(
    height=512,
    width=512,
    frequency=0.08,
    theta=0.0,
    sigma_x=5.0,
    sigma_y=2.0,
    seed=42,
):
    rng = np.random.default_rng(seed)

    # Random source signal
    random_signal = rng.normal(0, 1, (height, width))

    # Gabor kernel contains real and imaginary components.
    kernel = gabor_kernel(
        frequency=frequency,
        theta=theta,
        sigma_x=sigma_x,
        sigma_y=sigma_y,
    )

    real_response = ndimage.convolve(
        random_signal,
        np.real(kernel),
        mode="wrap",
    )

    imaginary_response = ndimage.convolve(
        random_signal,
        np.imag(kernel),
        mode="wrap",
    )

    # Magnitude makes the noise independent of phase.
    noise = np.sqrt(real_response**2 + imaginary_response**2)

    noise -= noise.mean()
    noise /= noise.std() + 1e-8

    return noise


def make_brick_texture(
    height,
    width,
    brick_width,
    brick_height,
    mortar_width,
    frequency,
    theta,
    sigma_x,
    sigma_y,
    vertical_frequency,
    vertical_theta,
    vertical_sigma_x,
    vertical_sigma_y,
    horizontal_weight,
    vertical_weight,
    mortar_frequency,
    mortar_theta,
    mortar_sigma_x,
    mortar_sigma_y,
    brick_base,
    brick_contrast,
    mortar_base,
    mortar_contrast,
    seed,

):
    y, x = np.indices((height, width))

    row = y // brick_height
    row_offset = (row % 2) * (brick_width / 2)

    local_x = (x + row_offset) % brick_width
    local_y = y % brick_height

    # True where the pixel belongs to mortar.
    mortar = (
        (local_x < mortar_width)
        | (local_y < mortar_width)
    )

    # Gabor noise at two orientations.
    horizontal_noise = make_gabor_noise(
        height,
        width,
        frequency=frequency,
        theta=theta,
        sigma_x=sigma_x,
        sigma_y=sigma_y,
        seed=seed,
    )

    vertical_noise = make_gabor_noise(
        height,
        width,
        frequency=vertical_frequency,
        theta=vertical_theta,
        sigma_x=vertical_sigma_x,
        sigma_y=vertical_sigma_y,
        seed=seed + 1,
    )

    # Combine directional components.
    surface_noise = (
        horizontal_weight * horizontal_noise
        + vertical_weight * vertical_noise
    )

    surface_noise = normalize(surface_noise)

    # Base brick color and variation.
    brick = brick_base + brick_contrast * surface_noise

    # Darker mortar with a little variation.
    mortar_noise = make_gabor_noise(
        height,
        width,
        frequency=mortar_frequency,
        theta=mortar_theta,
        sigma_x=mortar_sigma_x,
        sigma_y=mortar_sigma_y,
        seed=seed + 2,
    )

    mortar_texture = mortar_base + mortar_contrast * normalize(mortar_noise)

    texture = np.where(mortar, mortar_texture, brick)

    return np.clip(texture, 0.0, 1.0)


if __name__ == "__main__": 

    mode = input("Enter 'input' to provide parameters or 'dataset' to use the GAN dataset: ")

    if mode == 'input':
            texture_type = input("Enter the texture type (brick, grass, gravel): ")
            if texture_type =="brick":
                height = int(input("Enter the height of the generated image: "))
                width = int(input("Enter the width of the generated image: "))
                brick_width = int(input("Enter the width of the bricks: "))
                brick_height = int(input("Enter the height of the bricks: "))
                mortar_width = int(input("Enter the width of the mortar: "))
                image_count = int(input("Enter the number of images to generate: "))
                textureRotation = int(input("Enter the rotation angle for the texture (in degrees): "))
                frequency = float(input("Enter the horizontal Gabor frequency: "))
                theta = float(input("Enter the horizontal Gabor angle (in degrees): "))
                sigma_x = float(input("Enter the horizontal Gabor sigma_x: "))
                sigma_y = float(input("Enter the horizontal Gabor sigma_y: "))
                vertical_frequency = float(input("Enter the vertical Gabor frequency: "))
                vertical_theta = float(input("Enter the vertical Gabor angle (in degrees): "))
                vertical_sigma_x = float(input("Enter the vertical Gabor sigma_x: "))
                vertical_sigma_y = float(input("Enter the vertical Gabor sigma_y: "))
                horizontal_weight = float(input("Enter the horizontal noise weight: "))
                vertical_weight = float(input("Enter the vertical noise weight: "))
                mortar_frequency = float(input("Enter the mortar Gabor frequency: "))
                mortar_theta = float(input("Enter the mortar Gabor angle (in degrees): "))
                mortar_sigma_x = float(input("Enter the mortar Gabor sigma_x: "))
                mortar_sigma_y = float(input("Enter the mortar Gabor sigma_y: "))
                brick_base = float(input("Enter the base brick intensity (0 to 1): "))
                brick_contrast = float(input("Enter the brick contrast: "))
                mortar_base = float(input("Enter the base mortar intensity (0 to 1): "))
                mortar_contrast = float(input("Enter the mortar contrast: "))
                seed = int(input("Enter the seed for random number generation: "))
            elif texture_type == "grass":
                pass
            elif texture_type == "gravel":
                pass

    elif mode == 'dataset':
        # Use the GAN dataset parameters
        texture_type = input("Enter the texture type (brick, grass, gravel): ")
        if texture_type == 'brick':
            pass
        pass
    else:
        print("Invalid mode selected. Please enter 'input' or 'dataset'.")
        exit(1)

    main(
        height=height,
        width=width,
        brick_width=brick_width,
        brick_height=brick_height,
        mortar_width=mortar_width,
        image_count=image_count,
        textureRotation=textureRotation,
        frequency=frequency,
        theta=theta,
        sigma_x=sigma_x,
        sigma_y=sigma_y,
        vertical_frequency=vertical_frequency,
        vertical_theta=vertical_theta,
        vertical_sigma_x=vertical_sigma_x,
        vertical_sigma_y=vertical_sigma_y,
        horizontal_weight=horizontal_weight,
        vertical_weight=vertical_weight,
        mortar_frequency=mortar_frequency,
        mortar_theta=mortar_theta,
        mortar_sigma_x=mortar_sigma_x,
        mortar_sigma_y=mortar_sigma_y,
        brick_base=brick_base,
        brick_contrast=brick_contrast,
        mortar_base=mortar_base,
        mortar_contrast=mortar_contrast,
        seed=seed,
    )