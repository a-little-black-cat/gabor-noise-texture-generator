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
import tkinter as Tk
import csv
from tkinter import messagebox, ttk, filedialog
import gabor_extractor as extractor


def prompt_value(prompt, default, converter):
    value = input(f"{prompt} [{default}]: ").strip()
    return default if not value else converter(value)
    # This is to minimize the number of input prompts for the user. If the user presses enter without typing anything, the default value will be used. Otherwise, the input will be converted to the specified type (int or float) using the provided converter function.


def consoleInputMode():
    texture_type = input("Enter the texture type (brick, grass, gravel) [brick]: ").strip().lower() or "brick"
    if texture_type not in {"brick", "grass", "gravel"}:
        raise ValueError("texture type must be brick, grass, or gravel")

    height = prompt_value("Enter the height of the generated image", 512, int)
    width = prompt_value("Enter the width of the generated image", 512, int)
    image_count = prompt_value("Enter the number of images to generate", 1, int)
    texture_rotation = prompt_value("Enter the rotation angle for the texture (in degrees)", 0.0, float)
    seed = prompt_value("Enter the seed for random number generation", 42, int)

    if image_count < 1:
        raise ValueError("image_count must be at least 1")

    texture_parameters = {}
    if texture_type == "brick":
        texture_parameters = {
            "brick_width": prompt_value("Enter the width of the bricks", 64, int),
            "brick_height": prompt_value("Enter the height of the bricks", 32, int),
            "mortar_width": prompt_value("Enter the width of the mortar", 2, int),
            "mortar_height": prompt_value("Enter the height of the mortar", 2, int),
            "frequency": prompt_value("Enter the horizontal Gabor frequency", 0.08, float),
            "theta": np.deg2rad(prompt_value("Enter the horizontal Gabor angle (in degrees)", 0.0, float)),
            "sigma_x": prompt_value("Enter the horizontal Gabor sigma_x", 5.0, float),
            "sigma_y": prompt_value("Enter the horizontal Gabor sigma_y", 2.0, float),
            "vertical_frequency": prompt_value("Enter the vertical Gabor frequency", 0.08, float),
            "vertical_theta": np.deg2rad(prompt_value("Enter the vertical Gabor angle (in degrees)", 90.0, float)),
            "vertical_sigma_x": prompt_value("Enter the vertical Gabor sigma_x", 5.0, float),
            "vertical_sigma_y": prompt_value("Enter the vertical Gabor sigma_y", 2.0, float),
            "horizontal_weight": prompt_value("Enter the horizontal noise weight", 0.7, float),
            "vertical_weight": prompt_value("Enter the vertical noise weight", 0.3, float),
            "mortar_frequency": prompt_value("Enter the mortar Gabor frequency", 0.12, float),
            "mortar_theta": np.deg2rad(prompt_value("Enter the mortar Gabor angle (in degrees)", 0.0, float)),
            "mortar_sigma_x": prompt_value("Enter the mortar Gabor sigma_x", 2.0, float),
            "mortar_sigma_y": prompt_value("Enter the mortar Gabor sigma_y", 1.0, float),
            "brick_base": prompt_value("Enter the base brick intensity (0 to 1)", 0.55, float),
            "brick_contrast": prompt_value("Enter the brick contrast", 0.2, float),
            "mortar_base": prompt_value("Enter the base mortar intensity (0 to 1)", 0.25, float),
            "mortar_contrast": prompt_value("Enter the mortar contrast", 0.05, float),
        }
    else:
        texture_parameters = {
            "frequency": prompt_value("Enter the Gabor frequency", 0.08, float),
            "theta": prompt_value("Enter the Gabor angle (in degrees)", 0.0, float),
            "sigma_x": prompt_value("Enter the Gabor sigma_x", 5.0, float),
            "sigma_y": prompt_value("Enter the Gabor sigma_y", 2.0, float),
        }

    textures = []
    for image_index in range(image_count):
        image_seed = seed + image_index

        if texture_type == "brick":
            texture = make_brick_texture(
                height=height,
                width=width,
                **texture_parameters,
                texture_rotation=texture_rotation,
                seed=image_seed,
            )
        else:
            noise_function = make_grass_texture if texture_type == "grass" else make_gravel_texture
            texture = noise_function(
                height=height,
                width=width,
                frequency=texture_parameters["frequency"],
                theta=np.deg2rad(texture_parameters["theta"]) + np.deg2rad(texture_rotation),
                sigma_x=texture_parameters["sigma_x"],
                sigma_y=texture_parameters["sigma_y"],
                seed=image_seed,
            )

        textures.append(texture)

        plt.figure(figsize=(8, 8))
        plt.imshow(texture, cmap="gray")
        plt.title(f"{texture_type} {image_index + 1}")
        plt.axis("off")
        plt.tight_layout()

    plt.show()
    return textures


def userInputMode():
    root = Tk.Tk()
    root.title("Gabor Texture Generator")
    root.geometry("620x760")

    outer_frame = ttk.Frame(root, padding=12)
    outer_frame.pack(fill="both", expand=True)

    canvas = Tk.Canvas(outer_frame, highlightthickness=0)
    scrollbar = ttk.Scrollbar(outer_frame, orient="vertical", command=canvas.yview)
    form_frame = ttk.Frame(canvas)
    canvas_window = canvas.create_window((0, 0), window=form_frame, anchor="nw")
    canvas.configure(yscrollcommand=scrollbar.set)
    canvas.pack(side="left", fill="both", expand=True)
    scrollbar.pack(side="right", fill="y")

    def update_scroll_region(_event=None):
        canvas.configure(scrollregion=canvas.bbox("all"))

    def resize_form(event):
        canvas.itemconfigure(canvas_window, width=event.width)

    form_frame.bind("<Configure>", update_scroll_region)
    canvas.bind("<Configure>", resize_form)

    ttk.Label(form_frame, text="Gabor Texture Generator", font=("TkDefaultFont", 16, "bold")).pack(anchor="w", pady=(0, 12))

    general_defaults = {
        "height": 512,
        "width": 512,
        "image_count": 1,
        "texture_rotation": 0.0,
        "seed": 42,
    }
    brick_defaults = {
        "brick_width": 64,
        "brick_height": 32,
        "mortar_width": 2,
        "mortar_height": 2,
        "frequency": 0.08,
        "theta": 0.0,
        "sigma_x": 5.0,
        "sigma_y": 2.0,
        "vertical_frequency": 0.08,
        "vertical_theta": 90.0,
        "vertical_sigma_x": 5.0,
        "vertical_sigma_y": 2.0,
        "horizontal_weight": 0.7,
        "vertical_weight": 0.3,
        "mortar_frequency": 0.12,
        "mortar_theta": 0.0,
        "mortar_sigma_x": 2.0,
        "mortar_sigma_y": 1.0,
        "brick_base": 0.55,
        "brick_contrast": 0.2,
        "mortar_base": 0.25,
        "mortar_contrast": 0.05,
    }
    grass_defaults = {
        "frequency": 0.08,
        "theta": 0.0,
        "sigma_x": 5.0,
        "sigma_y": 2.0,
    }

    gravel_defaults = {
        "frequency": 0.08,
        "theta": 0.0,
        "sigma_x": 5.0,
        "sigma_y": 2.0,
    }
    variables = {}

    def add_section(title):
        section = ttk.LabelFrame(form_frame, text=title, padding=8)
        section.pack(fill="x", pady=(0, 10))
        return section

    def add_field(section, name, label, default):
        row = ttk.Frame(section)
        row.pack(fill="x", pady=2)
        ttk.Label(row, text=label).pack(side="left", anchor="w")
        variable = Tk.StringVar(value=str(default))
        variables[name] = variable
        ttk.Entry(row, textvariable=variable, width=14).pack(side="right")

    def upload_texture():
        file_path = filedialog.askopenfilename(
            title="Select Texture Image",
            filetypes=[("Image Files", "*.png *.jpg *.jpeg *.bmp *.tiff")],
        )
        if file_path:
            print(f"Selected texture image: {file_path}")

    

        """
        Values extracted from uploaded image:
        frequency: {frequency}
        theta: {theta}
        sigma_x: {sigma_x}
        sigma_y: {sigma_y}
        color_mean: {color_mean}
        color_std: {color_std}
        specular_mean: {specular_mean}
        specular_std: {specular_std}
        uv_map: {uv_map} -> this can be used to create textures that match mesh shapes through uv mapping. It can be used to create textures that match the mesh shapes through uv mapping.
        """

        frequency, theta, sigma_x, sigma_y, color_mean, color_std, specular_mean, specular_std, uv_map = extractor.extract_gabor_parameters(file_path)


    general_section = add_section("General")
    texture_type = Tk.StringVar(value="brick")
    ttk.Label(general_section, text="Texture type").pack(side="left", anchor="w")
    ttk.Combobox(general_section, textvariable=texture_type, values=("brick", "grass", "gravel"), state="readonly", width=11).pack(side="right")
    for name, default in general_defaults.items():
        add_field(general_section, name, name.replace("_", " ").title(), default)

    upload_texture_section = add_section("Upload Texture")
    ttk.Label(upload_texture_section, text="Upload texture image").pack(side="left", anchor="w")
    ttk.Button(upload_texture_section, text="Upload", command=upload_texture).pack(side="right")

    brick_section = add_section("Brick Parameters")
    for name, default in brick_defaults.items():
        add_field(brick_section, name, name.replace("_", " ").title(), default)

    grass_section = add_section("GrassParameters")
    for name, default in grass_defaults.items():
        add_field(grass_section, name, name.replace("_", " ").title(), default)

    gravel_section = add_section("GravelParameters")
    for name, default in gravel_defaults.items():
        add_field(gravel_section, name, name.replace("_", " ").title(), default)

    status = Tk.StringVar(value="Ready")
    generated_textures = []
    results_figure = None

    def read_value(name, converter):
        return converter(variables[name].get().strip())

    def export_textures():
        if not generated_textures:
            messagebox.showinfo("No results", "Generate at least one texture first.", parent=root)
            return

        file_path = filedialog.asksaveasfilename(
            title="Export texture data",
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv")],
        )
        if not file_path:
            return

        with open(file_path, "w", newline="", encoding="utf-8") as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(("image", "row", "column", "intensity"))
            for image_index, texture in enumerate(generated_textures, start=1):
                for row, column in np.ndindex(texture.shape):
                    writer.writerow((image_index, row, column, float(texture[row, column])))
        status.set(f"Exported {len(generated_textures)} texture(s) to CSV")

    def generate_from_form():
        nonlocal results_figure
        try:
            height = read_value("height", int)
            width = read_value("width", int)
            image_count = read_value("image_count", int)
            texture_rotation = read_value("texture_rotation", float)
            seed = read_value("seed", int)
            if height < 1 or width < 1 or image_count < 1:
                raise ValueError("height, width, and image count must be positive")

            selected_type = texture_type.get()
            textures = []
            for image_index in range(image_count):
                image_seed = seed + image_index
                if selected_type == "brick":
                    parameters = {name: read_value(name, float if isinstance(default, float) else int) for name, default in brick_defaults.items()}
                    parameters["theta"] = np.deg2rad(parameters["theta"])
                    parameters["vertical_theta"] = np.deg2rad(parameters["vertical_theta"])
                    parameters["mortar_theta"] = np.deg2rad(parameters["mortar_theta"])
                    texture = make_brick_texture(height=height, width=width, texture_rotation=texture_rotation, seed=image_seed, **parameters)

                elif selected_type == "grass":
                    parameters = {name: read_value(name, float) for name in grass_defaults}
                    texture = make_grass_texture(
                        height=height,
                        width=width,
                        frequency=parameters["frequency"],
                        theta=np.deg2rad(parameters["theta"] + texture_rotation),
                        sigma_x=parameters["sigma_x"],
                        sigma_y=parameters["sigma_y"],
                        seed=image_seed,
                    )
                else:
                    parameters = {name: read_value(name, float) for name in gravel_defaults}
                    texture = make_gravel_texture(
                        height=height,
                        width=width,
                        frequency=parameters["frequency"],
                        theta=np.deg2rad(parameters["theta"] + texture_rotation),
                        sigma_x=parameters["sigma_x"],
                        sigma_y=parameters["sigma_y"],
                        seed=image_seed,
                    )
                textures.append(texture)
            generated_textures[:] = textures
            if results_figure is not None:
                plt.close(results_figure)
            columns = min(4, max(1, image_count))
            rows = int(np.ceil(image_count / columns))
            results_figure, axes = plt.subplots(rows, columns, squeeze=False, figsize=(4 * columns, 4 * rows))
            for image_index, texture in enumerate(textures):
                axis = axes.flat[image_index]
                axis.imshow(texture, cmap="gray", vmin=0, vmax=1)
                axis.set_title(f"{selected_type} {image_index + 1}")
                axis.axis("off")
            for axis in axes.flat[image_count:]:
                axis.axis("off")
            results_figure.tight_layout()
            results_figure.show()
            status.set(f"Generated {image_count} {selected_type} texture(s)")
        except (TypeError, ValueError) as error:
            messagebox.showerror("Invalid input", str(error), parent=root)
            status.set("Invalid input")

    ttk.Button(form_frame, text="Generate Textures", command=generate_from_form).pack(anchor="e", pady=(0, 8))
    ttk.Button(form_frame, text="Export Results as CSV", command=export_textures).pack(anchor="e", pady=(0, 8))
    ttk.Label(form_frame, textvariable=status).pack(anchor="w")
    root.mainloop()
    


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
    mortar_height,
    texture_rotation,
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
    texture_angle = np.deg2rad(texture_rotation)

    # True where the pixel belongs to mortar.
    mortar = (
        (local_x < mortar_width)
        | (local_y < mortar_height)
    )

    # Gabor noise at two orientations.
    horizontal_noise = make_gabor_noise(
        height,
        width,
        frequency=frequency,
        theta=theta + texture_angle,
        sigma_x=sigma_x,
        sigma_y=sigma_y,
        seed=seed,
    )

    vertical_noise = make_gabor_noise(
        height,
        width,
        frequency=vertical_frequency,
        theta=vertical_theta + texture_angle,
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
        theta=mortar_theta + texture_angle,
        sigma_x=mortar_sigma_x,
        sigma_y=mortar_sigma_y,
        seed=seed + 2,
    )

    mortar_texture = mortar_base + mortar_contrast * normalize(mortar_noise)

    texture = np.where(mortar, mortar_texture, brick)

    return np.clip(texture, 0.0, 1.0)

def make_grass_texture(
    height,
    width,
    frequency,
    theta,
    sigma_x,
    sigma_y,
    seed,
):
    grass_noise = make_gabor_noise(
        height=height,
        width=width,
        frequency=frequency,
        theta=theta,
        sigma_x=sigma_x,
        sigma_y=sigma_y,
        seed=seed,
    )

    grass_texture = normalize(grass_noise)

    return grass_texture

def make_gravel_texture(
    height,
    width,
    frequency,
    theta,
    sigma_x,
    sigma_y,
    seed,
):
    gravel_noise = make_gabor_noise(
        height=height,
        width=width,
        frequency=frequency,
        theta=theta,
        sigma_x=sigma_x,
        sigma_y=sigma_y,
        seed=seed,
    )

    gravel_texture = normalize(gravel_noise)

    return gravel_texture

if __name__ == "__main__":
    userInputMode()

    