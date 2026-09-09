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
from PIL import Image, ImageTk
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
        "variation_size": 512,
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
        "frequency": 0.055,
        "theta": 0.0,
        "sigma_x": 7.0,
        "sigma_y": 1.2,
    }

    gravel_defaults = {
        "frequency": 0.08,
        "theta": 0.0,
        "sigma_x": 5.0,
        "sigma_y": 2.0,
    }

    rock_defaults = { #LIKELY to be converted into woronoi tessellation or something similar -- might be combined with proc. rock generator in future
        "frequency": 0.08,  
        "theta": 0.0,
        "sigma_x": 5.0,
        "sigma_y": 2.0,
    }


    variables = {}
    uploaded_parameters = None
    uploaded_image = None
    variation_window = None

    def add_section(title):
        section = ttk.LabelFrame(form_frame, text=title, padding=8)
        section.pack(fill="x", pady=(0, 10))
        return section

    def add_field(section, name, label, default):
        row = ttk.Frame(section)
        row.pack(fill="x", pady=2)
        ttk.Label(row, text=label).pack(side="left", anchor="w")
        variable = Tk.StringVar(value=str(default))
        variables.setdefault(name, []).append(variable)
        ttk.Entry(row, textvariable=variable, width=14).pack(side="right")

    def upload_texture():
        nonlocal uploaded_parameters, uploaded_image
        file_path = filedialog.askopenfilename(
            title="Select Texture Image",
            filetypes=[("Image Files", "*.png *.jpg *.jpeg *.bmp *.tiff")],
        )
        if file_path:
            print(f"Selected texture image: {file_path}")
            try:
                extracted = extractor.extract_gabor_parameters(file_path)
                uploaded_parameters = extracted
                uploaded_image = np.asarray(
                    Image.open(file_path).convert("RGB"), dtype=np.float32
                ) / 255.0
                for name in ("frequency", "sigma_x", "sigma_y"):
                    for variable in variables[name]:
                        variable.set(f"{extracted[name]:.6g}")
                for variable in variables["theta"]:
                    variable.set(f"{np.rad2deg(extracted['theta']):.3f}")
                status.set(
                    f"Extracted {len(extracted['components'])} noise components "
                    f"from {extracted['image_size']['width']}x{extracted['image_size']['height']} image"
                )
                print(extracted)
            except (OSError, ValueError) as error:
                messagebox.showerror("Texture extraction failed", str(error), parent=root)
                status.set("Texture extraction failed")

    def show_variations(textures):
        nonlocal variation_window
        if variation_window is not None and variation_window.winfo_exists():
            variation_window.destroy()

        variation_window = Tk.Toplevel(root)
        variation_window.title("Uploaded Texture Variations")
        variation_window.geometry("980x760")
        variation_window.columnconfigure(0, weight=1)
        variation_window.rowconfigure(0, weight=1)

        canvas = Tk.Canvas(variation_window, highlightthickness=0)
        scrollbar = ttk.Scrollbar(variation_window, orient="vertical", command=canvas.yview)
        grid_frame = ttk.Frame(canvas, padding=12)
        canvas_window = canvas.create_window((0, 0), window=grid_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        canvas.grid(row=0, column=0, sticky="nsew")
        scrollbar.grid(row=0, column=1, sticky="ns")

        def update_scroll_region(_event=None):
            canvas.configure(scrollregion=canvas.bbox("all"))

        def resize_grid(event):
            canvas.itemconfigure(canvas_window, width=event.width)

        grid_frame.bind("<Configure>", update_scroll_region)
        canvas.bind("<Configure>", resize_grid)
        photos = []
        for index, texture in enumerate(textures):
            image = Image.fromarray(np.uint8(np.clip(texture, 0.0, 1.0) * 255.0))
            image.thumbnail((220, 220), Image.Resampling.LANCZOS)
            photo = ImageTk.PhotoImage(image)
            photos.append(photo)
            card = ttk.Frame(grid_frame, padding=6)
            card.grid(row=index // 4, column=index % 4, padx=6, pady=6, sticky="nsew")
            ttk.Label(card, image=photo).pack()
            ttk.Label(card, text=f"Variation {index + 1}").pack(pady=(4, 0))
        variation_window._photos = photos

    def generate_uploaded_variations():
        try:
            if uploaded_parameters is None or uploaded_image is None:
                raise ValueError("Upload a texture before generating variations")
            size = read_value("variation_size", int)
            image_count = read_value("image_count", int)
            seed = read_value("seed", int)
            rotation = np.deg2rad(read_value("texture_rotation", float))
            if size < 1 or image_count < 1:
                raise ValueError("variation size and image count must be positive")

            components = uploaded_parameters["components"]
            energies = np.asarray([component["energy"] for component in components])
            weights = energies / (energies.sum() + 1e-8)
            textures = []
            for image_index in range(image_count):
                rng = np.random.default_rng(seed + image_index)
                source_height, source_width = uploaded_image.shape[:2]
                scale = max(size / source_height, size / source_width)
                scale *= rng.uniform(1.0, 1.2)
                resized = Image.fromarray(np.uint8(uploaded_image * 255.0)).resize(
                    (max(size, int(source_width * scale)), max(size, int(source_height * scale))),
                    Image.Resampling.BICUBIC,
                )
                max_x = resized.width - size
                max_y = resized.height - size
                crop_x = int(rng.integers(0, max_x + 1))
                crop_y = int(rng.integers(0, max_y + 1))
                crop = resized.crop((
                    crop_x,
                    crop_y,
                    crop_x + size,
                    crop_y + size,
                ))
                if rng.random() < 0.5:
                    crop = crop.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
                if rng.random() < 0.5:
                    crop = crop.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
                crop = crop.rotate(np.rad2deg(rotation), resample=Image.Resampling.BICUBIC)
                texture = np.asarray(crop, dtype=np.float32) / 255.0

                combined = np.zeros((size, size), dtype=np.float32)
                for component_index, (component, weight) in enumerate(zip(components, weights)):
                    combined += weight * make_gabor_noise(
                        size, size, component["frequency"],
                        component["theta"] + rotation,
                        component["sigma_x"], component["sigma_y"],
                        seed + image_index * len(components) + component_index,
                    )
                perturbation = normalize(combined)[..., None] - 0.5
                texture = np.clip(texture + perturbation * 0.12, 0.0, 1.0)

                # Vary the RGB distribution while preserving the crop's spatial detail.
                base_mean = np.asarray(uploaded_parameters["color_mean"], dtype=np.float32)
                base_std = np.asarray(uploaded_parameters["color_std"], dtype=np.float32)
                target_mean = np.clip(base_mean + rng.normal(0.0, 0.025, 3), 0.0, 1.0)
                target_std = np.maximum(base_std * rng.uniform(0.9, 1.1, 3), 0.005)
                texture_uint8 = extractor.create_color_variation(
                    np.uint8(texture * 255.0), target_mean, target_std
                )
                texture = texture_uint8.astype(np.float32) / 255.0
                textures.append(texture)
            generated_textures[:] = textures
            show_variations(textures)
            status.set(f"Generated {image_count} uploaded-texture variation(s) at {size}x{size}")
        except (TypeError, ValueError) as error:
            messagebox.showerror("Variation generation failed", str(error), parent=root)
            status.set("Variation generation failed")


    general_section = add_section("General")
    texture_type = Tk.StringVar(value="brick")
    ttk.Label(general_section, text="Texture type").pack(side="left", anchor="w")
    ttk.Combobox(general_section, textvariable=texture_type, values=("brick", "grass", "gravel"), state="readonly", width=11).pack(side="right")
    for name, default in general_defaults.items():
        add_field(general_section, name, name.replace("_", " ").title(), default)

    upload_texture_section = add_section("Upload Texture")
    ttk.Label(upload_texture_section, text="Upload texture image").pack(side="left", anchor="w")
    ttk.Button(upload_texture_section, text="Upload", command=upload_texture).pack(side="right")
    ttk.Button(
        upload_texture_section,
        text="Generate Uploaded Variations",
        command=generate_uploaded_variations,
    ).pack(side="right", padx=(0, 8))

    brick_section = add_section("Brick Parameters")
    for name, default in brick_defaults.items():
        add_field(brick_section, name, name.replace("_", " ").title(), default)

    grass_section = add_section("GrassParameters")
    for name, default in grass_defaults.items():
        add_field(grass_section, name, name.replace("_", " ").title(), default)

    gravel_section = add_section("GravelParameters")
    for name, default in gravel_defaults.items():
        add_field(gravel_section, name, name.replace("_", " ").title(), default)

    rock_section = add_section("Rock Parameters")
    for name, default in rock_defaults.items():
        add_field(rock_section, name, name.replace("_", " ").title(), default)

    status = Tk.StringVar(value="Ready")
    generated_textures = []
    results_figure = None

    def read_value(name, converter):
        return converter(variables[name][0].get().strip())

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
                    value = texture[row, column]
                    if np.ndim(value) > 0:
                        value = np.mean(value)
                    writer.writerow((image_index, row, column, float(value)))
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

                elif selected_type == "gravel":
                
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
                elif selected_type == "rock":
                    parameters = {name: read_value(name, float) for name in rock_defaults}
                    texture = make_rock_texture(
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

#gotta differentiate between calculations of grass and gravel here onwards womp
def make_grass_texture(
    height,
    width,
    frequency,
    theta,
    sigma_x,
    sigma_y,
    seed,
):
    rng = np.random.default_rng(seed)
    density = np.clip(frequency * 1.8, 0.035, 0.18)
    impulse_field = (rng.random((height, width)) < density).astype(np.float32)

    # Blur sparse impulses into elongated strands in the requested direction.
    aligned = ndimage.rotate(
        impulse_field,
        angle=-np.rad2deg(theta),
        reshape=False,
        order=1,
        mode="reflect",
    )
    strands = ndimage.gaussian_filter(
        aligned,
        sigma=(max(sigma_y, 0.5), max(sigma_x, 0.5)),
        mode="wrap",
    )
    strands = ndimage.rotate(
        strands,
        angle=np.rad2deg(theta),
        reshape=False,
        order=1,
        mode="reflect",
    )
    strands = normalize(strands)

    broad_variation = make_gabor_noise(
        height=height,
        width=width,
        frequency=max(frequency * 0.45, 0.015),
        theta=theta,
        sigma_x=max(sigma_x * 1.5, 2.0),
        sigma_y=max(sigma_y, 1.0),
        seed=seed + 1,
    )
    broad_variation = normalize(broad_variation)
    grass_texture = 0.12 + 0.68 * strands + 0.20 * broad_variation
    return np.clip(grass_texture, 0.0, 1.0)

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

def make_rock_texture(
    height,
    width,
    frequency,
    theta,
    sigma_x,
    sigma_y,
    seed,
):
    rock_noise = make_gabor_noise(
        height=height,
        width=width,
        frequency=frequency,
        theta=theta,
        sigma_x=sigma_x,
        sigma_y=sigma_y,
        seed=seed,
    )

    rock_texture = normalize(rock_noise)

    return rock_texture

if __name__ == "__main__":
    userInputMode()

    