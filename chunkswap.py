from PIL import Image, ImageOps
import math
import tqdm
import random
import numpy as np
import io
import imageio

INPATH = "lauraphoto.jpg"
OUTPATH = "me.mp4"

def get_random_coordinate(image, size):
    """
    Returns a random coordinate on the image.
    """
    x = random.randint(size // 2, image.height - size // 2 - 1)
    y = random.randint(size // 2, image.width - size // 2 - 1)
    return x, y

def get_square_coords(center_x, center_y, square_size):
    half_size = square_size // 2
    x1 = center_x - half_size
    y1 = center_y - half_size
    x2 = center_x + half_size
    y2 = center_y + half_size
    return (x1, y1, x2, y2)

def swap_pixels(img, x1, y1, x2, y2, square_size):
    """
    Swaps the squares of pixels centered at (x1, y1) and (x2, y2), respectively.
    """
    image_array = np.array(img)

    x1_start = max(x1 - square_size // 2, 0)
    y1_start = max(y1 - square_size // 2, 0)
    x2_start = max(x2 - square_size // 2, 0)
    y2_start = max(y2 - square_size // 2, 0)

    x1_end = min(x1_start + square_size, image_array.shape[0])
    y1_end = min(y1_start + square_size, image_array.shape[1])
    x2_end = min(x2_start + square_size, image_array.shape[0])
    y2_end = min(y2_start + square_size, image_array.shape[1])

    square1 = image_array[x1_start:x1_end, y1_start:y1_end].copy()
    square2 = image_array[x2_start:x2_end, y2_start:y2_end].copy()

    # Flip the squares with 50% probability
    if np.random.rand() > 0.5:
        square1 = np.flip(square1, axis=0)
        square2 = np.flip(square2, axis=0)
    if np.random.rand() > 0.5:
        square1 = np.flip(square1, axis=1)
        square2 = np.flip(square2, axis=1)

    # Swap the pixels between the squares
    image_array[x1_start:x1_end, y1_start:y1_end] = square2
    image_array[x2_start:x2_end, y2_start:y2_end] = square1

    return Image.fromarray(image_array)

def create_gif(images, duration):
    gif_data = io.BytesIO()
    print("Saving mp4 (might take a while).")
    writer = imageio.get_writer(OUTPATH, fps=1/duration)
    for im in images:
        writer.append_data(np.array(im))
    writer.close()

def simulated_annealing(image, min_square_size=32,max_square_size=256, duration=.1):
    image.convert("RGB")
    new_width  = 1024
    new_height = new_width * image.height // image.width 
    image = ImageOps.fit(image, (new_width, new_height), Image.ANTIALIAS)
    
    current_state = image.copy()
    
    # Iterate the algorithm for the specified number of iterations
    iterations = 50
    images = [image.copy()]
    print(f"Running for {iterations} steps.")
    f = 0
    for i in tqdm.tqdm(range(iterations)):
        square_size = random.randint(min_square_size, max_square_size)
        x1, y1 = get_random_coordinate(image, square_size)
        x2, y2 = get_random_coordinate(image, square_size)
        current_state = swap_pixels(current_state, x1, y1, x2, y2, square_size)

        images.append(current_state.copy())

    print(f"Skipped {f} steps.")
    create_gif(images,duration)

    return current_state

img = Image.open(INPATH)
simulated_annealing(img)
# img.show()
# img2 = 