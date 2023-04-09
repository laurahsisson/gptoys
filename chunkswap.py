from PIL import Image, ImageOps, ImageDraw
import math
import tqdm
import random
import numpy as np
import io
import base64
import IPython.display as display
import imageio

def calculate_distance(image, x, y):
    """
    Calculates the distance between the color value of a pixel and the average color value of that pixel's neighbors.
    """
    pixel = image.getpixel((x, y))
    neighbors = [(i, j) for i in range(x-1, x+2) for j in range(y-1, y+2) if (i != x or j != y) and i >= 0 and i < image.width and j >= 0 and j < image.height]
    avg_color = tuple(sum(image.getpixel((i, j))) // 3 for i, j in neighbors)
    return sum((pixel[i] - avg_color[i])**2 for i in range(3))


def calculate_score(image, x, y, size):
    """
    Calculates the sum of distances between each pixel in a square centered on (x, y) of the specified size and its neighbors.
    """
    half_size = size // 2
    x_start = max(0, x - half_size)
    x_end = min(image.width, x + half_size + 1)
    y_start = max(0, y - half_size)
    y_end = min(image.height, y + half_size + 1)
    return sum(calculate_distance(image, i, j) for i in range(x_start, x_end) for j in range(y_start, y_end))


def can_swap_pixels(img, x1, y1, x2, y2, square_size):
    """
    Calculates the difference in score between swapping the squares of pixels
    centered at (x1, y1) and (x2, y2), respectively.
    """
    score1 = calculate_score(img, x1,y1, square_size)
    score2 = calculate_score(img, x2,y2, square_size)

    # Swap squares of pixels
    img1 = swap_pixels(img, x1, y1, x2, y2, square_size)

    new_score1 = calculate_score(img1,  x1, y1, square_size)
    new_score2 = calculate_score(img1,  x2, y2, square_size)

    # Calculate difference in score
    delta_score = (new_score1 + new_score2) - (score1 + score2)

    return delta_score


def get_random_coordinate(image, size):
    """
    Returns a random coordinate on the image.
    """
    x = random.randint(size // 2, image.width - size // 2 - 1)
    y = random.randint(size // 2, image.height - size // 2 - 1)
    return x, y

def get_square_coords(center_x, center_y, size):
    half_size = size // 2
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

# get the starting and ending coordinates of the squares to be swapped
    x1_start, y1_start, x1_end, y1_end = get_square_coords(x1, y1, square_size)
    x2_start, y2_start, x2_end, y2_end = get_square_coords(x2, y2, square_size)
    
    # swap the squares of pixels surrounding each center
    temp = image_array[x1_start:x1_end, y1_start:y1_end].copy()
    image_array[x1_start:x1_end, y1_start:y1_end] = image_array[x2_start:x2_end, y2_start:y2_end]
    image_array[x2_start:x2_end, y2_start:y2_end] = temp
    
    # add a random flip or rotation
    flip_or_rotate = random.randint(0, 4)
    if flip_or_rotate == 0:  # flip horrizontally
        image_array[x1_start:x1_end, y1_start:y1_end] = np.fliplr(image_array[x1_start:x1_end, y1_start:y1_end])
        image_array[x2_start:x2_end, y2_start:y2_end] = np.fliplr(image_array[x2_start:x2_end, y2_start:y2_end])
    if flip_or_rotate == 1:  # flip vertically
        image_array[x1_start:x1_end, y1_start:y1_end] = np.flipud(image_array[x1_start:x1_end, y1_start:y1_end])
        image_array[x2_start:x2_end, y2_start:y2_end] = np.flipud(image_array[x2_start:x2_end, y2_start:y2_end])
    elif flip_or_rotate == 2:  # rotate 90 degrees clockwise
        image_array[x1_start:x1_end, y1_start:y1_end] = np.rot90(image_array[x1_start:x1_end, y1_start:y1_end], k=-1)
        image_array[x2_start:x2_end, y2_start:y2_end] = np.rot90(image_array[x2_start:x2_end, y2_start:y2_end], k=-1)
    elif flip_or_rotate == 3:  # rotate 90 degrees clockwise
        image_array[x1_start:x1_end, y1_start:y1_end] = np.rot90(image_array[x1_start:x1_end, y1_start:y1_end], k=1)
        image_array[x2_start:x2_end, y2_start:y2_end] = np.rot90(image_array[x2_start:x2_end, y2_start:y2_end], k=1)
    
    return Image.fromarray(image_array)

def create_gif(images, duration):
    gif_data = io.BytesIO()
    print("Saving gif (might take a while).")
    imageio.mimsave("me.gif", images, format='gif', duration=duration)

def simulated_annealing(image, init_temp=1000,alpha=0.01,min_square_size=32,max_square_size=128, duration=.01):
    image.convert("RGB")
    image = ImageOps.fit(image, (256, 256), Image.ANTIALIAS)
    
    # Set the initial state and temperature
    current_state = image.copy()
    # current_score = sum(calculate_distance(current_state, i, j) for i in range(current_state.width) for j in range(current_state.height))
    temperature = init_temp
    
    # Iterate the algorithm for the specified number of iterations
    iterations = 100
    # display_freq = 100#int(iterations/100)
    # print(f"Displaying every {display_freq}.")
    images = [image]
    print(f"Running for {iterations}.")
    for i in tqdm.tqdm(range(iterations)):
        # if i % display_freq == 0 and i > 0:
        #     current_state.show()
        try:
            # Get two random coordinates
            square_size = random.randint(min_square_size, max_square_size)
            x1, y1 = get_random_coordinate(image, square_size)
            x2, y2 = get_random_coordinate(image, square_size)

            current_state = swap_pixels(current_state, x1, y1, x2, y2, square_size)
            images.append(current_state.copy())
        except IndexError:
            continue

        # Lower the temperature
        temperature *= (1 - alpha)

    create_gif(images,duration)

    return current_state

image_path = "lauraphoto.jpg"
img = Image.open(image_path)
simulated_annealing(img)
# img.show()
# img2 = 