from PIL import Image, ImageOps
import math
import tqdm
import random
import numpy as np
import io
import imageio

def get_random_coordinate(image, size):
    """
    Returns a random coordinate on the image.
    """
    x = random.randint(size // 2, image.width - size // 2 - 1)
    y = random.randint(size // 2, image.height - size // 2 - 1)
    return x, y

def get_square_coords(img,center_x, center_y, size):
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
    x1_start, y1_start, x1_end, y1_end = get_square_coords(img,x1, y1, square_size)
    x2_start, y2_start, x2_end, y2_end = get_square_coords(img,x2, y2, square_size)
    
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
    elif flip_or_rotate == 3:  # rotate 90 degrees anticlockwise
        image_array[x1_start:x1_end, y1_start:y1_end] = np.rot90(image_array[x1_start:x1_end, y1_start:y1_end], k=1)
        image_array[x2_start:x2_end, y2_start:y2_end] = np.rot90(image_array[x2_start:x2_end, y2_start:y2_end], k=1)
    
    return Image.fromarray(image_array)

def create_gif(images, duration):
    gif_data = io.BytesIO()
    print("Saving gif (might take a while).")
    imageio.mimsave("me.gif", images, format='gif', duration=duration)

def simulated_annealing(image, init_temp=1000,alpha=0.01,min_square_size=32,max_square_size=256, duration=.1):
    image.convert("RGB")
    new_width  = 1024
    new_height = new_width * image.height // image.width 
    image = ImageOps.fit(image, (new_width, new_height), Image.ANTIALIAS)
    
    # Set the initial state and temperature
    current_state = image.copy()
    # current_score = sum(calculate_distance(current_state, i, j) for i in range(current_state.width) for j in range(current_state.height))
    temperature = init_temp
    
    # Iterate the algorithm for the specified number of iterations
    iterations = 50
    # display_freq = 100#int(iterations/100)
    # print(f"Displaying every {display_freq}.")
    images = [image.copy()]
    print(f"Running for {iterations} steps.")
    for i in tqdm.tqdm(range(iterations)):
        # if i % display_freq == 0 and i > 0:
        #     current_state.show()
        try:
            # Get two random coordinates
            square_size = random.randint(min_square_size, max_square_size)
            x1, y1 = get_random_coordinate(image, square_size)
            x2, y2 = get_random_coordinate(image, square_size)
            print(x1,y1,x2,y2)

            current_state = swap_pixels(current_state, x1, y1, x2, y2, square_size)
            images.append(current_state.copy())
        except IndexError:
            continue
        except ValueError:
            print("F")
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