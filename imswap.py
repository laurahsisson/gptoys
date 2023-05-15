from PIL import Image, ImageOps
import math
import numpy as np
import tqdm
import random
import io
import imageio
from multiprocessing import Pool
from io import BytesIO

INPATH = "lauraphoto3.jpg"
OUTPATH = "fade.mp4"

def calculate_distance(image, x, y):
    # Get the RGB value of the center pixel
    center_pixel = image.getpixel((x, y))

    # Define the neighborhood size (adjust this as needed)
    neighborhood_size = 6

    # Define the boundaries of the neighborhood
    left = max(0, x - neighborhood_size)
    upper = max(0, y - neighborhood_size)
    right = min(image.width - 1, x + neighborhood_size)
    lower = min(image.height - 1, y + neighborhood_size)

    # Get the RGB values of the neighbors
    neighbor_pixels = np.array(image.crop((left, upper, right+1, lower+1)))

    # Calculate the average RGB value of the neighbors
    avg_red = neighbor_pixels[:, 0].mean()
    avg_green = neighbor_pixels[:, 1].mean()
    avg_blue = neighbor_pixels[:, 2].mean()

    # Calculate the distance between the center pixel and the average neighbor color
    distance = math.sqrt((center_pixel[0] - avg_red)**2 + (center_pixel[1] - avg_green)**2 + (center_pixel[2] - avg_blue)**2)

    return distance

def get_random_coordinate(image):
    x = random.randint(0, image.width - 1)
    y = random.randint(0, image.height - 1)
    return x, y

def can_swap_pixels(image, x1, y1, x2, y2):
    # Get the distances between the pixels and their neighbors at their current positions
    distance1 = calculate_distance(image, x1, y1)
    distance2 = calculate_distance(image, x2, y2)

    # Swap the pixels and get the distances between the pixels and their neighbors in the new positions
    t1 = image.getpixel((x1, y1))
    t2 = image.getpixel((x2, y2))
    image.putpixel((x1, y1), t2)
    image.putpixel((x2, y2), t1)
    new_distance1 = calculate_distance(image, x1, y1)
    new_distance2 = calculate_distance(image, x2, y2)
    image.putpixel((x1, y1), t1)
    image.putpixel((x2, y2), t2)

    # Calculate the sum of distances before and after swapping the pixels
    sum_before = distance1 + distance2
    sum_after = new_distance1 + new_distance2

    # Return the difference in sums
    return sum_before - sum_after

def create_gif(images, duration):
    gif_data = io.BytesIO()
    print("Saving mp4 (might take a while).")
    writer = imageio.get_writer(OUTPATH, fps=1/duration)
    for im in images:
        writer.append_data(np.array(im))
    writer.close()

def simulated_annealing(image, init_temp=10000, alpha=1e-3, duration=.01,num_frames=100):
    # Calculate the initial score
    # print("Calculate initial score")
    # score = sum([calculate_distance(image, x, y) for x in tqdm.tqdm(range(image.width)) for y in range(image.height)])
    new_width  = 256
    new_height = new_width * image.height // image.width 
    image = ImageOps.fit(image, (new_width, new_height), Image.ANTIALIAS)
    with BytesIO() as output:
        # Convert the image to PNG format and save it to the BytesIO object
        image.save(output, 'PNG')
        # Get the value of the BytesIO object and assign it to a variable
        image = Image.open(io.BytesIO(output.getvalue()))
    # Initialize the temperature
    temperature = init_temp

    # Set the number of iterations based on the size of the image (adjust this as needed)
    iterations = int(image.width * image.height * 5)
    print(f"Running for {iterations}.")

    # Perform the simulated annealing algorithm
    images = [image.copy()]
    skips = 0
    for i in tqdm.tqdm(range(iterations)):
        if i % int(iterations/num_frames) == 0 and i > 0:
            images.append(image.copy())
            # image.show()
        # Get two random coordinates
        x1, y1 = get_random_coordinate(image)
        x2, y2 = get_random_coordinate(image)

        # Check if the pixels can be swapped and if so, calculate the difference in score
        diff = can_swap_pixels(image, x1, y1, x2, y2)

        # If swapping the pixels decreases the score, swap them
        if diff > 0:
            t1 = image.getpixel((x1, y1))
            t2 = image.getpixel((x2, y2))
            image.putpixel((x1, y1), t2)
            image.putpixel((x2, y2), t1)
            # score -= diff

        # If swapping the pixels increases the score, swap them with a probability based on the temperature
        else:
            p = math.exp(diff / temperature)
            if random.random() < p:
                t1 = image.getpixel((x1, y1))
                t2 = image.getpixel((x2, y2))
                image.putpixel((x1, y1), t2)
                image.putpixel((x2, y2), t1)
                # score -= diff
            else:
                skips+=1

        # Decrease the temperature
        temperature *= (1 - alpha)
    images.append(image.copy())

    print(f"{skips}={skips/iterations}")
    create_gif(images,duration)

    # Return the final image and score
    return image

img = Image.open(INPATH)
simulated_annealing(img)
# img.show()
# img2 = 