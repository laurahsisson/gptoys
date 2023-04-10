from PIL import Image, ImageOps
import math
import tqdm
import random
import io
import imageio

def calculate_distance(image, x, y):
    # Get the RGB value of the center pixel
    center_pixel = image.getpixel((x, y))

    # Define the neighborhood size (adjust this as needed)
    neighborhood_size = 2

    # Define the boundaries of the neighborhood
    left = max(0, x - neighborhood_size)
    upper = max(0, y - neighborhood_size)
    right = min(image.width - 1, x + neighborhood_size)
    lower = min(image.height - 1, y + neighborhood_size)

    # Initialize a list to hold the RGB values of the neighbors
    neighbor_pixels = []

    # Loop over the neighborhood and append the RGB values to the list
    for i in range(left, right + 1):
        for j in range(upper, lower + 1):
            if i != x or j != y:
                neighbor_pixels.append(image.getpixel((i, j)))

    # Calculate the average RGB value of the neighbors
    avg_red = sum([p[0] for p in neighbor_pixels]) / len(neighbor_pixels)
    avg_green = sum([p[1] for p in neighbor_pixels]) / len(neighbor_pixels)
    avg_blue = sum([p[2] for p in neighbor_pixels]) / len(neighbor_pixels)

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
    image = image.copy()
    temp = image.getpixel((x1, y1))
    image.putpixel((x1, y1), image.getpixel((x2, y2)))
    image.putpixel((x2, y2), temp)
    new_distance1 = calculate_distance(image, x1, y1)
    new_distance2 = calculate_distance(image, x2, y2)

    # Calculate the sum of distances before and after swapping the pixels
    sum_before = distance1 + distance2
    sum_after = new_distance1 + new_distance2

    # Return the difference in sums
    return sum_before - sum_after

def create_gif(images, duration):
    gif_data = io.BytesIO()
    print(f"Saving gif of length {len(images)} (might take a while).")
    imageio.mimsave("anneal.gif", images, format='gif', duration=duration)

def simulated_annealing(image, init_temp=10000, alpha=1e-6, duration=.01,num_frames=100):
    # Calculate the initial score
    # print("Calculate initial score")
    # score = sum([calculate_distance(image, x, y) for x in tqdm.tqdm(range(image.width)) for y in range(image.height)])
    image = ImageOps.fit(image, (256, 256), Image.ANTIALIAS)
    # Initialize the temperature
    temperature = init_temp

    # Set the number of iterations based on the size of the image (adjust this as needed)
    iterations = int(image.width * image.height * 100)
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
            temp = image.getpixel((x1, y1))
            image.putpixel((x1, y1), image.getpixel((x2, y2)))
            image.putpixel((x2, y2), temp)
            # score -= diff

        # If swapping the pixels increases the score, swap them with a probability based on the temperature
        else:
            p = math.exp(diff / temperature)
            if random.random() < p:
                temp = image.getpixel((x1, y1))
                image.putpixel((x1, y1), image.getpixel((x2, y2)))
                image.putpixel((x2, y2), temp)
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

image_path = "lauraphoto.jpg"
img = Image.open(image_path)
simulated_annealing(img)
# img.show()
# img2 = 