import random

def count_unique_colors(image, num_samples=2000):
    width, height = image.size
    all_pixels = [(x, y) for x in range(width) for y in range(height)]
    random.shuffle(all_pixels)
    num_samples = min(num_samples, len(all_pixels))
    unique_colors = set()
    
    for i in range(num_samples):
        color = image.getpixel(all_pixels[i])
        unique_colors.add(color)
    return len(unique_colors)