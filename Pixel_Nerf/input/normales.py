from PIL import Image
import sys

def transparent_to_white(input_path: str, output_path: str):
    # Open the source image
    img = Image.open(input_path).convert("RGBA")

    # Create a white background image of the same size
    white_bg = Image.new("RGBA", img.size, (255, 255, 255, 255))

    # Composite the original image over the white background
    flattened = Image.alpha_composite(white_bg, img)

    # Drop alpha channel and save as RGB PNG or JPEG
    flattened.convert("RGB").save(output_path, "PNG")

if __name__ == "__main__":

    print("Usage: python transparent_to_white.py input.png output.png")
 
    transparent_to_white("./input/000001.png", "./input/norm_000001.png")
    print("Converted image saved as output.png")
