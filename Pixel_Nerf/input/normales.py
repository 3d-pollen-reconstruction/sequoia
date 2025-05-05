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
    if len(sys.argv) != 3:
        print("Usage: python transparent_to_white.py input.png output.png")
        sys.exit(1)

    transparent_to_white(sys.argv[1], sys.argv[2])
