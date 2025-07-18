from PIL import Image
import base64
import os 

def crop_image_into_tiles(image_path, output_folder):
   
    """

    Crops an image into a collection of smaller images (tiles).

    Args:
        image_path (str): The path to the input image.
        output_folder (str): The folder to save the cropped tiles.
    """
    try:
        img = Image.open(image_path)

    except FileNotFoundError:
        print(f"Error: Image not found at {image_path}")
        return

    img_width, img_height = img.size
    tile_width, tile_height = img_width//16, img_height//16
    tile_num = 0

    for i in range(0, img_height, tile_height):
        for j in range(0, img_width, tile_width):
            box = (j, i, j + tile_width, i + tile_height)
            cropped_img = img.crop(box)
            cropped_img.save(f"{output_folder}/tile_{tile_num}.png")
            tile_num += 1
            
    print("Image cropped successfully")
    return


def image_to_base64(image_path): 
    with open(image_path, "rb") as image_file: 
        encoded_string = base64.b64encode(image_file.read()).decode('utf-8')


def directory_size(directory_path):    
    path = directory_path
    files_in_directory = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]
    num_files = len(files_in_directory)
    print(f"Number of files in '{directory_path}': {num_files}")
    return num_files



