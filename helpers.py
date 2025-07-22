from PIL import Image
import base64
import os 
import requests
import io
import numpy as np 

def crop_image_into_tiles(image, output_folder):
   
    """

    Crops an image into a collection of smaller images (tiles).

    Args:
        image_path (str): The path to the input image.
        output_folder (str): The folder to save the cropped tiles.
    """
    img= None
    if isinstance(image, Image.Image):
        img = image

    elif isinstance(image, str): 

        #check of online URL
        if image.startswith("http://") or image.startswith("https://"):

            try:
                response = requests.get(image)
                response.raise_for_status() # Raise an exception for HTTP errors
                img = Image.open(io.BytesIO(response.content))

            except requests.exceptions.RequestException as e:
                print(f"Error downloading image from URL: {e}")
                return  # Return None on failure
            except Exception as e:
                print(f"Error processing downloaded image: {e}")
                return  # Return None on failure
        else:
            # It's a local file path
            try:
                img = Image.open(image)
            except FileNotFoundError:
                print(f"Error: Image not found at {image}")
                return

    img_width, img_height = img.size
    tile_width, tile_height = img_width//5, img_height//5
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

def clear_directory(directory):

    file_list = [f for f in os.listdir(directory)]
    for f in file_list: 
        os.remove(os.path.join(directory, f))



def remove_files_from_directory(directory_path): 
    """
    Removes all files within a specified directory. 
    Subdirectories and ther contens are not affected.
    """

    try: 
        for filename in os.listdir(directory_path): 
            file_path = os.path.join(directory_path, filename)
            if os.path.isfile(file_path): 
                os.remove(file_path)
        
    except OSError as e: 
        print(f"Errror:{e}")
    
