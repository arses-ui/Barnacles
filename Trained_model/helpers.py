from PIL import Image
import base64
import os 
import requests
import io
import tempfile
import pandas as pd
from ultralytics import YOLO
import numpy as np 
from scipy.ndimage import gaussian_filter
import cv2

from PIL import Image
import os

def compute_patch_coordinates(image_size, patch_size, overlap_percentage):
    img_width, img_height = image_size
    patch_width, patch_height = patch_size
    stride_x = int(patch_width * (1 - overlap_percentage))
    stride_y = int(patch_height * (1 - overlap_percentage))

    coords = []
    for y in range(0, img_height - patch_height + 1, stride_y):
        for x in range(0, img_width - patch_width + 1, stride_x):
            coords.append((x, y, x + patch_width, y + patch_height))

    # Add right/bottom edge patches if needed
    if img_width % patch_width != 0:
        for y in range(0, img_height - patch_height + 1, stride_y):
            coords.append((img_width - patch_width, y, img_width, y + patch_height))
    if img_height % patch_height != 0:
        for x in range(0, img_width - patch_width + 1, stride_x):
            coords.append((x, img_height - patch_height, x + patch_width, img_height))
    if (img_width % patch_width != 0) and (img_height % patch_height != 0):
        coords.append((img_width - patch_width, img_height - patch_height, img_width, img_height))

    return coords

def extract_patches_from_coords(img, coords, output_dir, prefix):
    os.makedirs(output_dir, exist_ok=True)
    for i, (left, top, right, bottom) in enumerate(coords):
        patch = img.crop((left, top, right, bottom))
        patch.save(os.path.join(output_dir, f"{prefix}_patch_{i:04d}.png"))




def process_images_in_folder(input_dir, patch_size=(256, 256), overlap_percentage=0.5, output_dir="output_patches"):
    supported_exts = [".jpg", ".jpeg", ".png", ".bmp", ".tiff"]

    image_files = [f for f in os.listdir(input_dir)
                   if os.path.splitext(f)[1].lower() in supported_exts]

    if not image_files:
        print("No valid images found in input directory.")
        return

    for image_file in image_files:
        image_path = os.path.join(input_dir, image_file)
        divide_image_into_overlapping_patches(image_path, patch_size, overlap_percentage, output_dir)




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
    

def save_file(filename:str, dataset1:list, dataset2:list):

    data = {'col1': [1, 2, 3], 'col2': [4, 5, 6]}
    df = pd.DataFrame(data)

    # Save the DataFrame to a CSV file
    df.to_csv(filename, index=False)

    # Save the DataFrame to a pickle file
    df.to_pickle('my_data.pkl')



def api_model(image_address, confidence_value, number_tiles=5):

    with tempfile.TemporaryDirectory() as tmpdir:
        output_directory  = tmpdir
        
    image_path = image_address

    try : 
        os.mkdir(output_directory)
        print(f"Directory '{output_directory}' created successfully.")

    #Already created this file
    except FileExistsError: 
        print(f"Directory '{output_directory}' already exists.")

    #Check access and permissions settings 
    except PermissionError: 
        print(f"Permission denied: Unable to create '{output_directory}.")

    #Any other type of errors 
    except Exception as e: 
        print(f"An error occured:{e}") 

    tiles = crop_image_into_tiles(image_path,output_directory, number_tiles)


    custom_configuration= InferenceConfiguration(confidence_threshold=confidence_value)
    CLIENT= InferenceHTTPClient(
        api_url ="https://serverless.roboflow.com", 
        api_key= "CW6dMrLkiMDw9IRcbujY"
    )
    number_of_barnacles= 0
    number_of_images= directory_size(output_directory)
    for i in range(number_of_images):
        with CLIENT.use_configuration(custom_configuration):
            result = CLIENT.infer(f"{output_directory}/tile_{i}.png", model_id = "barnacles-lnd34/1")
        number_of_barnacles+= len(result['predictions'])

    remove_files_from_directory(output_directory)

    return number_of_barnacles , tiles


def trained_model(image_address, number_tiles= 5): 
    
    model = YOLO('best.pt')

    with tempfile.TemporaryDirectory() as tmpdir:
        output_directory  = tmpdir
        
    image_path = image_address

    try : 
        os.mkdir(output_directory)
        print(f"Directory '{output_directory}' created successfully.")

    #Already created this file
    except FileExistsError: 
        print(f"Directory '{output_directory}' already exists.")

    #Check access and permissions settings 
    except PermissionError: 
        print(f"Permission denied: Unable to create '{output_directory}.")

    #Any other type of errors 
    except Exception as e: 
        print(f"An error occured:{e}")

    tiles = crop_image_into_tiles(image_path,output_directory, number_tiles)

    number_of_barnacles= 0
    number_of_images= directory_size(output_directory)
    for i in range(number_of_images):
        
        result = model(f"{output_directory}/tile_{i}.png", verbose= False)
        count = len(result[0].boxes)
        number_of_barnacles+= count
    remove_files_from_directory(output_directory)

    return number_of_barnacles, tiles


def traditional_cv_medthod(image_address, threshold):

    #filtering and grayscale
    image = Image.open(image_address).convert("RGB")
    grayscale_image = image.convert("L")
    image_array = np.array(grayscale_image)
    gaussian_filtered_image = gaussian_filter(image_array, sigma=1)

    #applying otsu threshold
    otsu_threshold, image_after_otsu = cv2.threshold(
    gaussian_filtered_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU, )

    #cleaning 
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    cleaned_image = cv2.morphologyEx(image_after_otsu, 
                           cv2.MORPH_OPEN,
                               kernel,
                           iterations=1)    

    # sure background area
    intermediary_bg = cv2.erode(cleaned_image, kernel, iterations=3)
    sure_bg = cv2.dilate(intermediary_bg, kernel, iterations=4)
    # Distance transform
    dist = cv2.distanceTransform(cleaned_image, cv2.DIST_L2,5)
    # foreground area
    ret, sure_fg = cv2.threshold(dist, threshold * dist.max(), 255, cv2.THRESH_BINARY)
    sure_fg = sure_fg.astype(np.uint8)
    # unknown area
    unknown = cv2.subtract(sure_bg, sure_fg)
    # Marker labelling
    # sure foreground 
    ret, markers = cv2.connectedComponents(sure_fg)

    # Add one to all labels so that background is not 0, but 1
    markers += 1
    # mark the region of unknown with zero
    markers[unknown == 255] = 0

    # watershed Algorithm
    img_array = np.array(image).astype(np.uint8)
    markers = cv2.watershed(img_array, markers)

    labels = np.unique(markers)

    barnacles = []
    for label in labels[2:]:  

    # Create a binary image in which only the area of the label is in the foreground and the rest of the image is in the background   
        target = np.where(markers == label, 255, 0).astype(np.uint8)
    
    # Perform contour extraction on the created binary image
        contours, hierarchy = cv2.findContours(
            target, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        barnacles.append(contours[0])

    # Draw the outline
    image = cv2.drawContours(image_array, barnacles, -1, color=(0, 23, 223), thickness=2)


    return len(barnacles)
        

 