from inference_sdk import InferenceHTTPClient, InferenceConfiguration
import streamlit as st 
import pandas as pd 
import numpy as np 
from helpers import *
import os 

DATE_COLUMN = 'date/time'

def intro(): 
    import streamlit as st
    DATE_COLUMN = 'date/time'
    st.write("# Welcome to my Project.")
    st.sidebar.success("Select the project you want to chekck out")

    st.markdown("""
        This is my project for DALI LAB. applications While the task was to create a single automation 
        technique, I went ahead and tried to approach the problem in three different ways. In this 
        web app, I present all three of my different approaches. You can select which one you want to 
        check out below!!!
            
        ### Want to reach out for suggestions or providefeedback? 
        -Check out my [GitHub](https://github.com/arses-ui)\\
        -Send me a message through email: arses.prasai.28@dartmouth.edu

                """)

    st.markdown("""
    <style>
    div.stButton > button:first-child {
        background-color: #007bff; /* Blue */
        color: white;
        border-radius: 5px;
        border: none;
        padding: 10px 20px;
        font-size: 16px;
    }
    div.stButton > button:hover {
        background-color: #0056b3; /* Darker blue on hover */
    }
    </style>
    """, unsafe_allow_html=True)

def API_call(): 
    import streamlit as st 
    import time
    import numpy as np
    import base64
    from PIL import Image
    import tempfile 
    import shutil 


    #bckground picture

    original_title = '<h1 style="font-family: serif; color:white; font-size: 20px'
    st.markdown(original_title, unsafe_allow_html=True)


    # Set the background image
    background_image = """
    <style>
    [data-testid="stAppViewContainer"] > .main {
        background-image: url("https://images.unsplash.com/photo-1542281286-9e0a16bb7366");
        background-size: 100vw 100vh;  # This sets the size to cover 100% of the viewport width and height
        background-position: center;  
        background-repeat: no-repeat;
    }
    </style>
    """

    st.markdown(background_image, unsafe_allow_html=True)

    st.text_input("", placeholder="Streamlit CSS ")

    input_style = """
    <style>
    input[type="text"] {
        background-color: transparent;
        color: #a19eae;  // This changes the text color inside the input box
    }
    div[data-baseweb="base-input"] {
        background-color: transparent !important;
    }
    [data-testid="stAppViewContainer"] {
        background-color: transparent !important;
    }
    </style>
    """
    st.markdown(input_style, unsafe_allow_html=True)

    st.markdown(f"# {list(page_names_to_funcs.keys())[1]}")
    st.markdown(
        """
        This was the quickest approach. Through the [Roboflow Universe](https://universe.roboflow.com/stephen-7b2qu/barnacles-lnd34/model/1), I 
        came across a YOLO model that had been trained to detect Barnacles on a dataset of 255 images. Utilizing this model's API, feed infer the 
        number of Barnacles on our image.\
        

        However, there is a twist. Upon playing around with the model, 
        I discovered that the model only accurately detects Barnacles when there is a few nnumber of them on the scren as that is the type of dataset it was trained on. 
        Utilizing this fact, I take an extra step.\
        

        I divide the image into smaller 'tiles' (256+) and make the model individual run inference through every single one of them. While less efficient, it was much more accurate (~80%) while only taking few minutes to run.

""")
    # Initializing my API Client
    custom_configuration= InferenceConfiguration(confidence_threshold=0.3)
    CLIENT= InferenceHTTPClient(
    api_url ="https://serverless.roboflow.com", 
    api_key= "CW6dMrLkiMDw9IRcbujY"
    )

    #Displaying two widgets for inputting image or image URL
    img_file_buffer=None
    image_url_input= ""
    col1, col2 = st.columns(2)
    with col1: 
        img_file_buffer = st.file_uploader('Upload an image', type=['png', 'jpg', 'jpeg'])
    with col2: 
        input_path = st.text_input("Entire the Image URL here")
    
    output_directory = tempfile.mkdtemp()


    def crop_image_into_tiles(pil_image_object, output_folder):
        """
        Crops a PIL Image into a collection of smaller images (tiles) and saves them.
        Assumes pil_image_object is ALREADY a valid PIL.Image.Image object.
        """
        if not isinstance(pil_image_object, Image.Image):
            print(f"Error: Expected PIL.Image.Image, but got {type(pil_image_object)}")
            return None # Indicate failure if input type is wrong

        # Ensure output folder exists
        try:
            os.makedirs(output_folder, exist_ok=True)
        except PermissionError:
            print(f"Permission denied: Unable to create or access '{output_folder}'.")
            return None
        except Exception as e:
            print(f"An unexpected error occurred creating directory: {e}")
            return None

        img_width, img_height = pil_image_object.size
        tile_width = img_width // 16
        tile_height = img_height // 16
        tile_num = 0

        if tile_width == 0 or tile_height == 0:
            print(f"Warning: Image {img_width}x{img_height} is too small for 1/16 tiling. No tiles will be generated.")
            return 0

        for i in range(0, img_height, tile_height):
            for j in range(0, img_width, tile_width):
                right = min(j + tile_width, img_width)
                bottom = min(i + tile_height, img_height)
                box = (j, i, right, bottom)
                cropped_img = pil_image_object.crop(box)
                cropped_img.save(os.path.join(output_folder, f"tile_{tile_num}.png")) # Use os.path.join
                tile_num += 1

        print(f"Image cropped successfully. Generated {tile_num} tiles in {output_folder}")
        return tile_num


    @st.cache_data(show_spinner=False) # Underscore for _image_for_processing not needed if it's hashable
    def run_barnacle_analysis(image_for_processing_bytes_or_url): # This argument must be hashable
        """
        Performs the full barnacle analysis pipeline:
        1. Converts input (bytes or URL) to PIL Image.
        2. Crops the image into tiles and saves them.
        3. Runs inference on each tile using the Roboflow client.
        Returns total barnacles detected and a status message.
        """

        # 1. Conert hashable input (bytes or URL) into a PIL Image object
        pil_image_to_process = None
        if isinstance(image_for_processing_bytes_or_url, bytes):
            # It's bytes from an uploaded file
            try:
                pil_image_to_process = Image.open(io.BytesIO(image_for_processing_bytes_or_url))
            except Exception as e:
                st.warn(f"Error opening bytes as PIL image: {e}")
                return None, f"Image conversion error: {e}"
            
        elif isinstance(image_for_processing_bytes_or_url, str):
            # It's a URL string
            try:
                response = requests.get(image_for_processing_bytes_or_url)
                response.raise_for_status()
                pil_image_to_process = Image.open(io.BytesIO(response.content))
            except requests.exceptions.RequestException as e:
                print(f"Error downloading image from URL: {e}")
                return None, f"Download error: {e}"
            except Exception as e:
                print(f"Error processing URL content: {e}")
                return None, f"URL image process error: {e}"
            
        else:
            # Should not happen if input preparation logic is correct
            st.warn(f"Unhandled input type for cached function: {type(image_for_processing_bytes_or_url)}")
            return None, "Invalid input type for analysis."

        if pil_image_to_process is None:
            return 0, "Failed to load image for processing."

        # 2. Perform the cropping (tiles are saved to output_directory)
        tiles_count = crop_image_into_tiles(pil_image_to_process, output_directory) 

        if tiles_count is None or tiles_count == 0:
            return 0, "No tiles generated or cropping failed."

        # 3. Barnacle Inference Loop
        number_of_barnacles = 0
        number_of_images = tiles_count

        for i in range(number_of_images):
            with CLIENT.use_configuration(custom_configuration):
                image_tile_path = os.path.join(output_directory, f"tile_{i}.png")
                if not os.path.exists(image_tile_path):
                    print(f"Warning: Tile {image_tile_path} not found for inference.")
                    continue

                try:
                    result = CLIENT.infer(image_tile_path, model_id="barnacles-lnd34/1")
                    number_of_barnacles += len(result['predictions'])
                except Exception as e:
                    print(f"Error during inference for tile {i}: {e}")
                    continue

        return number_of_barnacles, "Success"


    #Trigger for the Analysis to start 
    analysis_triggered = st.button("Start Barnacle Analysis")
    if analysis_triggered:
        # Prepare the input for the cached function to be hashable
        hashable_input = None

        if img_file_buffer is not None:
            # Read the bytes from the UploadedFile. This makes it hashable.
            # Use seek(0) to ensure the buffer is read from the beginning
            # in case it was already read (e.g., by st.image above).
            img_file_buffer.seek(0)
            hashable_input = img_file_buffer.read()


        elif image_url_input:
            # URL string is already hashable
            hashable_input = image_url_input

        if hashable_input is None:
            st.warning("Please upload an image or provide a valid image URL to proceed.")

        else:
            with st.spinner("Running analysis... This might take a moment."):
                total_barnacles, status = run_barnacle_analysis(hashable_input)

            if status == "Success":
                st.metric("Total Barnacles Detected", value=total_barnacles)
                st.info("Analysis complete!")
            else:
                st.error(f"Analysis failed: {status}")
                st.warning(f"Please check the input image/URL. Reason: {status}")

    try: 
        shutil.rmtree(output_directory)
    except Exception as e: 
        st.warning(f"Error cleaning up the temporary directory: {e}")




        


page_names_to_funcs= {
"-":intro, 
"Project 1: API call": API_call
}

project_name = st.sidebar.selectbox("Choose a project", page_names_to_funcs.keys())
page_names_to_funcs[project_name]()