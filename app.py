from inference_sdk import InferenceHTTPClient, InferenceConfiguration
import streamlit as st 
import pandas as pd 
import numpy as np 
from scripts.helpers import *
import os 

DATE_COLUMN = 'date/time'

def intro(): 

    import streamlit as st
    
    st.set_page_config(page_title= "Arses' Project", page_icon = '👋' )
    
    def set_bg_url():
        '''
        A function to unpack an image from url and set as bg.
        Returns
        -------
        The background.
        '''
            
        st.markdown(
            f"""
            <style>
            .stApp {{
                background: url("https://images.pexels.com/photos/1166644/pexels-photo-1166644.jpeg?_gl=1*nzqsrx*_ga*MTk5NDcyOTMxMC4xNzUzMjM1MTcx*_ga_8JE65Q40S6*czE3NTMyMzUxNzEkbzEkZzEkdDE3NTMyMzUyNjEkajMwJGwwJGgw");
                background-size: cover
            }}
            </style>
            """,
            unsafe_allow_html=True
        )
    

    st.markdown("""
    <style>
    /* Existing button styles */
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

    /* New Bounding Box Style */
    .bounding-box {
        border: 2px solid #4CAF50; /* Green border */
        border-radius: 10px; /* Rounded corners */
        padding: 20px; /* Space inside the box */
        margin-bottom: 20px; /* Space below the box */
        background-color: #f0fff0; /* Light green background */
        box-shadow: 3px 3px 8px rgba(0, 0, 0, 0.1); /* Subtle shadow */
    }

    /* Style for text inside the bounding box, if desired */
    .bounding-box p {
        font-size: 1.1em;
        line-height: 1.6;
        color: #333;
    }

    </style>
    """, unsafe_allow_html=True) 


    st.markdown("""<div class="bounding-box">
                <span style="color:black"> 
            <h1>Welcome to my project!
                </h1>
                </span>
                </div>
        """, unsafe_allow_html=True)
    
    st.sidebar.success("Select the project you want to check out")
        
    st.markdown(f"""
    <div class="bounding-box">
        <p>
        As part of my DALI Lab application, I built this project to tackle the challenge of counting large numbers of barnacles. I approached the problem using three different techniques, both to explore the most optimal solution and to expand my understanding of computer vision. 
            All three methods are available in this interactive web app. Use the sidebar to explore and compare them!
        </p>
    </div>
    """, unsafe_allow_html=True)


    st.markdown(f"""
    <div class="bounding-box">
         <span style="color:black"> 
        <h3>Want to reach out or learn more about the project/challenge?</h3>
        <p>
            <li>Check out the code in my <a href="https://github.com/arses-ui/Barnacles.git" target="_blank">Github Repository</a></li>
            <li>Check out the DALI Challenge <a href="https://dalilab.notion.site/Data-Challenge-2b3ecf13c9e14ce18932c95b095519a3">here!</a></li>
            <li>Send me a message through email: <a href="mailto:arses.prasai.28@dartmouth.edu">arses.prasai.28@dartmouth.edu</a></li>
        </p></span>     
    </div>
    """, unsafe_allow_html=True)




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
    set_bg_url()






def Trained_model(): 

    import streamlit as st 
    from PIL import Image
    import tempfile 
    import shutil 


    st.set_page_config(page_title= "Models", page_icon = '🖥️' )
    


     #function to change the background color 
    def set_bg_url():
        '''
        A function to unpack an image from url and set as bg.
        Returns
        -------
        The background.
        '''
            
        st.markdown(
            f"""
            <style>
            .stApp {{
                background: url("https://images.pexels.com/photos/1166644/pexels-photo-1166644.jpeg?_gl=1*nzqsrx*_ga*MTk5NDcyOTMxMC4xNzUzMjM1MTcx*_ga_8JE65Q40S6*czE3NTMyMzUxNzEkbzEkZzEkdDE3NTMyMzUyNjEkajMwJGwwJGgw");
                background-size: cover
            }}
            </style>
            """,
            unsafe_allow_html=True
        )
    
    set_bg_url()

    

    st.markdown("""
    <style>
    /* Existing button styles */
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

    /* New Bounding Box Style */
    .bounding-box {
        border: 2px solid #4CAF50; /* Green border */
        border-radius: 10px; /* Rounded corners */
        padding: 20px; /* Space inside the box */
        margin-bottom: 20px; /* Space below the box */
        background-color: #f0fff0; /* Light green background */
        box-shadow: 3px 3px 8px rgba(0, 0, 0, 0.1); /* Subtle shadow */
    }

    /* Style for text inside the bounding box, if desired */
    .bounding-box p {
        font-size: 1.1em;
        line-height: 1.6;
        color: #333;
    }

    </style>
    """, unsafe_allow_html=True) 


    st.markdown("""<div class="bounding-box">
                <span style="color:black"> 
            <h1>Deep Learning Models
                </h1>
                </span>
                </div>
        """, unsafe_allow_html=True)
    

    st.markdown(f"""
    <div class="bounding-box">
        <p>
        In this page, I try to solve the problem with the assistance of two tools. First is a YOLO model that had been pre-trained on 
        detecting barnacles. Second is a YOLO11s model that I trained on my own using datasets I found in <a href="https://universe.roboflow.com/stephen-7b2qu/barnacles-lnd34/model/1" target="_blank">Roboflow Universe.</a>
    </p>

       
    </div>
    """, unsafe_allow_html=True)


    st.markdown(f"""
    <div class="bounding-box">
                <p>
    The dataset I used to train my own model contained relatively few barnacles per image, similar to the dataset used to train the pre-trained model 
                I access through an API. As a result, both models struggled to accurately detect barnacles in images with a large number of them. To
                 address this limitation, I introduced an additional step in my approach: breaking each input image into approximately 30 smaller 
                tiles and running inference on each tile individually.
                </p>
                </div>  
        """, unsafe_allow_html=True)
    
    
    #Displaying two widgets for inputting image or image URL
    img_file_buffer=None
    image_url_input= ""

    col1, col2 = st.columns(2)
    with col1: 
        result = f'<p style="font-family:sans-serif; color:Green; font-size: 20px;">Upload an image</p>'
        st.markdown(result, unsafe_allow_html=True)
        img_file_buffer = st.file_uploader(label= "",type=['png', 'jpg', 'jpeg'],)
    with col2:    
        result = f'<p style="font-family:sans-serif; color:Green; font-size: 20px;">Entire the Image URL</p>'
        st.markdown(result, unsafe_allow_html=True)
        image_url_input = st.text_input("")
    
    output_directory = tempfile.mkdtemp()
    st.markdown(''' <p style="font-family:sans-serif; color:Green; font-size: 20px;">Select the model you want to use</p>''', unsafe_allow_html=True)
    option = st.selectbox(
    '',
    ('Trained Model', 'API model'),
    index=None,
    placeholder="Select contact method..."
    )

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
        tile_width = img_width // 5
        tile_height = img_height // 5
        tile_num = 0

        if tile_width == 0 or tile_height == 0:
            print(f"Warning: Image {img_width}x{img_height} is too small for 1/30 tiling. No tiles will be generated.")
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
    def run_barnacle_analysis_api(image_for_processing_bytes_or_url): # This argument must be hashable
        """
        Performs the full barnacle analysis pipeline:
        1. Converts input (bytes or URL) to PIL Image.
        2. Crops the image into tiles and saves them.
        3. Runs inference on each tile using the Roboflow client.
        Returns total barnacles detected and a status message.
        """

        #  Convert hashable input (bytes or URL) into a PIL Image object
        pil_image_to_process = None
        if isinstance(image_for_processing_bytes_or_url, bytes):
            # It's bytes from an uploaded file
            try:
                pil_image_to_process = Image.open(io.BytesIO(image_for_processing_bytes_or_url))
            except Exception as e:
                st.warn(f"Error opening bytes as PIL image: {e}")
                return None, f"Image conversion error: {e}"
            
        elif isinstance(image_for_processing_bytes_or_url, str):
            
            # It's a string, could be a URL or a local path
            input_string = image_for_processing_bytes_or_url

            # Check if it's a local file path first
            if os.path.exists(input_string):
                try:
                    # Check if it's a valid image file using Pillow's verification
                    with Image.open(input_string) as img:
                        img.verify() # Verify if it's an image without fully loading it
                    
                    # If verification passes, it's a local image. Load it.
                    pil_image_to_process = Image.open(input_string)
                    pil_image_to_process.load() # Load image data into memory
                    print(f"Loaded local image from path: {input_string}")
                except FileNotFoundError:
                    error_message = f"Local file not found: {input_string}"
                    st.warn(error_message)
                except (IOError, SyntaxError) as e:
                    error_message = f"Local file is not a valid image: {e}"
                    st.warn(error_message)
                except Exception as e:
                    error_message = f"An unexpected error occurred with local file: {e}"
                    st.warn(error_message)
            
            # If not a local file path, check if it is a URL
            else:
                #URL validation
                # For simplicity, we'll proceed assuming it's a URL if it's not a local path
                print(f"Attempting to load as URL: {input_string}")
                try:
                    # Make a HEAD request first to check content type if you want to be more efficient
                    response = requests.get(input_string, timeout=10) # Added a timeout
                    response.raise_for_status() # Raises an HTTPError for bad responses 

                    # Check content-type if you want to explicitly verify it's an image before opening
                    content_type = response.headers.get('Content-Type', '').lower()
                    if not content_type.startswith('image/'):
                        error_message = f"URL content type is '{content_type}', not an image."
                        st.warn(error_message)
                    else:
                        pil_image_to_process = Image.open(io.BytesIO(response.content))
                        print(f"Loaded image from URL: {input_string}")

                except requests.exceptions.RequestException as e:
                    error_message = f"Error downloading image from URL: {e}"
                    print(error_message) 
                except Exception as e:
                    error_message = f"Error processing URL content: {e}"
                    print(error_message) 
        

        else:
            st.warn(f"Unhandled input type for cached function: {type(image_for_processing_bytes_or_url)}")
            return None, "Invalid input type for analysis."

        if pil_image_to_process is None:
            return 0, "Failed to load image for processing."

        # Perform the cropping (tiles are saved to output_directory)
        tiles_count = crop_image_into_tiles(pil_image_to_process, output_directory) 

        if tiles_count is None or tiles_count == 0:
            return 0, "No tiles generated or cropping failed."
        
        # Initializing my API Client
        custom_configuration= InferenceConfiguration(confidence_threshold=0.2)
        CLIENT= InferenceHTTPClient(
        api_url ="https://serverless.roboflow.com", 
        api_key= "CW6dMrLkiMDw9IRcbujY"
        )

        # Barnacle Inference Loop
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
    

    def run_barnacle_analysis_trained(image_for_processing_bytes_or_url): # This argument must be hashable
        """
        Performs the full barnacle analysis pipeline:
        1. Converts input (bytes or URL) to PIL Image.
        2. Crops the image into tiles and saves them.
        3. Runs inference on each tile using the Roboflow client.
        Returns total barnacles detected and a status message.
        """

        # 1. Convert hashable input (bytes or URL) into a PIL Image object
        pil_image_to_process = None
        if isinstance(image_for_processing_bytes_or_url, bytes):
            # It's bytes from an uploaded file
            try:
                pil_image_to_process = Image.open(io.BytesIO(image_for_processing_bytes_or_url))
            except Exception as e:
                st.warn(f"Error opening bytes as PIL image: {e}")
                return None, f"Image conversion error: {e}"
            
        elif isinstance(image_for_processing_bytes_or_url, str):
            
            # It's a string, could be a URL or a local path
            input_string = image_for_processing_bytes_or_url

            # Check if it's a local file path first
            if os.path.exists(input_string):
                try:
                    # Check if it's a valid image file using Pillow's verification
                    with Image.open(input_string) as img:
                        img.verify() # Verify if it's an image without fully loading it
                    
                    # If verification passes, it's a local image. Load it.
                    pil_image_to_process = Image.open(input_string)
                    pil_image_to_process.load() # Load image data into memory
                    print(f"Loaded local image from path: {input_string}")
                except FileNotFoundError:
                    error_message = f"Local file not found: {input_string}"
                    st.warn(error_message)
                except (IOError, SyntaxError) as e:
                    error_message = f"Local file is not a valid image: {e}"
                    st.warn(error_message)
                except Exception as e:
                    error_message = f"An unexpected error occurred with local file: {e}"
                    st.warn(error_message)
            
            # If not a local file path, check if it is a URL
            else:
                #URL validation
                # For simplicity, we'll proceed assuming it's a URL if it's not a local path
                print(f"Attempting to load as URL: {input_string}")
                try:
                    # Make a HEAD request first to check content type if you want to be more efficient
                    response = requests.get(input_string, timeout=10) # Added a timeout
                    response.raise_for_status() # Raises an HTTPError for bad responses 

                    # Check content-type if you want to explicitly verify it's an image before opening
                    content_type = response.headers.get('Content-Type', '').lower()
                    if not content_type.startswith('image/'):
                        error_message = f"URL content type is '{content_type}', not an image."
                        st.warn(error_message)
                    else:
                        pil_image_to_process = Image.open(io.BytesIO(response.content))
                        print(f"Loaded image from URL: {input_string}")

                except requests.exceptions.RequestException as e:
                    error_message = f"Error downloading image from URL: {e}"
                    print(error_message) 
                except Exception as e:
                    error_message = f"Error processing URL content: {e}"
                    print(error_message) 
        
        else:
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
        model = YOLO('scripts/best.pt')


        for i in range(number_of_images):
            image_tile_path = os.path.join(output_directory, f"tile_{i}.png")
            if not os.path.exists(image_tile_path):
                print(f"Warning: Tile {image_tile_path} not found for inference.")
                continue

            try: 
                result = model(f"{output_directory}/tile_{i}.png", verbose= False)
                count = len(result[0].boxes)
                number_of_barnacles+= count

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
            # Use seek(0) to ensure the buffer is read from the beginning in case it was already read (e.g., by st.image above).
            img_file_buffer.seek(0)
            hashable_input = img_file_buffer.read()


        elif image_url_input:
            # URL string is already hashable
            hashable_input = image_url_input

        if hashable_input is None:
            st.markdown(
                    """
                        <p style="color:red; font-weight:bold;">Error! Please enter a valid image or URL address</p>
                        """,
                     unsafe_allow_html=True
                        )


        else:
            if option == 'API model':
                with st.spinner(""):
                    total_barnacles, status = run_barnacle_analysis_api(hashable_input)

            elif option  == 'Trained Model':
                with st.spinner(""):
                    total_barnacles, status = run_barnacle_analysis_trained(hashable_input)
            else: 
                st.error(f"Please select a training model")
                status = "No model selected"

            if status == "Success":
                result = f'<p style="font-family:sans-serif; color:Black; font-size: 42px;">Total Number of Barnacles:{total_barnacles}</p>'
                st.markdown(result, unsafe_allow_html=True)
                st.markdown("""
                        <p style="color:green;font-size:20px; font-weight:bold;">✅ Analysis complete!</p>
                        """,
                     unsafe_allow_html=True
                        )
            else:
                st.markdown(
                    f"""
                        <p style="color:red; font-weight:bold;">Error! {status}</p>
                        """,
                     unsafe_allow_html=True
                        )

    try: 
        shutil.rmtree(output_directory)
    except Exception as e: 
        st.markdown(
                    f"""
                        <p style="color:red; font-weight:bold;">Error cleaning up the temporary directory: {e}</p>
                        """,
                     unsafe_allow_html=True
        )



def Computer_vision(): 
    import numpy as np
    import cv2
    from matplotlib import pyplot as plt 
    from scipy.ndimage import gaussian_filter
    import matplotlib.pyplot as plt
    from PIL import Image
    import matplotlib.image as mpimg
    from IPython.display import  display
    import tempfile 
    import shutil

    st.set_page_config(page_title= "CV", page_icon = '🎨' )

    #function to change the background color 
    def set_bg_url():
        '''
        A function to unpack an image from url and set as bg.
        Returns
        -------
        The background.
        '''
            
        st.markdown(
            f"""
            <style>
            .stApp {{
                background: url("https://images.pexels.com/photos/1166644/pexels-photo-1166644.jpeg?_gl=1*nzqsrx*_ga*MTk5NDcyOTMxMC4xNzUzMjM1MTcx*_ga_8JE65Q40S6*czE3NTMyMzUxNzEkbzEkZzEkdDE3NTMyMzUyNjEkajMwJGwwJGgw");
                background-size: cover
            }}
            </style>
            """,
            unsafe_allow_html=True
        )
    
    set_bg_url()
    
    st.markdown("""
    <style>
    /* Existing button styles */
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

    /* New Bounding Box Style */
    .bounding-box {
        border: 2px solid #4CAF50; /* Green border */
        border-radius: 10px; /* Rounded corners */
        padding: 20px; /* Space inside the box */
        margin-bottom: 20px; /* Space below the box */
        background-color: #f0fff0; /* Light green background */
        box-shadow: 3px 3px 8px rgba(0, 0, 0, 0.1); /* Subtle shadow */
    }

    /* Style for text inside the bounding box, if desired */
    .bounding-box p {
        font-size: 1.1em;
        line-height: 1.6;
        color: #333;
    }

    </style>
    """, unsafe_allow_html=True) 

    st.markdown("""<div class="bounding-box">
                <span style="color:black"> 
            <h1>Traditional Computer Vision 
                </h1>
                </span>
                </div>
        """, unsafe_allow_html=True)
  
    st.markdown(f"""
    <div class="bounding-box">
        <p>
        On this page, I present my approach to tackling the challenge using traditional computer vision and image processing techniques. 
        I chose this direction out of both curiosity and a desire to explore classical methods. This approach involves a sequence of operations, 
        including filtering, binarization, morphological transformations, and contour detection. I utilize well-known techniques such as Otsu’s 
        thresholding for binarization and the Watershed algorithm for segmentation. By identifying and counting distinct contour objects, the number 
        of barnacles in the image is estimated.
        </p>
    </div>
                
    """, unsafe_allow_html=True)
    st.markdown(f"""
    <div class="bounding-box">
        <p>
    Please upload an image of barnacles below. The processing pipeline will apply the steps above and return the original image with detected contours overlaid.
        </p>
            </div>
            """, unsafe_allow_html=True)


    col1, col2 = st.columns(2)
    
    img_file_buffer=None
    image_url_input= ""
    with col1: 
        result = f'<p style="font-family:sans-serif; color:Green; font-size: 20px;">Upload an image</p>'
        st.markdown(result, unsafe_allow_html=True)
        img_file_buffer = st.file_uploader(label= "",type=['png', 'jpg', 'jpeg'],)
    with col2:    
        result = f'<p style="font-family:sans-serif; color:Green; font-size: 20px;">Entire the Image URL</p>'
        st.markdown(result, unsafe_allow_html=True)
        image_url_input = st.text_input("")
    
    output_directory = tempfile.mkdtemp()

    @st.cache_data(show_spinner=False) 
    def image_processing(image_for_processing_bytes_or_url, THRESHOLD=0.27): 
        # 1. Convert hashable input (bytes or URL) into a PIL Image object
        pil_image_to_process = None
        if isinstance(image_for_processing_bytes_or_url, bytes):
            # It's bytes from an uploaded file
            try:
                pil_image_to_process = Image.open(io.BytesIO(image_for_processing_bytes_or_url))
            except Exception as e:
                st.warn(f"Error opening bytes as PIL image: {e}")
                return None, f"Image conversion error: {e}"
            
        elif isinstance(image_for_processing_bytes_or_url, str):
            
            # It's a string, could be a URL or a local path
            input_string = image_for_processing_bytes_or_url

            # Check if it's a local file path first
            if os.path.exists(input_string):
                try:
                    # Check if it's a valid image file using Pillow's verification
                    with Image.open(input_string) as img:
                        img.verify() # Verify if it's an image without fully loading it
                    
                    # If verification passes, it's a local image. Load it.
                    pil_image_to_process = Image.open(input_string)
                    pil_image_to_process.load() # Load image data into memory
                    print(f"Loaded local image from path: {input_string}")
                except FileNotFoundError:
                    error_message = f"Local file not found: {input_string}"
                    st.warn(error_message)
                except (IOError, SyntaxError) as e:
                    error_message = f"Local file is not a valid image: {e}"
                    st.warn(error_message)
                except Exception as e:
                    error_message = f"An unexpected error occurred with local file: {e}"
                    st.warn(error_message)
            
            # If not a local file path, check if it is a URL
            else:
                #URL validation
                # For simplicity, we'll proceed assuming it's a URL if it's not a local path
                print(f"Attempting to load as URL: {input_string}")
                try:
                    # Make a HEAD request first to check content type if you want to be more efficient
                    response = requests.get(input_string, timeout=10) # Added a timeout
                    response.raise_for_status() # Raises an HTTPError for bad responses 

                    # Check content-type if you want to explicitly verify it's an image before opening
                    content_type = response.headers.get('Content-Type', '').lower()
                    if not content_type.startswith('image/'):
                        error_message = f"URL content type is '{content_type}', not an image."
                        st.warn(error_message)
                    else:
                        pil_image_to_process = Image.open(io.BytesIO(response.content))
                        print(f"Loaded image from URL: {input_string}")

                except requests.exceptions.RequestException as e:
                    error_message = f"Error downloading image from URL: {e}"
                    print(error_message) 
                except Exception as e:
                    error_message = f"Error processing URL content: {e}"
                    print(error_message) 
        

        else:
            # Should not happen if input preparation logic is correct
            st.warn(f"Unhandled input type for cached function: {type(image_for_processing_bytes_or_url)}")
            return None, "Invalid input type for analysis."

        if pil_image_to_process is None:
            return 0, "Failed to load image for processing."

        image = pil_image_to_process.convert("RGB")
        grayscale_image = image.convert("L")
        image_array = np.array(grayscale_image)
        gaussian_filtered_image = gaussian_filter(image_array, sigma=1)

        otsu_threshold, image_after_otsu = cv2.threshold(
        gaussian_filtered_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU, 
        )
        
        # noise removal
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        cleaned_image = cv2.morphologyEx(image_after_otsu, 
                                cv2.MORPH_OPEN,
                                kernel,
                                iterations=1)
        

        # sure background area
        intermediary_bg = cv2.dilate(cleaned_image, kernel, iterations=3)
        sure_bg = cv2.dilate(intermediary_bg, kernel, iterations=3)

        # Distance transform
        dist = cv2.distanceTransform(cleaned_image, cv2.DIST_L2,5)

        # foreground area
        ret, sure_fg = cv2.threshold(dist, THRESHOLD * dist.max(), 255, cv2.THRESH_BINARY)
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
        fig, ax = plt.subplots(figsize=(5, 5))
        

        labels = np.unique(markers)
        coins = []
        for label in labels[2:]:  

        # Create a binary image in which only the area of the label is in the foreground 
        #and the rest of the image is in the background   
            target = np.where(markers == label, 255, 0).astype(np.uint8)
        
        # Perform contour extraction on the created binary image
            contours, hierarchy = cv2.findContours(
                target, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            coins.append(contours[0])

        # Draw the outline
        contour_image = cv2.drawContours(image_array, coins, -1, color=(255, 255, 255), thickness=2)
        plt.imshow(image, cmap ='grey')
        return  len(coins), "Success", contour_image
    

    
    #Trigger for the Analysis to start 
    analysis_triggered = st.button("Start Barnacle Analysis")
    if analysis_triggered:
        # Prepare the input for the cached function to be hashable
        hashable_input = None

        if img_file_buffer is not None:
            hashable_input = img_file_buffer.read()


        elif image_url_input:
            # URL string is already hashable
            hashable_input = image_url_input

        if hashable_input is None:
            st.markdown(
                    """
                        <p style="color:red; font-weight:bold;">Error! Please enter a valid image or URL address</p>
                        """,
                     unsafe_allow_html=True
                        )


        else:
            with st.spinner(""):
                total_barnacles, status , image= image_processing(hashable_input)

            if status == "Success":
                result = f'<p style="font-family:sans-serif; color:Black; font-size: 42px;">Total Number of Barnacles:{total_barnacles}</p>'
                st.markdown(result, unsafe_allow_html=True)
                st.markdown("""
                        <p style="color:green;font-size:20px; font-weight:bold;">✅ Analysis complete!</p>
                        """,
                     unsafe_allow_html=True
                        )
                new_title = '<p style="font-family:sans-serif; color:Green; font-size: 42px;">Visualization</p>'
                st.markdown(new_title, unsafe_allow_html=True)
                st.image(image)

            else:
                st.error(f"Analysis failed: {status}")
                st.warning(f"Please check the input image/URL. Reason: {status}")

    try: 
        shutil.rmtree(output_directory)
    except Exception as e: 
        st.markdown(
                    f"""
                        <p style="color:red; font-weight:bold;">Error cleaning up the temporary directory: {e}</p>
                        """,
                     unsafe_allow_html=True
        )




page_names_to_funcs= {
"Welcome Page":intro, 
"Approach 1: Traditional CV": Computer_vision,
"Approach 2: Training model": Trained_model

}

project_name = st.sidebar.selectbox("Choose a project", page_names_to_funcs.keys())
page_names_to_funcs[project_name]()
