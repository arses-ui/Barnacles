from inference_sdk import InferenceHTTPClient, InferenceConfiguration
import streamlit as st 
import pandas as pd 
import numpy as np 
from scripts.helpers import *
import os 

DATE_COLUMN = 'date/time'

def intro(): 

    import streamlit as st
    
    st.set_page_config(page_icon = '👋')
    
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
                background: url("https://upload.wikimedia.org/wikipedia/commons/thumb/7/72/Chthamalus_stellatus.jpg/2560px-Chthamalus_stellatus.jpg");
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
            <li>Chheck out the DALI Challenge <a href="https://dalilab.notion.site/Data-Challenge-2b3ecf13c9e14ce18932c95b095519a3">here!</a></li>
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
    import time
    import numpy as np
    import base64
    from PIL import Image
    import tempfile 
    import shutil 





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
                background: url("https://upload.wikimedia.org/wikipedia/commons/thumb/7/72/Chthamalus_stellatus.jpg/2560px-Chthamalus_stellatus.jpg");
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
    However, I ran into a problem. The pre-trained model I call using an API call is trained on dataset consisting of only a few barnacles
        per image. As a result, it did not perform great for images with large number of barnacles. Similarly, the the dataset I trained 
        to used to train my own model also contained a relatively small number of barnacles per image. As a result, I take an extra step in my 
        my approach, and break the image down into approximately 30 smaller images, and perform inference on all the images individually. 
                </p>
                </div>  
        """, unsafe_allow_html=True)
    
    
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
                background: url("https://upload.wikimedia.org/wikipedia/commons/thumb/7/72/Chthamalus_stellatus.jpg/2560px-Chthamalus_stellatus.jpg");
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
        In this page, I present my second approach to tackle the provided challenge. In all honestly, one of the main reasons I chose this
        approach was purely just to traditional computer vision image-processing techniques. To describe it briefly, I sequentlly perform 
        different transformations and processing techniques, namely filtering, binarization, Morphological Transformations and Contouring. I use popular methods like 
        Otsu' method for thresholding and Watershed Algorithm for image segmentation. Finally, by keeping track of the total different contour objects, 
        I count the number of barnacles in the given image. 
        </p>
    </div>
    """, unsafe_allow_html=True)

    custom_css = """
    <style>
    div[data-testid="stFileUploader"]>section[data-testid="stFileUploadDropzone"]>button[data-testid="baseButton-secondary"] {
        background-color: #4CAF50; /* Green */
        color: white;
    }
    </style>
    """
    st.markdown(custom_css, unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    
    img_file_buffer=None
    image_url_input= ""
    with col1: 
        img_file_buffer = st.file_uploader('Upload an image', type=['png', 'jpg', 'jpeg'])
    with col2:    
        input_path = st.text_input("Entire the Image URL here")
    
    output_directory = tempfile.mkdtemp()

    @st.cache_data(show_spinner=False) 
    def image_processing(image_for_processing_bytes_or_url, THRESHOLD=0.27): 
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
        contour_image = cv2.drawContours(image_array, coins, -1, color=(0, 23, 223), thickness=2)
        plt.imshow(image, cmap ='grey')
        return  len(coins), "Success", markers
    

    
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
                total_barnacles, status, display_image = image_processing(hashable_input)

            if status == "Success":
                st.metric("Total Barnacles Detected", value=total_barnacles)
                st.success("Analysis complete!", icon="✅")
                st.image(display_image, caption='Visualization of the barnacles')

            else:
                st.error(f"Analysis failed: {status}")
                st.warning(f"Please check the input image/URL. Reason: {status}")

    try: 
        shutil.rmtree(output_directory)
    except Exception as e: 
        st.warning(f"Error cleaning up the temporary directory: {e}")




page_names_to_funcs= {
"-":intro, 
"Approach 1: Training model": Trained_model, 
"Approach 2: Traditional CV": Computer_vision
}

project_name = st.sidebar.selectbox("Choose a project", page_names_to_funcs.keys())
page_names_to_funcs[project_name]()
