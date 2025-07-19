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
        This is my project for DALI LAB. While the task was to create a single automation 
        technique, I went ahead and tried to approach the problem in three different ways. In this 
        web app, I present all three of my different approaches. You can select which one you want to 
        check out below!!!
            
        ### Want to reach out for suggestions or providefeedback? 
        -Check out my [GitHub](https://github.com/arses-ui)\\
        -Send me a message through email: arses.prasai.28@dartmouth.edu

                """)

def API_call(): 
    import streamlit as st 
    import time
    import numpy as np
    import base64
    from PIL import Image


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
    
    col1, col2 = st.columns(2)
    with col1: 
        img_file_buffer = st.file_uploader('Upload a PNG image', type='png')
        if img_file_buffer is not None:
            image = Image.open(img_file_buffer)
            img_array = np.array(image)
    with col2: 
        input_path = st.text_input("Entire the Image URL here")

 
 
    
    
    @st.cache_data
    def input_data(input):


        output_directory  = "C:/Users/arses/Desktop/Cropped_images"

        try : 
            os.mkdir(output_directory)
            st.badge(f"Directory '{output_directory}' created successfully.", icon=":material/check:", color="green" )

        #Already created this file
        except FileExistsError: 
            st.badge(f"Directory '{output_directory}' already exists.",  icon=":material/thumb_up:")

        #Check access and permissions settings 
        except PermissionError: 
            st.markdown(":orange-badge[⚠️Permission denied: Unable to create '{output_directory}.]")

        #Any other type of errors 
        except Exception as e: 
            st.markdown(f":orange-badge[⚠️ An error occured:{e}.]")



        crop_image_into_tiles(input,output_directory)

        number_of_barnacles= 0
        number_of_images= directory_size(output_directory)

        #Progress
        progress_text = "Operation in progress. Hold on tight!"
        my_bar = st.progress(0, text= progress_text)
        for i in range(number_of_images):

            with CLIENT.use_configuration(custom_configuration):
                result = CLIENT.infer(f"{output_directory}/tile_{i}.png", model_id = "barnacles-lnd34/1")
            number_of_barnacles+= len(result['predictions'])
            my_bar.progress(i + 1/(number_of_images), text= progress_text)

        time.sleep(1)
        my_bar.empty() 
        st.badge(f"The number of barnacles present in the image is: {number_of_barnacles}")






page_names_to_funcs= {
"-":intro, 
"Project 1: API call": API_call
}

project_name = st.sidebar.selectbox("Choose a project", page_names_to_funcs.keys())
page_names_to_funcs[project_name]()