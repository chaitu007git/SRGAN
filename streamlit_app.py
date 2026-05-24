import pandas as pd 
import numpy as np 
import streamlit as st
from streamlit import runtime
import pickle

# Local imports
from components import home_page
from components import image_enhancer
from components import new_image_enhancer
from components import about_us
from config import PAGES
from model import run_inference

@st.cache_data
def load_data():
    """ 
    Loads the required sample image paths into the session state

    Args:
        None

    Returns:
        None
    """

    ## Load some sample images
    with open(r'val_images.pkl', 'rb') as f:
        val_data_list = pickle.load(f)

    ## Save the sample image paths into the session state
    st.session_state.val_images = val_data_list

    model = run_inference.init_model()

    return model

## Set the page tab title
st.set_page_config(page_title="Image Enhancer", page_icon="🤖", layout="wide")

## Load the initial app data
model = load_data()

## Landing page UI
def run_UI():
    """
    The main UI function to display the UI for the webapp
    """

    ## Set the page title and navigation bar
    st.sidebar.title('Select Menu')
    if st.session_state["page"]:
        page = st.sidebar.radio('Navigation', PAGES, index=st.session_state["page"])
    else:
        page = st.sidebar.radio('Navigation', PAGES, index=0)
    st.query_params = {"page": page}  # Update the query parameters with the selected page


    ## Display the page selected on the navigation bar
    if page == 'Home':
        st.sidebar.write(""" 
            ## About

            This project aims to enhance the resolution and quality of images using state-of-the-art Generative Adversarial Networks. 

            The project implements the Swift-SRGAN model architecture to enhance the resolution of low-quality images.
        """)
        st.title("Images Super Resolution")
        home_page.home_page_UI()

    elif page == 'Image Enhancer Example':
        st.sidebar.write(""" 
            ## About

            This project aims to enhance the resolution and quality of images using state-of-the-art Generative Adversarial Networks. 

            The project implements the Swift-SRGAN model architecture to enhance the resolution of low-quality images.
        """)
        st.title("Image Super Resolution Examples ")
        image_enhancer.image_enhancer_UI(model)
    
    elif page == 'Try Your Own Image':
        st.sidebar.write(""" 
            ## About
            
            This project aims to enhance the resolution and quality of images using state-of-the-art Generative Adversarial Networks. 

            The project implements the Swift-SRGAN model architecture to enhance the resolution of low-quality images.
        """)
        st.title("Try Your Own Image ")
        new_image_enhancer.new_image_enhancer_UI(model)

    else:
        st.sidebar.write(""" 
            ## About
            
            This project aims to enhance the resolution and quality of images using state-of-the-art Generative Adversarial Networks. 

            The project implements the Swift-SRGAN model architecture to enhance the resolution of low-quality images.
        """)
        st.title("Images Super Resolution ")
        about_us.about_us_UI()


if __name__ == '__main__':
    ## Load the streamlit app with "Recipe Recommender" as default page
    if runtime.exists():

        ## Get the page name from the URL
        url_params = st.query_params  # Using the updated query_params
        print(f"URL Parameters: {url_params}")  # Print the variable to track its value

        # Check if "page" is present in url_params
        if 'page' not in url_params or len(url_params['page']) == 0:
            # If there is no value for page in the url, set default value
            st.session_state.page = 0  # Default to first page if no query param

        else:
            # Extract the page value (like "Home")
            page_value = url_params['page'][0]
            print(f"Page Value Extracted: {page_value}")  # Print to track the extracted value

            # Check if the extracted page exists in PAGES
            if page_value in PAGES:
                # If it exists in the list, use its index
                st.session_state.page = PAGES.index(page_value)
            else:
                # If not, set the default value
                st.session_state.page = 0

        ## If the page is not loaded yet, set it up
        if 'loaded' not in st.session_state:
            if 'page' not in st.session_state:
                # Set the default page if it's not loaded
                st.session_state.page = 0
        
        ## Call the main UI function
        run_UI()
