# DALI Challenge : Barnacles

# Introduction

This project is my submission for the DALI application challenge, focusing on the robust detection and counting of barnacles in images. Driven by an interest in comparing methodologies, I've developed and evaluated three distinct pipelines: traditional computer vision techniques, a custom-trained deep learning model (YOLO11), and an approach using an external API. This comparative study aims to not only identify the most optimal solution but also to provide insights into the strengths and weaknesses of classical computer vision methods versus state-of-the-art deep learning architectures like YOLO11. More specifically, I employ the given pipelines due to the following motivations: 

1. Traditional CV: To establish a baseline performance and explore popular, rule-based approaches.
2. Trained Model: To utilize the power of deep learning and test against Traditional CV methods.
3. API Model: To explore the feasibility and convenience of utilizing readily available, high-performance pre-trained models."


Each pipeline is thoroughly described within its respective Jupyter notebook, where I've also extensively evaluated their performance across various relevant conditions and hyperparameter configurations, aiming to provide comprehensive observations and explanations.

Feel free to reach out with any questions or suggestions regarding this project.


## Table of Contents 
-[Installation](#installation) 

-[Output](#output)

-[Files](#files)

-[Dependencies](#dependencies) 

-[Author](#author)


## Installation
1. Clone the repository:
```bash
git clone https://github.com/arses-ui/Barnacles.git
cd your-repository-name
```

2. Create and activate a virtual environment:
```bash
python -m venv venv 
a. source venv/Scripts/activate # For Windows + Git Bash or WSL
b. venv\Scripts\activate #For Windows Command Prompt 
c. venv/bin/activate #For Mac/Linux
```

3. Install requried packages:
```bash
pip install -r requirements.txt 
 ```
5. Run the Project Demo locally:
```bash
streamlit run app.py
```

## Files

The files for this project have been refactored into different directories for better organization. The contents of the files are as follows: 

- scripts
    - images
        - `mask1.png` 
        - `mask2.png`
        - `sample_image.png`
        - `sample_image2.png`
    - `traditionalcv.ipynb`  Jupyter Notebook with the Computer Vision Pipeline 
    - `trained_model.ipynb`  Jupyter Notebook with the Trained model Pipeline 
    - `api-model.ipynb`   Jupyter Notebook with the API-model Pipeline 
    - `best.pt`  Best parameters for the trained model 

- Training_Results
    - `results.csv` Results of the training on the YOLO11 model
    - `results.png` Visulization of the training statistics on the YOLO11 model
    - `YOLO11.ipynb` Jupyter Notebook containing the training code

- `app.py` #app module that demonstrates my work as a web app utilizing streamlit platform

## Dependencies
- Python 3.9.0 [Download here](https://www.python.org/downloads/release/python-390/) (The python packages recommended are 3.9.0 <=>3.10.0 due to some of the libraries installed)
- Packages listed in requirements.txt 

## Author
Arses Prasai -[Github](https://github.com/arses-ui)         
Email : arses.prasai.28@dartmouth.edu                 

