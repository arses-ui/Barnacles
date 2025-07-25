# DALI Challenge : Barnacles

# Introduction
<font size=4>
This project is my submission for the DALI application challenge, focused on the  detection and counting of barnacles in images. Motivated by a desire to compare different methodologies, I developed and evaluated three distinct pipelines: traditional computer vision techniques, a custom-trained deep learning model (YOLOv11), and an approach leveraging an external API. This comparative study not only aims to identify the most effective solution, but also to highlight the strengths and limitations of classical rule-based methods versus modern deep learning architectures like YOLOv11.

The chosen pipelines were selected based on the following motivations:

1. Traditional Computer Vision: To establish a baseline and explore popular rule-based methods.
2. rained Model: To utilize deep learning’s capabilities and benchmark them against classical techniques.
3. API Model: To assess the practicality and performance of using pre-trained, publicly available models.

Each pipeline is properly documented and evaluated in its respective Jupyter notebook, where I tested their performance under various conditions and hyperparameter settings. These evaluations are supplemented with observations and analysis to offer a well-rounded understanding of each method's behavior.

Feel free to reach out with any questions or suggestions regarding this project.

## Table of Contents  
[Installation](#installation) &nbsp;&nbsp;&nbsp;&nbsp; [Output](#output) &nbsp;&nbsp;&nbsp;&nbsp; [Files](#files) &nbsp;&nbsp;&nbsp;&nbsp; [Dependencies](#dependencies) &nbsp;&nbsp;&nbsp;&nbsp; [Conclusion](#conclusion) &nbsp;&nbsp;&nbsp;&nbsp; [Learning Process](#learning-process) &nbsp;&nbsp;&nbsp;&nbsp; [Author](#author)


## Installation
1. Clone the repository:
```bash
git clone https://github.com/arses-ui/Barnacles.git
cd your-repository-name
```

2. Install virtual env to specify the python version
```bash 
pip install virtualenv
```

4. Create and activate a virtual environment:
```bash
virtualenv --python=c:/path/to/your/python3.9/python.exe venv
a. source venv/Scripts/activate # For Windows + Git Bash or WSL
b. venv\Scripts\activate #For Windows Command Prompt 
c. venv/bin/activate #For Mac/Linux
```

5.Install ipykernel <br>
If you want to run the jupyter notebooks and reproduce the results, install ipykernel using pip . 

```bash 
pip install ipykernel 
python -m ipykernel install --user --name=venv --display_name="Barnacles" 
```

6. Install requried packages:
```bash
pip install -r requirements.txt 
 ```
7. Run the Project Demo locally:
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
    - `helpers.py` python module consisting of helper functions used in the three notebooks

- Training_Results
    - `results.csv` Results of the training on the YOLO11 model
    - `results.png` Visulization of the training statistics on the YOLO11 model
    - `YOLO11.ipynb` Jupyter Notebook containing the training code

- `app.py` Python module to demonstrate my work as a web app utilizing streamlit platform

## Dependencies
- Python 3.9.0 [Download here](https://www.python.org/downloads/release/python-390/) (The python packages recommended are 3.9.0 <=>3.10.0 due to some of the libraries installed)
- Packages listed in requirements.txt 

## Learning Process

This challenge provided the perfect opportunity for me to explore and better understand classical computer vision. While I had some prior experience with image processing techniques and large multimodal models through my research work, I was relatively unfamiliar with traditional CV algorithms such as the Watershed algorithm and Connected Components labeling. Working on this project allowed me to engage with these techniques hands-on.

Additionally, I used this challenge as a chance to experiment with different large models before ultimately selecting YOLOv11 for further training. This growing familiarity with both classical CV methods and state-of-the-art vision models is something I look forward to applying in future projects, both at DALI and in other settings.


## Conclusion

Among the three approaches tested, the trained model demonstrated the most promise. Although it was trained on a relatively small dataset, it was able to accurately count the number of barnacles in images, particularly when the number of objects was limited. In contrast, the traditional image processing method performed poorer than expected, offering little adaptability and requiring  manual tuning of parameters to work for each individual image.

For future work, I believe the best direction is to curate a larger dataset and fine-tune a pre-trained object detection model such as YOLOv11. Although the data preparation process may be time-consuming, this approach would allow the model to generalize better across various image conditions and barnacle types, ultimately leading to more robust and scalable performance.

## Author
Arses Prasai -[Github](https://github.com/arses-ui)         
Email : arses.prasai.28@dartmouth.edu                 

