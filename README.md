# DALI Challenge : Barnacles

# Introduction
<font size=4>
This project is my submission for the DALI application challenge, focused on the  detection and counting of barnacles in images. Motivated by a desire to compare different methodologies, I developed and evaluated three distinct pipelines: traditional computer vision techniques, a custom-trained deep learning model (YOLOv11), and an approach leveraging an external API. This comparative study not only aims to identify the most effective solution, but also to highlight the strengths and limitations of classical rule-based methods versus modern deep learning architectures like YOLOv11.

The chosen pipelines were selected based on the following motivations:

1. Traditional Computer Vision: To establish a baseline and explore popular rule-based methods.
2. Fine Tuning pre-trained Model: To utilize deep learning’s capabilities and benchmark them against classical techniques.

Each pipeline is properly documented and evaluated in its respective Jupyter notebook. These notebooks are supplemented with observations observations and analysis to offer my understandings. 

Feel free to reach out with any questions or suggestions regarding this project.

## Quick demo 

![alt text](Demo/DALIdemo.gif)

watch the demo video [here!]

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


```
├── __pycache__
├── app.py
├── Demo
├── README.md
├── TraditionalCV
│   ├── images
│   │   ├── sample_image.png
│   │   └── sample_image2.png
│   ├── masks
│   │   ├── mask1.png
│   │   └── mask2.png
│   └── traditionalcv.ipynb
└── Trained_model
    ├── COCO_dataset
    ├── helpers.py
    ├── image_patches
    ├── images
    ├── mask_patches
    ├── masks
    ├── Progressive_FineTune
    │   ├── Kfold_training_barnacles_PFT
    │   ├── trained-model_YOLOv8_PFT.ipynb
    │   ├── training_stage_1
    │   └── training_stage_2
    ├── README.md
    ├── Single_step_Finetune
    │   ├── Kfold_training_barnacles_SFT
    │   └── trained-model_YOLO8v_SFT.ipynb
    ├── unseen_images
    └── yolo_dataset          
```


## Conclusion

Among the three approaches tested, the trained model demonstrated the most promise. Although it was trained on a relatively small dataset, it was able to  count the number of barnacles in images pretty well, particularly when the number of objects was limited. In contrast, the traditional image processing method performed poorer than expected, offering little adaptability and requiring  manual tuning of parameters to work for each individual image.

For future work, I believe the best direction is to curate a larger dataset and fine-tune a pre-trained object detection model such as YOLOv11. Another approach that can be taken is progressive fine-tuning (what I do ) and training a model on datasets that closely resemble the tasks you want to perform before training it on your own dataset.

## Author
Arses Prasai -[Github](https://github.com/arses-ui)         
Email : arses.prasai.28@dartmouth.edu                 

