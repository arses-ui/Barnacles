# DALI Challenge : Barnacles

# Introduction
<font size=4>

My DALI application challenge project focuses on detecting and counting barnacles in images. To find the most effective approach, I developed and evaluated two distinct methodologies: a pipeline built on traditional computer vision techniques and another using fine-tuned deep learning models, specifically YOLOv11.

This comparative study aims to not only identify the superior solution but also to illuminate the strengths and limitations of classical rule-based methods versus modern deep learning architectures. Each pipeline's development and evaluation, complete with observations and analysis, are thoroughly documented in dedicated Jupyter notebooks. An additional README in the `Finetune_model` subdirectory details my learning journey with the deep learning models.

I'm eager for any questions or suggestions you might have about this project.

## Quick demo 

![alt text](Demo/DALIdemo.gif)

P.S The app isn't as informative or telling as the notebooks. I would suggest reading the notebooks to learn more about the project.

## Table of Contents  
[Installation](#installation) &nbsp;&nbsp;&nbsp;&nbsp;  [Files](#files) &nbsp;&nbsp;&nbsp;&nbsp; [Dependencies](#dependencies) &nbsp;&nbsp;&nbsp;&nbsp; [Conclusion](#conclusion) &nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp; [Author](#author)


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
├── CV_model
│   ├── images
│   │   ├── sample_image.png
│   │   └── sample_image2.png
│   ├── masks
│   │   ├── mask1.png
│   │   └── mask2.png
│   └── traditionalcv.ipynb
└── Finetune_model
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

## Dependencies
- Python 3.13.1 [Download here](https://www.python.org/downloads/release/python-3130/)
- Packages listed in requirements.txt 


## Conclusion

I apply three different models for this project. Among the three, the method I believe in most is multi-stage progressive finetuning method. While the pipeline using Computer Vision performs gave pretty similar results, its performance isn't very great on images where the contrast betweeen barnacles and background is low. Similarly, while the single-stage fine-tuned model performed pretty well, I believe it is grossly under-trained. The multi-stage post processing approach is not only more robust in my opinion, it is also a more transferrable approach to other situations where one might have a limited dataset. By sequentually training a model on tasks similar to what it is supposed to do before fine-tuning it on yout custom dataset, I believe there can be significant boosts to the performance. 

## Learning Process

Overall, I had a great learning experience through this challenge, and I would like to thank the DALI team for that. I not only got to play around and work with SOTA models, but I also got significant exposure and training with a variety of Classical Computer Vision algorithms and techniques. This entire journey has been a steep learning curve and I have enjoyed every bit of it. I hope to be able to utilize the skills I have learned through this challenge in the future, in DALI and beyond.


## Author
Arses Prasai -[Github](https://github.com/arses-ui)         
Email : arses.prasai.28@dartmouth.edu                 

