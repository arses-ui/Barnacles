# Barnacle Detection with Downstream Post-Training
## Introduction

In this project, I explore how to apply state-of-the-art (SOTA) models to the task of barnacle detection and counting. I approached the problem using two main strategies:

1. **Single-Step Fine-Tuning:** Train a model using only the two images and their segmentation masks provided.
2. **Progressive Fine-Tuning:** Gradually fine-tune a model across multiple datasets, ending with the provided barnacle data.

## Approaches

### 1. Single-Step Fine-Tuning

I trained a YOLOv8 model on a dataset I created by cropping the two original images into 124 overlapping patches of fixed size (256×256) using  Python. The model is trained for detection task. 

- Used [FiftyOne](https://voxel51.com/fiftyone/) and to create and handle datasets into.
- Converted the dataset into the required YOLO format.
- Ran 5-fold cross-validation.
- Used model predictions from each fold to generate averaged outputs.

### 2. Progressive Fine-Tuning

While YOLOv8 excels at object detection via bounding boxes, its smaller variants are not as effective at segmentation. However, due to time and resource constrainst, I wasn't able to use the larger models. So, to improve performance, I used the medium version of YOLOv8 and fine-tuned it progressively:

- **Stage 1:** Trained on a large building segmentation dataset.
- **Stage 2:** Trained on a smaller oil palm segmentation dataset.
- **Final Stage:** Fine-tuned on the cropped barnacle patches.


## Why YOLO

YOLO (You Only Look Once) is a State of the Art algorithms for object detection as well as localization and segmentation. I chose YOLOv8 as my model of choice, which runs using this algorithm, for the following reasons:

1. **Lightweight:** Smaller models (~12M parameters) enable faster training and inference.
2. **Versatile:** Supports both object detection and segmentation.
3. **Integrated Ecosystem:** Ultralytics makes it use to work with libraries like SAHI and various data formats.

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

## File Structure

The files for this project have been refactored into different directories for better organization. The contents of the files are as follows: 

```
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

## Learning Process

Although I’ve previously worked with large vision-language models, this project was my first time training models for pixel-level segmentation tasks. It gave me the opportunity to:

- Learn how to structure and convert datasets (e.g., fiftyone -> COCO -> YOLO).
- Explore SOTA frameworks and experiment with different training strategies.
- Improve my understanding of classical and modern computer vision techniques, including semantic vs. instance segmentation.
- Get exposed to and work with useful tools like YOLOv8, FiftyOne, and SAHI.

It took tens of hours of trial, error, and iteration, but ultimately became a very fulfilling learning experience.


## Limitations

While results were promising, there were several limitations:

1. Limited Data: The dataset (just two images) is far too small for deep learning. Even with augmentation and patching, it constrains generalization.
2. Model Constraints: I used small/medium YOLO variants and trained them for only 20 epochs due to time constraints. Larger models or longer training would likely improve results.

## Future Work

Assuming this is an actual project, the following points can be considered for furture work.

1. Curate a Larger Dataset: Collect more labeled barnacle images under varying conditions.
2. Model Scaling: Use larger pre-trained YOLO or Mask R-CNN variants and fine-tune extensively.
## Author
Arses Prasai -[Github](https://github.com/arses-ui)         
Email : arses.prasai.28@dartmouth.edu                 

