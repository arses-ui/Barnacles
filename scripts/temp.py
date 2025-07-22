import numpy as np
import cv2
from matplotlib import pyplot as plt 
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt
from PIL import Image
import matplotlib.image as mpimg
from IPython.display import  display




def image_processing(image_address, THRESHOLD): 

    image = Image.open("sample_image.png").convert("RGB")
    grayscale_image = image.convert("L")
    image_array = np.array(grayscale_image)
    gaussian_filtered_image = gaussian_filter(image_array, sigma=1)



    otsu_threshold, image_after_otsu = cv2.threshold(
    gaussian_filtered_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU, 
    )
   


    plt.axis('off')


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

    plt.tight_layout() # Adjust layout to prevent titles/labels from overlapping
    plt.axis('off')


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
    image = cv2.drawContours(image_array, coins, -1, color=(0, 23, 223), thickness=2)
    plt.imshow(image, cmap ='grey')
    print(f"The number of barnacles with {THRESHOLD} is {len(coins)}")

    return  



threshold_list = [0.31, 0.32, 0.33, 0.34, 0.35, 0.36, 0.37, 0.37, 0.38, 0.39, 0.40 ]

for i in threshold_list: 
    image_processing(1, i)