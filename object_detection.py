import glob
import pathlib

import cv2
from math import *
from matplotlib import pyplot as plt
import numpy as np
import time


def readImgAndMaskPaths():
    """
    Reads, orders and returns lists of absolute paths (strings) to the images and masks. 
    Also returns an absolute path to a reference image (which can be used for feature matching).
    Outputs
    sorted_img_paths    list    List of paths to the image, sorted on image number
    sorted_mask_paths   list    List of paths to the masks, sorted on mask number
    """
    imgs = []
    masks = []
    dir = str(pathlib.Path(__file__).parent.absolute())
    img_dir = dir + "\WashingtonOBRace\*.png"
    image_paths = glob.glob(img_dir)
    
    if len(image_paths)< 1:
        raise Exception("Images not found")

    for path in image_paths:
        if 'img_' in path:
            imgs.append(path)
        if 'mask_' in path:
            masks.append(path)

    # Sort lists
    img_nums = []
    for path in imgs:
        image_name = path.replace(img_dir[:len(img_dir)-5],'').replace('img_','') #  Remove directory and img_ from path to leave only image names (e.g. 1.png)
        img_nums.append(image_name[:len(image_name)-4]) # Remove extension and append
        for i in range(len(img_nums)):
            img_nums[i] = int(img_nums[i])
        img_nums = sorted(img_nums) #Sort image numbers from small to large

    #Reconstruct list
    sorted_img_paths = []
    for i in range(len(img_nums)):
        sorted_img_paths.append(img_dir[:len(img_dir)-5]+'img_'+str(img_nums[i])+'.png')

    # And now for the masks
    mask_nums = []
    for path in masks:
        mask_name = path.replace(img_dir[:len(img_dir)-5],'').replace('mask_','') #  Remove directory and img_ from path to leave only image names (e.g. 1.png)
        mask_nums.append(mask_name[:len(mask_name)-4]) # Remove extension and append
        for i in range(len(mask_nums)):
            mask_nums[i] = int(mask_nums[i])
        mask_nums = sorted(mask_nums) #Sort image numbers from small to large

    #Reconstruct list
    sorted_mask_paths = []
    for i in range(len(mask_nums)):
        sorted_mask_paths.append(img_dir[:len(img_dir)-5]+'mask_'+str(mask_nums[i])+'.png')    

    return sorted_img_paths, sorted_mask_paths

def harrisCornerConvolution(img_path,threshold_Harris,threshold_box,erosion_dilation):
    """
    Uses Harris Corner Detection followed by convolutional filters to create a binary image of the location of the gates
    Inputs
    img_paths           str     Path to the image
    threshold_Harris    float   Threshold value for which pixel is marked as corner
    threshold_box       float   Fraction (0-1) of pixels required white in box filter
    erosion_dilation    list    List of 2 values, size of erosion kernel and size of dilation kernel

    Outputs
    prediction_bin      array   Image with predicted mask
    frame_runtime       float   Time in seconds frame took to be processed
    """
    start_time=time.time() # Record start time to calculate computational time for frame

    # Read image
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    gray = np.float32(gray)

    # Step 1: Apply Harris corner detection and threshold
    dst = cv2.cornerHarris(gray,2,3,0.04)
    retval, dst = cv2.threshold(dst,thresh = threshold_Harris*dst.max(),maxval = 255,type = cv2.THRESH_BINARY)
    dst = dst.astype(np.uint8) # Convert to 8 bit format

    # Step 2: Apply box filter and threshold (looks to find at least 1/kernel_size corners within a kernel, filters noise)
    kernel_size = 7
    kernel = np.ones((kernel_size,kernel_size))/kernel_size ** 2 
    fdst = cv2.filter2D(dst, -1, kernel)
    retval, fdst = cv2.threshold(fdst,thresh = threshold_box*255,maxval = 255,type = cv2.THRESH_BINARY)


    # Step 3: Erosion - Dilation step for further noise filtering
    eroded = cv2.erode(fdst,np.ones((erosion_dilation[0],erosion_dilation[0])), iterations = 1)
    prediction_bin = cv2.dilate(eroded,np.ones((erosion_dilation[1],erosion_dilation[1])),iterations = 1)

    # Uncomment below if you want to see a lot of images
    #cv2.imshow('predicted mask', prediction_bin)
    #cv2.imshow('Corner detection',dst)
    #cv2.imshow('Eroded', eroded)
    #cv2.imshow('Filtered corner detection',fdst)
    #if (cv2.waitKey(20) & 0xff) == ord('q'):
    #    pass

    frame_runtime = time.time() - start_time
    return prediction_bin, frame_runtime

def createConfusionMatrix(prediction, label_path):
    """
    Generate confusion matrix and IoU of the prediction - label pair
    True Positives (TP): Number of pixels that was positive correctly.
    False Positives (FP): Number of pixels that was positive incorrectly.
    False Negatives (FN): Number of pixels that was negative incorrectly.
    True Negatives (TN): Number of pixels that was negative correctly.
    Inputs
    prediction      array   Predicted binary image
    label_path      str     Path to the binary image label
    Outputs
    iou             float   Intersection over union
    TP              int     True positive pixels
    FP              int     False positive pixels
    FN              int     False negative pixels
    TN              int     True negative pixels
    """
    #Load label
    label = cv2.imread(label_path,cv2.IMREAD_GRAYSCALE)
    
    # Calculate True Positive pixels
    intersection = cv2.countNonZero(cv2.bitwise_and(label,prediction))
    union = cv2.countNonZero(cv2.bitwise_or(label, prediction))
    TP = intersection

    # Calculate False Positive pixels
    pred_area = cv2.countNonZero(prediction)
    FP = pred_area - intersection

    # Calculate False Negative pixels   
    label_area = cv2.countNonZero(label)
    FN = label_area - intersection

    # Calculate True Negative pixels
    TN = cv2.countNonZero(cv2.bitwise_and(cv2.bitwise_not(label),cv2.bitwise_not(prediction)))

    # Calculate IoU
    iou = intersection / union

    return iou, TP, FP, FN, TN

def drawROC(confusion_matrix):
    """
    Calculate ratios and draw ROC curves for given confusion matrix.
    Inputs
    confusion_matrix    array   matrix containing all iou, TP, TN, FP, FN values
    """
    # Color list to pick colours from for different ROC curves
    color_list = ['b', 'g', 'r', 'y', 'c', 'k', 'm', 'w']

    # Create empty arrays to store TPR and FPR
    TPR_list = np.empty((len(confusion_matrix),len(confusion_matrix[0])))
    FPR_list = np.empty((len(confusion_matrix),len(confusion_matrix[0])))

    # Determine ratios
    for n in range(len(confusion_matrix)):
        for i in range(len(confusion_matrix[0])):
            TP = sum(confusion_matrix[n][i][1])
            FP = sum(confusion_matrix[n][i][2])
            FN = sum(confusion_matrix[n][i][3])
            TN = sum(confusion_matrix[n][i][4])
            
            TPR_list[n][i]= (TP/(TP + FN))
            FPR_list[n][i]=(FP/(FP + TN))
        # Plot ROC curve
        plt.plot(FPR_list[n],TPR_list[n],('-o'+color_list[n]),label=("Line " + str(n)))

    # Generate random classifier curve (TPR = FPR) in red dashed
    random_classifier = np.arange(0,1,0.1) # For drawing TPR = FPR line
    plt.plot(random_classifier,random_classifier, 'r--',label='Random')

    plt.xlabel('False positive rate (FPR)')
    plt.ylabel('True positive rate (TPR)')
    plt.legend()
    plt.show()
    # TO DO: Let it show area under curve
    return

def printBestmIoU(confusion_matrix):
    """
    Takes in confusion matrix and prints out the best mean Intersection over Union along with the indexes
    of the parameters and threshold that it belongs to.
    Inputs
    confusion_matrix    array   matrix containing all iou, TP, TN, FP, FN values
    Outputs
    best_miou   float   Best mIoU score found in all runs
    best_m      int     Index of the best configuration 
    best_n      int     Index of the best threshold value
    """
    best_miou = 0.
    best_m = 0
    best_n = 0
    miou_list = []

    # Iterate over the parameter sets and threshold values
    for m in range(len(confusion_matrix)):
        for n in range(len(confusion_matrix[0])):
            mean_iou = sum(confusion_matrix[m][n][0])/len(confusion_matrix[0][0][0]) # Determine mIoU
            miou_list.append(mean_iou)
            if mean_iou > best_miou: # Save if better than previous best mIoU
                best_miou = mean_iou
                best_m = m
                best_n = n
            else:
                pass
    
    # Display results
    print(miou_list)
    print("Best mIoU is: ", best_miou)
    print("Best params index: ", best_m)
    print("Best threshold index:", best_n)
    return best_miou, best_m, best_n


def main(video = False):
    img_paths, mask_paths = readImgAndMaskPaths()

    # Array with parameter combinations to generate different ROC curves ([erosion_kernel, dilation_kernel])
    params = np.array([[1, 20],[1, 35],[1, 50],[3, 20],[3, 35],[3, 50]])

    # List with threshold values to iterate over for ROC curve
    threshold_list = np.arange(0,1.0,0.01)

    # Create empty confusion matrix to store results
    confusion_matrix = np.empty((len(params),len(threshold_list),5,len(img_paths)))
    fps_list = []
    fps = 0

    # Iterate over the parameter sets and threshold values in the threshold list
    for m in range(len(params)):
        for n in range(len(threshold_list)):
            total_frame_time = 0.
            print('Parameter set: ', params[m])
            print('Threshold value: ', threshold_list[n])
            for i in range(len(img_paths)):
                
                # Run algorithm on frame
                pred, frame_time = harrisCornerConvolution(img_paths[i],threshold_Harris=threshold_list[n],threshold_box=0.15,erosion_dilation = params[m]) # Put in path to image, output is binary mask
                if video: # If video is true, show video
                    cv2.imshow('Camera feed', cv2.imread(img_paths[i]))
                    cv2.imshow('Mask', cv2.imread(mask_paths[i]))
                    cv2.imshow('Predicted mask',pred)
                    if (cv2.waitKey(20) & 0xff) == ord('q'):
                        break
                
                # Calculate statistics of frame
                iou, TP, FP, FN, TN = createConfusionMatrix(pred, mask_paths[i])

                # And save in the confusion matrix
                confusion_matrix[m][n][0][i] = iou
                confusion_matrix[m][n][1][i] = TP
                confusion_matrix[m][n][2][i] = FP
                confusion_matrix[m][n][3][i] = FN
                confusion_matrix[m][n][4][i] = TN

                # Calculate fps of algorithm
                total_frame_time += frame_time
                fps = (i+1)/total_frame_time
            # Save FPS of all frames of one run and print average
            fps_list.append(fps)
            print("Average FPS this run: ",fps)
    
    # Print average FPS over all runs and determine and show other results
    print("Total average FPS: ", sum(fps_list)/len(fps_list))
    printBestmIoU(confusion_matrix)
    drawROC(confusion_matrix)
    return
    
    
if __name__ == '__main__':
    start_time = time.time()
    video = False # Set to true to have video
    main(video)
    print("Done! Total runtime was %s seconds." % (time.time() - start_time))
    