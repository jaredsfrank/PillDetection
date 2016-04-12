import cv2
import numpy as np
import os
from matplotlib import pyplot as plt
from scipy import stats

def clip(img, top,bottom,left,right):
    return(img[top:bottom,left:right]) 

def findTop(mask):
    h, w = mask.shape
    j = 0
    while j < h and np.sum(mask[j]) == 0 :
        j+=1
    return j if j != h else 0

def findBounding(mask):
    h, w = mask.shape
    top = findTop(mask)
    mask = np.rot90(mask)
    right = findTop(mask)
    mask = np.rot90(mask)
    bottom = findTop(mask)
    mask = np.rot90(mask)
    left = findTop(mask)
    mask = np.rot90(mask)
    return(top, h-bottom, left, w-right)


def makeMask(img, extraStrength, showSteps):
    imgH, imgW, chan = img.shape

    print(extraStrength)
    #Edge Detection
    p1 = 30 if extraStrength else 90
    p2 = 40 if extraStrength else 100
    edges = cv2.Canny(img,p1,p2)

    #Filling in Holes in edges
    iters = 3 if extraStrength else 1
    maskSize = 3 if extraStrength else 1
    kernel = np.ones((maskSize,maskSize),np.uint8)
    edges = cv2.dilate(edges,kernel,iterations = iters)
    edges = cv2.erode(edges,kernel,iterations = iters)
    filled = edges==255

    if showSteps:
       plt.imshow(cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)), plt.colorbar(), plt.show()

    #Flooding area outside edges
    mask = edges.copy()
    floodMask = np.zeros((imgH+2, imgW+2), np.uint8)
    cv2.floodFill(mask, floodMask, (0,0), 255);
    mask = np.logical_not(mask/255).astype('uint8')


    #Filling back in original edges
    mask = mask | filled

    #Erroding porbable noise
    #plt.imshow(cv2.cvtColor(mask*255, cv2.COLOR_GRAY2RGB)), plt.colorbar(), plt.show()
    k2 = np.ones((3,3),np.uint8)
    mask = cv2.erode(mask,k2,iterations = 1)
    mask2 = clip(mask, *findBounding(mask))
    #plt.imshow(cv2.cvtColor(mask2*255, cv2.COLOR_GRAY2RGB)), plt.colorbar(), plt.show()

    percentage =  float(np.sum(mask2))/mask2.ravel().shape[0]
    print(percentage)
    wellDefined = percentage > .7
    return (mask, wellDefined)



def edgeStuff(imgName, showSteps):
    img = cv2.imread('TrainingSet/{}'.format(imgName))
    mask,good = makeMask(img, False, showSteps)
    if not good:
        mask,_  = makeMask(img, True, showSteps)
    bounding = findBounding(mask)
    if showSteps:
       plt.imshow(cv2.cvtColor(mask*255, cv2.COLOR_GRAY2RGB)), plt.colorbar(), plt.show()
    #Apply mask to original image
    isolatedPill = clip((img*mask[:,:,np.newaxis]).astype('uint8'),*bounding)
    plt.imshow(cv2.cvtColor(isolatedPill, cv2.COLOR_BGR2RGB)), plt.colorbar(), plt.show()



for root, dirs, filenames in os.walk('TrainingSet'):
    for i in range(5):
        edgeStuff(filenames[i], True)
        

