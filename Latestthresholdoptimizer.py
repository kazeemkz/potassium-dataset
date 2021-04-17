# -*- coding: utf-8 -*-
"""
Created on Thu Sep 19 10:59:07 2019

@author: koes
"""

# -*- coding: utf-8 -*-

import os
import cv2
import numpy as np
from tqdm import tqdm




from skimage.transform import resize
from keras.models import Model, load_model
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img, save_img
import warnings
import skimage.filters as flt
import scipy.ndimage.filters as flt
import cv2
import os


def thresholdingOptimizer(gtfolder,segmentedfolder,steps,channnels,deeplearningmodel,modelsshape,workspace):

    answer = []
    images = []
    accuracy = []
    ids = next(os.walk(segmentedfolder))[2]
    counter = 1

    for n, id_ in tqdm(enumerate(ids), total=len(ids)):
        initialvalue = -200
        stopper = 0
        images.append(id_)
        print(id_)

        segmented = os.path.join(segmentedfolder,id_)
        print(segmented)
        if (channnels == 3):
             image = cv2.imread(segmented)
        else:
             image = cv2.imread(segmented,0)
        #image = anisodiff((image))
        #image = histogram_equalization(image) is was enabled for white blood cell images

        gt = os.path.join(gtfolder,id_)
        print(gt)
        imagegt = cv2.imread(gt, 0)

        model = load_model(deeplearningmodel)

        currentth = 0
        for xx in np.arange(0, 1, steps):
            tp = 0
            tn = 0
            fn = 0
            fp = 0
            x_img = img_to_array(image)
            if (channnels ==3):
                x_img = resize(x_img, (modelsshape, modelsshape, channnels), mode='constant', preserve_range=True)
            else:
                x_img = resize(x_img, (modelsshape, modelsshape, channnels), mode='constant', preserve_range=True)
            x_img = x_img / 255
            x = np.expand_dims(x_img, axis=0)
            x = np.array(x)

            predict = model.predict(x)
            thres =  xx
            print(xx)
            predict = (predict > (xx)).astype(np.uint8)
            img_array = img_to_array(predict[0])
            img_array = resize(img_array, (imagegt.shape[0], imagegt.shape[1], 1), mode='constant', preserve_range=True)
            thesavedpath =os.path.join(workspace,str(counter)+ '.png')
            save_img(thesavedpath, img_array)
            seg = cv2.imread(thesavedpath,0)

            #seg = seg/255

            img_arraygt = resize(imagegt, (imagegt.shape[0],imagegt.shape[1], 1), mode='constant', preserve_range=True)
            theimage = np.array(img_array)
            img_arraygt = np.array(img_arraygt)
            c = 0
            for hex in range(0,imagegt.shape[0],1):
                for way in range(0,imagegt.shape[1],1):
                    if img_arraygt[hex][way] >= 0.1 and theimage[hex][way] >= 0.1:
                        tp = tp +1
                    if img_arraygt[hex][way] < 0.1 and theimage[hex][way] < 0.1:
                        tn = tn + 1
                    if img_arraygt[hex][way] < 0.1 and theimage[hex][way] >= 0.1:
                        fp = fp + 1
                    if img_arraygt[hex][way] >= 0.1 and theimage[hex][way] < 0.1:
                        fn = fn + 1
            ac = (tp) / ( tp + fn + fp)
            print(ac)
            if(ac > initialvalue):
                initialvalue = ac
                currentth= thres
            else:
                stopper  = stopper + 1
            if (stopper ==1):
                break
        accuracy.append(initialvalue)
        answer.append(currentth)
        print('accuracy  is' + str(currentth))
        counter= counter+1
    finalvalue =sum(answer)/len(answer)
    print('final answer is' +str(finalvalue))
    print(answer)
    print(images)
    print(accuracy)

# example of how it is called
#ws ='C:\\Users\\t\\Desktop\\optimizing deep learning models segmentation outputs\\ws'
#gt ='C:\\Users\\t\\Desktop\\optimizing deep learning models segmentation outputs\\gtdt1new'
#seg ='C:\\Users\\t\\Desktop\\optimizing deep learning models segmentation outputs\\dt1'
#thresholdingOptimizer(gt,seg,0.1,3,'upwhitebloodcellgrayscalebwlatest.h5',128,ws)


