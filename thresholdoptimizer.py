
import numpy as np
from tqdm import tqdm
from skimage.transform import resize
from keras.models import Model, load_model
from keras.preprocessing.image import img_to_array, save_img
import cv2
import os


def thresholdingOptimizer(gtfolder,segmentedfolder,steps,channnels,deeplearningmodel,modelsshape,workspace):

    answer = []
    accuracy = []
    ids = next(os.walk(segmentedfolder))[2]
    counter = 1
    for n, id_ in tqdm(enumerate(ids), total=len(ids)):
        initialvalue = -200

        print(id_)
        segmented = os.path.join(segmentedfolder,id_)
        print(segmented)
        if (channnels == 3):
             image = cv2.imread(segmented)
        else:
             image = cv2.imread(segmented,0)

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

            seg = seg/255

            img_arraygt = resize(imagegt, (imagegt.shape[0],imagegt.shape[1], 1), mode='constant', preserve_range=True)
            #print(imagegt.shape[0])
            theimage = np.array(img_array)
            #theimage = np.array(img_array)
            img_arraygt = np.array(img_arraygt)
            c = 0
            for hex in range(0,imagegt.shape[0],1):
                for way in range(0,imagegt.shape[1],1):
                    #print(img_arraygt[hex][way])
                    if img_arraygt[hex][way] >= 0.8 and theimage[hex][way] >= 0.8:
                        tp = tp +1
                    if img_arraygt[hex][way] <= 0.1 and theimage[hex][way] <= 0.1:
                        tn = tn + 1
                    if img_arraygt[hex][way] <= 0.1 and theimage[hex][way] >= 0.8:
                        fp = fp + 1
                    if img_arraygt[hex][way] >= 0.8 and theimage[hex][way] <= 0.1:
                        fn = fn + 1
            ac = (tp) / ( tp + fn + fp)
            print(ac)
            if(ac > initialvalue):
                initialvalue = ac
                currentth= thres
            #print(ac)
        #accuracy.append(initialvalue)
        answer.append(currentth)
        print('accuracy  is' + str(currentth))
        counter= counter+1
    finalvalue =sum(answer)/len(answer)
    print('final answer is' +str(finalvalue))


# example of how it is called
ws ='C:\\Users\\kazeem\\Desktop\\PAN\\personal paper research\\optimizing deep learning models segmentation outputs\\ws'
gt ='C:\\Users\\kazeem\\Desktop\\PAN\\personal paper research\\optimizing deep learning models segmentation outputs\\gtdt2new'
seg ='C:\\Users\\kazeem\\Desktop\\PAN\\personal paper research\\optimizing deep learning models segmentation outputs\\dt2'
thresholdingOptimizer(gt,seg,0.1,3,'whitebloodcellgrayscalebwlatest.h5',128,ws)


