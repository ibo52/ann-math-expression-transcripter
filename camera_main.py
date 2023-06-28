import numpy as np
import matplotlib.pyplot as plt
import cv2                     
from keras.models import load_model 
import numpy as np
import imutils
from imutils.contours import sort_contours
try:
    from labels import labels_#labels of model
except Exception as e:
    print(e)
    print("train-model.py generates this file")
    print("you have to run train-model")
    print("or give labels for classes manually" )
    
img_rows, img_cols =64, 64

model =  load_model('transCripterModel.h5')
labels=labels_

MODEL_INPUT_SIZE= img_rows, img_cols


#give image to function to decipher formula
def digitize_math_expression(frame, model):
    global MODEL_INPUT_SIZE

    #preprocessing input
    img = frame

    #no need to color image to process
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    #padding grayed image to make rectangle input
    img_w,img_h= img.shape[1],img.shape[0]
    if img_w>img_h:
        
        diff=(img_w-img_h)//2#top and bottom padsize(+10 fır roi_w)
        img_gray = cv2.copyMakeBorder(img_gray,diff,diff,0,0,cv2.BORDER_CONSTANT,None,value=255)
        img = cv2.copyMakeBorder(img,diff,diff,0,0,cv2.BORDER_CONSTANT,None,value=255)

    elif img_h>img_w:
                
        diff=(img_h-img_w)//2#left and right padsize(+10 for roi_h)
        img_gray = cv2.copyMakeBorder(img_gray,0,0,diff,diff,cv2.BORDER_CONSTANT,None,value=255)
        img = cv2.copyMakeBorder(img,0,0,diff,diff,cv2.BORDER_CONSTANT,None,value=255)
        
    else:
        pass

    #resize after padding to preserve shapes in image(avoid strecthing problem)
    img = cv2.resize(img, (MODEL_INPUT_SIZE[0]*13, MODEL_INPUT_SIZE[1]*13))

    '''
    ########################
    rectx,recty=img.shape[0]//2 - 200, img.shape[1]//2 - 200
    rectx_w, recty_h= img.shape[0] - 200 , recty+300
    
    cv2.rectangle(img, (rectx, recty), (rectx_w, recty_h), (255, 96, 255), 2)#to show area over main image
    cv2.putText(img, "pose your expression to that area" , (rectx, recty), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 96, 255),2)
    ########################################
    img_gray=img_gray[recty:recty_h ,rectx:rectx_w]
    '''
    
    img_gray = cv2.resize(img_gray, (MODEL_INPUT_SIZE[0]*13, MODEL_INPUT_SIZE[1]*13))
    # blurred = cv2.GaussianBlur(img_gray, (3, 3), 0)
    
    #find edges in image to find contours(to determine boundaries of symbol)
    edged = cv2.Canny(img_gray, 30, 150)
    
    contours = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(contours)

    # order detected symbols left-to-right to write mathematical expression properly
    contours = sort_contours(contours, method="left-to-right")[0][:16]

    text=''#deciphered math expression

    for c in contours:
        #founded symbols area
        (x, y, w, h) = cv2.boundingRect(c)

        
        if 40<=w and 10<=h  and 200>w and 250 >h:

            #image sub-area of founded symbol
            roi = img_gray[y:y+h, x:x+w]#detected area of symbol by contouring

            #binarize image
            T, roi =cv2.threshold(roi,0,255,cv2.THRESH_OTSU)
            
            #pad cause wrong prediciton
            #determine padsize to conservate shape
            #or else for example constant padding stretches 1
            #and cause model to predict as 7 or anything
            roi_w,roi_h= roi.shape[1],roi.shape[0]
            if roi_w>roi_h:
                
                diff=(6+roi_w-roi_h)//2#top and bottom padsize(+10 fır roi_w)
                roi=cv2.copyMakeBorder(roi,diff,diff,5,5,cv2.BORDER_CONSTANT,None,value=255)

            elif roi_h>roi_w:
                
                diff=(6+roi_h-roi_w)//2#left and right padsize(+10 for roi_h)
                roi=cv2.copyMakeBorder(roi,5,5,diff,diff,cv2.BORDER_CONSTANT,None,value=255)
            else:
                pass


            #process image to input to model.Model expects (1,32,32,1) as input

            #by blurring, we obtain diffusion of values to conservate
            #inputs when resizig image
            #BLUR DİSTORTS REAL İMAGE!! DO NOT APPLY or apply little kernel
            #works on digital images
            #roi=cv2.GaussianBlur(roi,(7,7),0)
            #normalize to give to model
            roi=cv2.resize(roi,MODEL_INPUT_SIZE)/255.
            """
            plt.imshow(roi)
            plt.show()
            """
            
            #roi=cv2.GaussianBlur(roi,(3,3),0)
            """
            plt.imshow(roi)
            plt.show()
            
            cv2.imshow("contours",roi)
            cv2.waitKey(0)
            """
            shape4D=(-1, MODEL_INPUT_SIZE[0], MODEL_INPUT_SIZE[1], 1)
            roi=np.reshape(roi, shape4D)
            
            #predict symbol
            val=model.predict(roi,verbose=0).argmax()
            """
            print("prediction:", labels[val.argmax()])
            
            input("devam: ")
            """
            text=text+labels[val]
            cv2.rectangle(img, (x, y), (x+w, y+h), (0, 0, 255), 2)#to show area over main image
            cv2.putText(img, labels[val] , (x-5, y), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255),2)

    print("predicted :",text,end="=")
    try:
        print(eval(text))
    except Exception:
        print(" Could not solved. possibly bad prediciton")

        

    return img

cap=cv2.VideoCapture(0)
while True:
    ret,frame=cap.read()#get image frame from camera
    
    frame=digitize_math_expression(frame, model)

    cv2.imshow("output",frame)

    if cv2.waitKey(1) & 0xFF == ord('q'): 
        break

#run camera---------------------------
cap.release()
cv2.destroyAllWindows()
