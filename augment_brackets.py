import os
import cv2
from random import randint
import PIL
from PIL import Image
from datetime import datetime as dt
#
#
LEFT_BRACKET_PATH= os.path.join(os.getcwd(), "left-bracket")
RIGHT_BRACKET_PATH= os.path.join(os.getcwd(), "right-bracket")
#
#
#
def klasör_oluştur(yol):
    """belirtilen yolda klasör oluşturur"""
    cwd=os.getcwd()
    if not os.path.exists( os.path.join(cwd,yol) ):
        os.makedirs( os.path.join(cwd,yol) )
        print("created directory:",yol)
    else:
        print("directory already exits:",yol,)


def save_image(image, save_dir, file_format=".jpg"):
    image.save( os.path.join( save_dir, "AUGMENT_"+str(dt.now())+ file_format ))


def augment_n_save(img_path, rot_range=0, flip_axis=0):
    global RIGHT_BRACKET_PATH
    global LEFT_BRACKET_PATH
    """augment given image path
    let flip axis remain 0(x axis) for brackets"""
    PARENT_DIR = os.path.relpath(  os.path.join(img_path,os.pardir)  )
    i=Image.open(img_path)

    angle=0;angle2=0
    if rot_range>0:
        while angle==0:
            angle=randint(-rot_range, rot_range)

        while angle2==0:
            angle2=randint(-rot_range, rot_range)
            
        #print("selected angle:",angle,angle2)


    #augmetation1: rotate
    i=i.rotate(angle,PIL.Image.NEAREST,expand=False,fillcolor="white")
    save_image(i,PARENT_DIR)#save rotated image
    
    #augmetation2: flip (axis: 0==x, 1==y)
    axis=(Image.FLIP_TOP_BOTTOM if flip_axis==0 else Image.FLIP_LEFT_RIGHT)
    i=i.transpose(axis)
    save_image(i,PARENT_DIR)#save flipped image

    #augmetation3: rot n flip
    i=i.transpose(axis)
    i=i.rotate(angle2,PIL.Image.NEAREST,expand=False,fillcolor="white")
    save_image(i,PARENT_DIR)#save flipped image
    
    
def augment_from_path(read_path):
    """create right bracket image
    by flipping left bracket images
    from directory"""
    klasör_oluştur("right-bracket")
    
    counter_aug=0
    print("augmenting left brackets")
    for root_dir,sub_dir,files in os.walk(read_path,topdown=False):
        print(f"{len(files)} images found on directory")
        
        for img in files:
            img_path= os.path.join(root_dir,img)
            
            #augmenting left brackets
            augment_n_save( img_path ,rot_range=15 )
            counter_aug+=1

    print(f"totally {counter_aug} augmention images saved")
    #
    #
    #
    
    counter=0
    print("---------------------------\nflipping to create right bracket images")
    #right brackets by flipping left bracket images
    for root_dir,sub_dir,files in os.walk(read_path,topdown=False):
        print(f"{len(files)} images found on directory")

        for img in files:
            img_path= os.path.join(root_dir,img)

            i=cv2.imread( img_path )
            flipped=cv2.flip( i ,1)#flip over y axis

            cv2.imwrite( os.path.join(RIGHT_BRACKET_PATH,str(dt.now())+".jpg"), flipped)
            """
            cv2.imshow("flip",flipped)
            cv2.imshow("org",i)
            cv2.waitKey(0)
            """
            counter+=1
    print(f"totally {counter} images flipped to create right-brackets")
    
augment_from_path(LEFT_BRACKET_PATH)
exit(0)
