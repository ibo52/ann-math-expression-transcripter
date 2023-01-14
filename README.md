# ann-math-expression-transcripter
## CNN model to predict simple mathematical expressions. (ubuntu 20.04)
---
#### Program provides best accuracy on digitally handwritten images. On manually handwritten images, there should be __jet-black, thick writings__ over white paper. __on a stable lightning environment__. If environment had bad lightning, Contouring process will be broke and can not extract possible symbols area properly, which will cause meaningless predictions.
---
# Running
  - ___Run `sh reqs_python.sh` to install required libraries first. Or else program will not work due to library dependency issues.___
  - I uploaded the trained model, so you can see results yourself by downloading this page source.
  - Run `test_main.py` file to see results of my digitally written image samples. It will show predictions on screen.
  - Run `camera_main.py` to test your real-time inputs. Use __black and thick__ writings on your paper. Also you can see my sample video for this process -> ![laptop camera realtime prediction](https://github.com/ibo52/ann-math-expression-transcripter/blob/main/realtime%20test%20outputs/Kazam_screencast_00001.webm)
---
---
## Data set
  - place your numeric and operators dataset to path : 'dataset/train'
  - Since i could not find brackets image, i prepared my own brackets data set.
    1. `augment-brackets.py` file augments my left-bracket image-set and expands image size and also creates right bracket images by flipping expanded images. Place the left-bracket folder and py file to dataset/train and run py file.

## train-model.py
  - contains the CNN model specifications for training. Run to build transCripter.h5 file. Architecture inherited from LeNet-5 represents as follows:

    * 4 stacks as:
      Convolution→Activation(‘relu’)→Batch Normalization→MaxPooling
    * Final stack as:
	  →Flatten→Dropout→FullyConnected→ FullyConnected→ FullyConnected→Result

## test-main.py
  - the testing module that takes sample images from 'dataset/test' and predicts the symbols of that inputs.

## camera-main.py
  - realtime prediction module. takes input from camera and tries to predict possible symbols.
---
---
# Sample Testing Results
## some of test images prepared are digitally handwritten on GIMP, some are images of manually written, some are manually enhanced(sharpening etc.) images of those manually written and some are the images provided from laptop camera,which is not pre-processed.
---
## digitally handwritten images
### we estimate best predictions on digitally written images since these are constant contrast as black writings over white canvas
![digitally handwritten1](https://github.com/ibo52/ann-math-expression-transcripter/blob/main/realtime%20test%20outputs/out6-digital.png)

![digitally handwritten2](https://github.com/ibo52/ann-math-expression-transcripter/blob/main/realtime%20test%20outputs/out5-digital.png)

![digitally handwritten3](https://github.com/ibo52/ann-math-expression-transcripter/blob/main/realtime%20test%20outputs/out3.png)

## manually handwritten images
### some symbols may not predict proper due to bad lightning(number 9 predicted as 3; / predicted as 1 because rotation augmentation on early implementations)
![manually handwritten1](https://github.com/ibo52/ann-math-expression-transcripter/blob/main/realtime%20test%20outputs/better%20manuel%20writing.png)
##another bad prediction: lightning causes improper extracting of contour areas
![manually handwritten2](https://github.com/ibo52/ann-math-expression-transcripter/blob/main/realtime%20test%20outputs/contrast%20issues.png)

## laptop camera real time prediction
### mostly accurate results, but rapid broke on prediction due to lightning issues
![camera realtime](https://github.com/ibo52/ann-math-expression-transcripter/blob/main/realtime%20test%20outputs/manual5.png)

![camera realtime](https://github.com/ibo52/ann-math-expression-transcripter/blob/main/realtime%20test%20outputs/manual6.png)

![camera realtime](https://github.com/ibo52/ann-math-expression-transcripter/blob/main/realtime%20test%20outputs/manual4.png)

# watch video of my real-time inputs results
![camera realtime](https://github.com/ibo52/ann-math-expression-transcripter/blob/main/realtime%20test%20outputs/Kazam_screencast_00001.webm)
