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
Sample dataset to use|
---|
https://www.kaggle.com/datasets/clarencezhao/handwritten-math-symbol-dataset|

  - place your numeric(0,..9) and operators(+,-,*,/) dataset to follwing path : '__dataset/train__'
  - Since i could not find brackets image, i prepared my own brackets data set.
    1. `augment-brackets.py` file augments my left-bracket image-set and expands image size and also creates right bracket images by flipping expanded images. No need for you to use this.
    - You can simply place my digitally prepared brackets.zip dataset to: __dataset/train__
***
---
## train-model.py
  - contains the CNN model specifications for training. Run to build transCripter.h5 file. Architecture inherited from LeNet-5 represents as follows:

    * 4 stacks as:
      Convolution→Activation(‘relu’)→Batch Normalization→MaxPooling
    * Final stack as:
	  →Flatten→Dropout→FullyConnected→ FullyConnected→ FullyConnected→Result
	  
	 * __Convolution__: looks for the input image and mapping appropriate feature weights to predict. There is no shrinking of input as proposed LeNet. I just reduce outputs to half on every layer.
	 * __Acvtivation__: pass through ‘relu’, since it is fairly accurate and good to learn faster, better performance while giving us non-linearity.
	 * __Batch normalization__: Its not hardly required since my network is not deep. But i think its good to be able to optimize weights by scaling outputs of Activation. This new technique can help to stabilize the training, and can increase accuracy of network. It helps to speed up training and optimize weights.
	 * __MaxPooling__: Reducing complexity of feature map  to half by this layer by choosing max of value resides in incoming kernels. So I can send this to next layer as input.

	 * __Flatten__:  Flatten all the weight maps calculated to 1-dimensional list, to send fully connected layers. Thus FC layers can select best one among features.
	 * __Dropout__: To overcome possible overfitting problem while learning, we throwing out random half of outputs. Thus we trying to avoid generalization errors by selecting outputs more or less random.
	 * __Fully Connected layers__: Choose best outputs among feature maps that represents input best, to determine which category it is represents.
	 
### Training process and Specifications
  * Data is splitted by ratio of %82 for training, %18 for validation. Data is small and we can not apply much augmentation due to misinterpretation problems(e.g: There is no much rotation augmentation since rotation of some symbols cause misinterpretations. For example if we rotate symbol ‘1’ it will be symbol ‘/’. Same for  ‘x’ and  ‘+’  cause wrong prediction).
And my model expects (64*64) as input image. The reason is my CNN is one layer deeper than inherited Le-Net 5. So on last layer it should not be to decrease 1*1 image.
  * Test set consists my handwritten images as 3 category:
    1. Digitally handwritten
    2. Manually handwritten
    3. Laptop camera data (which is real time). Every examples are tested and can be seen on last pages.
   
  * __Activation Function__: relu
    - Instead of LeNet, I used ‘relu’ activation since it increasing learning speed and architecture performance yet providing non-linearity to better predictions. But on out layer there is softmax to open gap of predictions.

  * __Loss Function__: Categorical_cross_entropy
    - Since model have to classify input from multi-categories as which category is it belongs, I used it as loss function.

  * __Regularization__: L2 as learn_rate=0.01
    - It will have better generalization by decreasing the gap between train and test loss by forcing weights to reach(but not equal) to zero by penalizing square sum of weights. L2 preserves little differences(weights) between categories.
  * __batch_size__: 32
    - Divides dataset to batches(number of samples that will be propagated through the network). Since this is standard of most CNN’s and my computer has poor performance, it is good to use higher.

  * __model_input_size__=(64,64)
    - Input Should not be smaller than this to avoid 1x1 input on last layer, since inputs will reduce to half by maxPooling at every stack.
***
---

## test-main.py
  - the testing module that takes sample images from 'dataset/test' and predicts the symbols of that inputs.
***
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
