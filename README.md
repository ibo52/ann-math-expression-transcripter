# ann-math-expression-transcripter
CNN model to predict simple mathematical expressions.
---
Program provides best accuracy on digitally handwritten images. On manually handwritten images, there should be jet-black, sharp writings over white paper. __on a stable lightning environment__. If environment had bad lightning, Contouring process will be broke and can not extract possible symbols area properly, which will cause meaningless predictions.
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
## train-model.py
  - contains the CNN model specifications for training. Run to build transCripter.h5 file. Architecture inherited from LeNet-5 represents as follows:

    * 4 stacks as:
      Convolution→Activation(‘relu’)→Batch Normalization→MaxPooling
    * Final stack as:
	  →Flatten→Dropout→FullyConnected→ FullyConnected→ FullyConnected→Result
***
## test-main.py
  - the testing module that takes sample images from 'dataset/test' and predicts the symbols of that inputs.
***
## camera-main.py
  - realtime prediction module. takes input from camera and tries to predict possible symbols.

