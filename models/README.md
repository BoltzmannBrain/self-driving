# Prediction models

## Note this is a WIP and will be updated soon with better explanation, more models, data, etc. Check back soon!

In this directory are deep learning models for prediction tasks on self-driving car data. The models here use Keras with the TensorFlow backend.

Requires:

* [Keras](https://github.com/fchollet/keras/)
* [TensorFlow](https://www.tensorflow.org/versions/r0.10/get_started/os_setup.html)
* [OpenCV](http://opencv.org/downloads.html)



### Speed prediction with CNN+LSTM

Will be posted soon! This model incorporates temporal information as an improvement over the static CNN model below.

### Speed prediction with CNN

The aim of this project is to train a deep CNN model to learn driving speeds from a dashboard camera such that it can be used for predicting speeds in the future. Please see cnn_prediction_model.py for model details on the model and experiment design.

To train:

```
./cnn_prediction_model.py -v <path to test video> -d <path to truth data>
```

To test the model:

```
./cnn_prediction_model.py --test --skipTraining -v <path to test video> -d <path to truth data>
```

The test results will be written to "speed_test.json" and plotted, and the RMSE will be calculated.

<!--![](test_screenshot.png?raw=true "Screenshot from testing" =350x)-->

#### Future work

There are a number of suggestions for improving this model, some listed below, but in the interest of time I did not pursue them extensively (I leave these for future work):

* Tune the optmizer learning rate and other params (currently using Keras standards). A good approach would be to decrease the learning rate as a function of epochs.
* More and/or larger layers, perhaps like the ["End-to-end Self-driving" model](https://arxiv.org/abs/1604.07316). Larger layers would help, considering the relatively large input frames (480x640).
* Crop the input stream to only the visual areas significant for vehicle speed prediction.


