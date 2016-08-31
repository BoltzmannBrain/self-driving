## Prediction models

The speed prediction models here use Keras with TensorFlow backend, and make use of some OpenCV tools.

### Speed prediction with CNN+LSTM

This is to implement some of the architectures described in ["Beyond Short Snippets: Deep Networks for Video Classification"](http://arxiv.org/abs/1503.08909) in an effort to incorporate temporal information as an improvement over the static CNN model below.

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


#### Future work

There are a number of suggestions for improving this model, some listed below, but in the interest of time I did not pursue them extensively (I leave these for future work):

* Tune the optmizer learning rate and other params (currently using Keras standards). A good approach would be to decrease the learning rate as a function of epochs.
* More and/or larger layers, perhaps like the ["End-to-end Self-driving" model](https://arxiv.org/abs/1604.07316). Larger layers would help, considering the relatively large input frames (480x640).
* Crop the input stream to only the visual areas significant for vehicle speed prediction.


