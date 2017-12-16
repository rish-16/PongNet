# PongNet

4 layer ConvNet playing the Atari game **Pong**:

---

Added the saved JSON model architecture and the training weights in the files **pong_model.json** and **pong_weights.h5**.

---

#### Prerequisites and packages

Run the code using ```python3``` onwards:

```
numpy
tensorflow
keras
time
sys
```

Install the following packages:

```
sudo pip3 install --upgrade numpy tensorflow keras
```

*Note: Install TensorFlow as per instructions on the [TensorFlow website](https://www.tensorflow.org/install/)*

---

#### Running the pre-trained model

Copy and paste the following line in your command line in the PongNet directory to render the Pong game environment, load the model and play against the opponent:

```
python3 pong.py
```

---

#### Training the model from scratch

Copy and paste the following line in your command line in the PongNet directory to start collecting the raw training data and training the model on the new data:

```
python3 pong.py 250000 200
```

where ```250000``` is the number of games and ```200``` is the number of time-steps per game. Training the model with these arguments takes about 2 days on a standard issue MacBook Pro CPU.

---
