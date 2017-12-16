import numpy as np
import keras
from keras.models import Sequential, model_from_json
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, Activation, Dropout, Flatten
np.random.seed(100)

def build(loss, optimizer, env):
	model = Sequential()

	model.add(Conv2D(32, (3,3), padding='same', input_shape=env.observation_space.shape))
	model.add(Activation('relu'))
	model.add(MaxPooling2D(pool_size=(2,2)))

	model.add(Conv2D(64, (3,3), padding='same'))
	model.add(Activation('relu'))
	model.add(MaxPooling2D(pool_size=(2,2)))

	model.add(Flatten())

	model.add(Dense(128))
	model.add(Activation('relu'))

	model.add(Dense(env.action_space.shape[0]))
	model.add(Activation('softmax'))

	model.compile(loss=loss, optimizer=optimizer, metrics=['accuracy'])
	model.summary()

	return model

def train(model, x_train, y_train, epochs=5, batch_size=64):
	model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size)

	model_json = model.to_json()
	with open("pong_model.json", "w") as json_file:
		json_file.write(model_json)
	model.save_weights("pong_weights.h5")
	print("Saved model and weights to disk")

	return model

def play(model, window):
	screen = np.array([window])
	prediction = model.predict(screen)
	index = (prediction[0].argmax())
	actions = [0, 1, 2, 3, 4, 5]
	# action = actions[index]
	action = np.random.choice(actions)

	return action

def load(pong_weights, pong_model):
	json_file = open(pong_model, 'r')
	loaded_model_json = json_file.read()
	json_file.close()
	loaded_model = model_from_json(loaded_model_json)
	loaded_model.load_weights(pong_weights)
	print("Loaded model from disk")
	loaded_model.summary()

	return loaded_model
