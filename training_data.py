import sys
import gym
import time
import numpy as np
from model import build, train
from keras.utils import np_utils
np.random.seed(100)

def get_trained_model(env):
	n_eps = int(sys.argv[1])
	n_ts = int(sys.argv[2])

	observations = []
	possible_actions = []

	print ("Playing {} training games across {} timesteps".format(n_eps, n_ts))
	for i in range(n_eps):

		if (i % 500 == 0):
			print ('Checkpoint {}'.format(i))

		observation = env.reset()

		for t in range(n_ts):
			action = env.action_space.sample()
			observation, reward, done, info = env.step(action)

			if reward == 1.0:
				possible_actions.append(action)
				observations.append(observation)

			if done:
				print ("Collected training data")
				break

	print ("Training model")
	observations = np.array(observations).astype('float32')
	observations = observations.reshape([len(observations), 210, 160, 3])
	observations /= 255

	possible_actions = np.array(possible_actions).astype('float32')
	possible_actions = np_utils.to_categorical(possible_actions, 6)

	try:
		PongNet = build('categorical_crossentropy', 'rmsprop', env)
		PongNet = train(PongNet, observations, possible_actions, 5, 64)
	except:
		pass

	return PongNet
