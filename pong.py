import gym
import time
import numpy as np
from model import play, load
from training_data import get_trained_model
np.random.seed(100)

env = gym.make('Pong-v0')

# PongNet = get_trained_model(env)
PongNet = load('pong_weights.h5', 'pong_model.json')

print ("Testing model")
input('Press any key to being testing: ')

for i in range(4)[::-1]:
	print (i+1)
	time.sleep(1)

for i in range(10):
	print ("Game {}".format(i+1))
	observation = env.reset()

	# for t in range(1,200):
	t = 0
	while True:
		if (t % 25 == 0):
			print ('\tTimestep {}'.format(t))

		t += 1

		env.render()

		step = play(PongNet, observation.reshape([210,160,3]))
		observation, reward, done, _ = env.step(step)

		if done:
			print ("Exiting game after {} timesteps".format(t+1))
			break
