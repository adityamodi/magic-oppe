import numpy as np
import Env
import policy
import trajectory

class ModelFail(Env.Environment):
	def getNumActions(self):
		return 2

	def getNumStates(self):
		return 1

	def getMaxTrajLength(self):
		return 2

	def getPolicy(self, idx):
		if idx == 1:
			return policy.Policy('config/p1_ModelFail.txt', self.getNumActions(), self.getNumStates())
		if idx == 2:
			return policy.Policy('config/p2_ModelFail.txt', self.getNumActions(), self.getNumStates())
		else:
			print("Unknown policy given to ModelFail::getPolicy")
			exit()

	def evaluatePolicy(self, pi): # Evaluate a given policy
		print "Evaluating given policy"
		numSamples = 100000
		result = 0.0
		for count in range(numSamples):
			mid_state = 0
			print '{0}\r'.format(str(float(count*100)/numSamples)+'%'),
			for t in range(self.getMaxTrajLength()):
				a = pi.getAction(0)
				if t == 0:
					mid_state = a
				reward = 0
				if t == 1:
					if mid_state == 0:
						reward = 1
					else:
						reward = -1
				result = result + reward
		return float(result)/numSamples

	def generateTrajectories(self, pi, numTraj):
		#if len(buff) < numTraj:
		#	n = len(buff)
		#	for idx in range(numTraj - len(buff)):
		#		buff.append(trajectory.Trajectory()) #TODO
		buff = [trajectory.Trajectory() for num in range(numTraj)]
		for idx in range(numTraj):
			buff[idx].len = 0
			buff[idx].actionProbabilities = []
			buff[idx].actions = []
			buff[idx].rewards = []
			buff[idx].states = [0]
			buff[idx].R = 0
			mid_state = 0 # Store the middle unobserved state according to the action
			for t in range(self.getMaxTrajLength()):
				buff[idx].len = buff[idx].len + 1
				a = pi.getAction(0) # Only one state is observed
				buff[idx].actions.append(a)
				a_prob = pi.getActionProbability(0,a)
				buff[idx].actionProbabilities.append(a_prob)

				if t == 0:
					mid_state = a

				reward = 0

				if t == 1:
					if mid_state == 0:
						reward = 1
					else:
						reward = -1
				buff[idx].rewards.append(reward)
				buff[idx].R = buff[idx].R + reward

				if t == self.getMaxTrajLength():
					break
				buff[idx].states.append(0)
		return buff
