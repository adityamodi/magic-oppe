import numpy as np
import Env
import policy
import trajectory

class Gridworld(Env.Environment):
	g_gridworld_size = 4
	g_gridworld_maxTrajLen = 100
	def __init__(self, trueHorizon):
		self.trueHorizon = trueHorizon

	def getNumActions(self):
		return 4

	def getNumStates(self):
		return self.g_gridworld_size*self.g_gridworld_size - 1

	def getMaxTrajLength(self):
		if self.trueHorizon:
			return self.g_gridworld_maxTrajLen
		else:
			return self.g_gridworld_maxTrajLen + 1 # Induce partial observability by falsifying the horizon

	def getPolicy(self, idx):
		if idx == 1:
			return policy.Policy('config/p1.txt', self.getNumActions(), self.getNumStates())
		if idx == 2:
			return policy.Policy('config/p2.txt', self.getNumActions(), self.getNumStates())
		if idx == 3:
			return policy.Policy('config/p3.txt', self.getNumActions(), self.getNumStates())
		if idx == 4:
			return policy.Policy('config/p4.txt', self.getNumActions(), self.getNumStates())
		if idx == 5:
			return policy.Policy('config/p5.txt', self.getNumActions(), self.getNumStates())
		else:
			print("Unknown policy given to Gridworld::getPolicy")
			exit()

	def evaluatePolicy(self, pi): # Evaluate a given policy
		print "Evaluating given policy"
		numSamples = 10000
		result = 0.0
		for count in range(numSamples):
			xc = 0
			yc = 0
			print '{0}\r'.format(str(float(count*100)/numSamples)+'%'),
			for t in range(self.g_gridworld_maxTrajLen):
				action = pi.getAction(xc + yc*self.g_gridworld_size)
				if action == 0 and (xc < self.g_gridworld_size - 1):
					xc = xc + 1 # Move one step right if not at end
				if action == 1 and (xc > 0):
					xc = xc - 1 # Move one step left if not at end
				if action == 2 and (yc < self.g_gridworld_size - 1):
					yc = yc + 1 # Move one step down if not at end
				if action == 3 and (yc > 0):
					yc = yc - 1 # Move one step up if not at end
				if (xc == 1) and (yc == 1):
					result = result - 10
				elif (xc == 1) and (yc == 3):
					result = result + 1
				elif (xc == self.g_gridworld_size - 1) and (yc == self.g_gridworld_size - 1):
					result = result + 10
				else:
					result = result - 1
				if (t == self.g_gridworld_maxTrajLen - 1) or ((xc == self.g_gridworld_size - 1) and (yc == self.g_gridworld_size - 1)):
					break
		return float(result)/float(numSamples)

	def generateTrajectories(self, pi, numTraj):
		buff = [trajectory.Trajectory() for num in range(numTraj)]
		for idx in range(numTraj):
			buff[idx].len = 0
			buff[idx].actionProbabilities = []
			buff[idx].actions = []
			buff[idx].rewards = []
			buff[idx].states = [0]
			buff[idx].R = 0
			x = 0
			y = 0
			t = 0
			for t in range(self.g_gridworld_maxTrajLen):
				buff[idx].len = buff[idx].len + 1
				action = pi.getAction(buff[idx].states[t])
				buff[idx].actions.append(action)
				a_prob = pi.getActionProbability(buff[idx].states[t],action)
				buff[idx].actionProbabilities.append(a_prob)

				if ((action == 0) and (x < self.g_gridworld_size - 1)):
					x = x + 1
				elif ((action == 1) and (x > 0)):
					x = x - 1
				elif ((action == 2) and (y < self.g_gridworld_size - 1)):
					y = y + 1
				elif ((action == 3) and (y > 0)):
					y = y - 1

				reward = 0.0
				if (x == 1) and (y == 1):
					reward = -10.0
				elif (x == 1) and (y == 3):
					reward = 1
				elif (x == self.g_gridworld_size - 1) and (y == self.g_gridworld_size - 1):
					reward = 10.0
				else:
					reward = -1.0
				buff[idx].rewards.append(reward)
				buff[idx].R = buff[idx].R + reward

				if (t == self.g_gridworld_maxTrajLen - 1) \
					or ((x == self.g_gridworld_size - 1) and (y == self.g_gridworld_size - 1)):
					break

				buff[idx].states.append(x + y*self.g_gridworld_size)
		return buff



