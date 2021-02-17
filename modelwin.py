import numpy as np
import Env
import policy
import trajectory

class ModelWin(Env.Environment):
	def getNumActions(self):
		return 2

	def getNumStates(self):
		return 3

	def getMaxTrajLength(self):
		return 20

	def getPolicy(self, idx):
		if idx == 1:
			return policy.Policy('config/p1_ModelWin.txt', self.getNumActions(), self.getNumStates())
		if idx == 2:
			return policy.Policy('config/p2_ModelWin.txt', self.getNumActions(), self.getNumStates())
		else:
			print("Unknown policy given to ModelWin::getPolicy")
			exit()

	def evaluatePolicy(self, pi): # Evaluate a given policy
		print "Evaluating given policy"
		numSamples = 1000000
		result = 0.0
		for count in range(numSamples):
			state = 0
			print '{0}\r'.format(str(float(count*100)/numSamples)+'%'),
			for t in range(self.getMaxTrajLength()):
				action = pi.getAction(state)
				if state != 0:
					state = 0 # Loop back to state 0 if in state 1 or 2
				else:
					if np.random.binomial(1,0.6):
						if action == 0:
							state = 2 # action 0 moves to state 2 with probability 0.6
						else:
							state = 1 # action 1 moves to state 1 with probability 0.6
					else:
						if action == 0:
							state = 1 # action 0 moves to state 1 with probability 0.4
						else:
							state = 2 # action 1 moves to state 2 with probability 0.4
				reward = 0
				if state == 1:
					reward = 1
				elif state == 2:
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
			state = 0
			for t in range(self.getMaxTrajLength()):
				buff[idx].len = buff[idx].len + 1
				action = pi.getAction(state) # Only one state is observed
				buff[idx].actions.append(action)
				a_prob = pi.getActionProbability(state,action)
				buff[idx].actionProbabilities.append(a_prob)

				if state != 0:
					state = 0
				else:
					if np.random.binomial(1,0.6):
						if action == 0:
							state = 2 # action 0 moves to state 2 with probability 0.6
						else:
							state = 1 # action 1 moves to state 1 with probability 0.6
					else:
						if action == 0:
							state = 1 # action 0 moves to state 1 with probability 0.4
						else:
							state = 2 # action 1 moves to state 2 with probability 0.4
				reward = 0
				if state == 1:
					reward = 1
				elif state == 2:
					reward = -1
				buff[idx].rewards.append(reward)
				buff[idx].R = buff[idx].R + reward
				buff[idx].states.append(state)
		return buff
