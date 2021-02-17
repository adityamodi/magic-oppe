import sys
import numpy as np

class Policy():
	def __init__(self, filename, numActions, numStates):
		self.numActions = numActions
		self.numStates = numStates
		self.theta = np.loadtxt(filename)
	def getActionProbabilities(self, state):
		ap = np.zeros(self.numActions)
		for a in range(self.numActions):
			ap[a] = self.theta[a*self.numStates + state]
		ap = np.exp(ap)
		ap = ap/sum(ap)
		return ap
	def getActionProbability(self, state, action):
		return self.getActionProbabilities(state)[action]
	def getAction(self, state):
		sample = np.random.multinomial(1,self.getActionProbabilities(state))
		return np.argmax(sample)