import numpy
import sys

class Environment(object):
	def getNumActions(self):
		return 0
	def getNumStates(self):
		return 0
	def generateTrajectories(self,buff, pi, numTraj):
		pass
	def evaluatePolicy(self,pi):
		return 0
	def getMaxTrajLength(self):
		return 0
	def getPolicy(self,idx):
		return 0

if __name__=='__main__':
	pass