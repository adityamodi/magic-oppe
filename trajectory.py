import numpy

class Trajectory():
	def __init__(self):
		self.len = 0
		self.states = []
		self.actions = []
		self.rewards = []
		self.actionProbabilities = []
		self.R = 0
		return