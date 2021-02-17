import trajectory
import numpy as np

class magicTrajectory():
	def __init__(self, traj, pie, Q, V):
		self.len = traj.len
		self.pib = traj.actionProbabilities
		self.r = traj.rewards
		self.pie = np.zeros(traj.len)
		self.Q = np.zeros(traj.len)
		self.V = np.zeros(traj.len)
		for t in range(traj.len):
			state = traj.states[t]
			action = traj.actions[t]
			self.pie[t] = pie.getActionProbability(state, action)
			self.Q[t] = Q[t, state, action]
			self.V[t] = V[t, state]
		return
