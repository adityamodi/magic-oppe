import trajectory
import policy
import numpy as np

'''	Class for building the model of MDP with 
	finite state-action space with specified finite horizon
'''

class Model():
	def __init__(self, trajs, numStates, numActions, L, unObserved):
		self.numStates = numStates
		self.numActions = numActions
		self.L = int(L)
		self.sA_Counts = np.zeros((numStates,numActions))
		self.sA_Counts_w_Horizon = np.zeros((numStates,numActions))
		self.sAS_Counts = np.zeros((numStates, numActions, numStates + 1))
		self.sAS_Counts_w_Horizon = np.zeros((numStates, numActions, numStates + 1))
		self.P = np.zeros((numStates, numActions, numStates + 1))
		self.R = np.zeros((numStates, numActions, numStates + 1))
		self.N = len(trajs)
		self.d0 = np.zeros((numStates))
		self.Q = np.zeros((self.L, self.numStates + 1, self.numActions))
		self.V = np.zeros((self.L, self.numStates + 1))
		self.evalPolicyvalue = 0

		for i in range(self.N): # Count the transitions and sum the rewards
			for j in range(trajs[i].len):
				curr_s = trajs[i].states[j]
				curr_a = trajs[i].actions[j]
				if j == trajs[i].len - 1:
					sPrime = self.numStates
				else:
					sPrime = trajs[i].states[j+1]
				r = trajs[i].rewards[j]
				if j != self.L - 1:
					self.sA_Counts[curr_s, curr_a] = self.sA_Counts[curr_s, curr_a] + 1
					self.sAS_Counts[curr_s, curr_a, sPrime] = self.sAS_Counts[curr_s, curr_a, sPrime] + 1
				else:
					if sPrime != self.numStates:
						exit()
				self.sA_Counts_w_Horizon[curr_s, curr_a] = self.sA_Counts_w_Horizon[curr_s, curr_a] + 1
				self.sAS_Counts_w_Horizon[curr_s, curr_a, sPrime] = self.sAS_Counts_w_Horizon[curr_s, curr_a, sPrime] + 1
				self.R[curr_s, curr_a, sPrime] = self.R[curr_s, curr_a, sPrime] + r

		for i in range(self.N): # Compute the initial distribution d0
			self.d0[trajs[i].states[0]] = self.d0[trajs[i].states[0]] + float(1.0)/self.N

		rMin = trajs[0].rewards[0]
		for i in range(self.N): # Compute the minimum value for reward
			for j in range(trajs[i].len):
				rMin = min(rMin, trajs[i].rewards[j])

		for s in range(self.numStates):
			for a in range(self.numActions):
				for sP in range(self.numStates + 1):
					if self.sA_Counts[s,a] == 0:
						if unObserved:
							self.P[s,a,sP] = (1 if s == sP else 0) # self-transition for unobserved state-action pair (Nan's paper)
						else:
							self.P[s,a,sP] = (1 if sP == self.numStates else 0) # terminating if unobserved
					else:
						self.P[s,a,sP] = float(self.sAS_Counts[s,a,sP])/float(self.sA_Counts[s,a])
					if self.sAS_Counts_w_Horizon[s,a,sP] == 0:
						if unObserved:
							self.R[s,a,sP] = rMin # set reward to minimum if unobserved
						else:
							self.R[s,a,sP] = 0 # set reward to 0 if unobserved as there is no data
					else:
						self.R[s,a,sP] = float(self.R[s,a,sP])/float(self.sAS_Counts_w_Horizon[s,a,sP])

	def loadEvalPolicy(self, pi, L):
		# Copy the action probabilities
		actionProbabilities = np.zeros((self.numStates, self.numActions))
		for s in range(self.numStates):
			actionProbabilities[s] = pi.getActionProbabilities(s)

		# Compute the Q values for the given policy at each time step
		self.Q = np.zeros((L, self.numStates + 1, self.numActions))
		for t in range(L)[::-1]:
			for s in range(self.numStates):
				for a in range(self.numActions):
					for sP in range(self.numStates + 1):
						self.Q[t,s,a] = self.Q[t,s,a] + self.P[s,a,sP]*self.R[s,a,sP]
						if (sP != self.numStates) and (t != L - 1):
							self.Q[t,s,a] = self.Q[t,s,a] + self.P[s,a,sP]*(np.dot(actionProbabilities[sP], self.Q[t+1,sP]))

		# Compute the value functions
		self.V = np.zeros((L, self.numStates + 1))
		for t in range(L):
			for s in range(self.numStates):
				self.V[t,s] = np.dot(actionProbabilities[s], self.Q[t,s])

		self.evalPolicyvalue = 0
		for s in range(self.numStates):
			self.evalPolicyvalue = self.evalPolicyvalue + self.d0[s]*self.V[0,s]

	def generateTrajectories(pi, N):
		_pi = np.zeros((self.numStates, self.numActions))
		for s in range(self.numStates):
			_pi[s] = pi.getActionProbabilities(s)

		trajs = [trajectory.Trajectory() for num in range(N)]
		for i in range(N):
			state = np.argmax(np.random.multinomial(1, self.d0))
			for t in range(L):
				action = np.argmax(np.random.multinomial(1, _pi[state]))
				newState = np.argmax(np.random.multinomial(1, self.P[state, action]))
				reward = self.R[state, action, newState]
				trajs[i].R = trajs[i].R + reward
				trajs[i].states.append(state)
				trajs[i].actions.append(action)
				trajs[i].rewards.append(reward)
				trajs[i].actionProbabilities.append(_pi[state, action])

				if (newState == self.numStates) or (t == self.L - 1):
					trajs[i].len = t+1
					break

				state = newState
		return trajs