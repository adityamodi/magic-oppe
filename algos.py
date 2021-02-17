import numpy as np
from numpy import linalg as LA
import trajectory
import sys
from gurobipy import *
from decimal import *

def IS(B, isWeighted, gamma):
	n = len(B)
	IWs = np.ones(n)
	R = np.zeros(n)
	for i in range(n):
		currGamma = 1.0
		for t in range(B[i].len):
			IWs[i] = IWs[i]*(float(B[i].pie[t])/float(B[i].pib[t]))
			R[i] = R[i] + currGamma*B[i].r[t]
			currGamma = currGamma*gamma
	if isWeighted:
		return np.dot(IWs,R)/sum(IWs)
	else:
		return np.dot(IWs, R)/float(n)

def PDIS(B, isWeighted, gamma):
	n = len(B)
	L = 0
	for i in range(n):
		L = max(0, B[i].len)
	IWs = np.ones(n)
	currGamma = 1.0
	result = 0.0
	for t in range(L):
		for i in range(n):
			if t < B[i].len:
				IWs[i] = IWs[i] * float(B[i].pie[t])/float(B[i].pib[t])
		IWSum = IWs.sum()
		for i in range(n):
			if t < B[i].len:
				if isWeighted:
					result = result + currGamma*(IWs[i]/IWSum)*B[i].r[t]
				else:
					result = result + currGamma*(IWs[i]/n)*B[i].r[t]
		currGamma = currGamma * gamma
	return result

def DR(B, isWeighted, gamma):
	n = len(B)
	L = 0
	for i in range(n):
		L = max(L, B[i].len)

	rho = np.zeros((L,n))
	for i in range(n):
		rho[0,i] = float(B[i].pie[0])/float(B[i].pib[0])
	for t in range(1,L):
		for i in range(n):
			if t < B[i].len:
				rho[t,i] = rho[t-1,i] * (float(B[i].pie[t])/float(B[i].pib[t]))
			else:
				rho[t,i] = rho[t-1,i] # In terminal state, only single action
	w = np.zeros((L,n))
	if isWeighted:
		for t in range(L):
			w[t] = rho[t]/np.sum(rho[t])
	else:
		w = rho/float(n)
	# Computed the weights in doubly robust, now computing the DR and WDR equations
	result = 0.0
	w2 = 0.0
	for i in range(n):
		currGamma = 1.0
		for t in range(min(L, B[i].len)):
			result = result + currGamma*w[t,i]*B[i].r[t]
			if t == 0:
				w2 = 1.0/float(n)
			else:
				w2 = w[t-1,i]
			result = result - currGamma*(w[t,i]*B[i].Q[t] - w2*B[i].V[t])
			currGamma = currGamma * gamma
	return result

# Function for covariance between two vectors
def cov(a, b):
	if len(a) <= 1:
		return 0.0
	muA = np.mean(a)
	muB = np.mean(b)
	temp = 0.0
	for i in range(len(a)):
		temp = temp + (a[i] - muA)*(b[i] - muB)
	return temp/float(len(a) - 1)

def MAGIC(D, gamma, J, mvpie, delta = 0.1, isWeighted = True, kappa = 200, \
		epsilon = 0.00001):
	'''
	Confidence interval taken as 0.1 as in the original paper
	Personal communication with Philip Thomas: ideally there is a multiplicative
	factor of n is missing in the covariance calculation
	'''
	n = len(D)
	if n < 2:
		sys.stderr.write('Error in MAAGIC: Need at least two trajectories\n')
		exit()

	for i in range(n):
		if D[i].len < 1:
			sys.stderr.write('Error in MAGIC: Empty trajectory')
			exit()
		if (len(D[i].pib) != D[i].len) or (len(D[i].pie) != D[i].len) \
			or (len(D[i].Q) != D[i].len) or (len(D[i].V) != D[i].len) \
			or (len(D[i].r) != D[i].len):
			sys.stderr.write('Error in MAGIC: Unexpected length in data vector')
			exit()

	# Calculate maximum trajectory length considered in n-step returns
	L = int(max(J) + 1)
	#for i in range(len(J)):
	#	L = max(L, J[i]+1)

	# Computing the importance weights
	rho = np.zeros((L,n))
	for i in range(n):
		rho[0,i] = float(D[i].pie[0])/float(D[i].pib[0])
	for t in range(1,L):
		for i in range(n):
			if t < D[i].len:
				rho[t,i] = rho[t-1,i] * (float(D[i].pie[t])/float(D[i].pib[t]))
			else:
				rho[t,i] = rho[t-1,i]

	w = np.zeros((L,n))
	if isWeighted:
		for t in range(L):
			w[t] = rho[t]/np.sum(rho[t])
	else:
		w = rho/float(n)
	# print w
	# print 'Eval Policy Value: ' + str(mvpie)

	#Computing the different length returns and storing in g as in paper's pseudocode
	g = np.zeros((n, len(J)))
	for i in range(n):
		for j in range(len(J)):
			rlen = int(J[j]) # Current return length
			# print 'RETURN_LENGTH: '+str(rlen)

			if rlen == -2:
				g[i,j] = float(mvpie)/float(n)
				continue
			# Computing the part a of the equation summing up rewards in (W)DR
			currGamma = 1.0
			for t in range(min(rlen,D[i].len-1)+1):
				g[i,j] = g[i,j] + currGamma*w[t,i]*D[i].r[t]
				currGamma = currGamma * gamma
				# print g[i,j]
			# Computing the part b in the equation using the model estimates
			if rlen < D[i].len - 1:
				if rlen == -1:
					g[i,j] = g[i,j] + currGamma*D[i].V[rlen+1]/float(n)
				else:
					g[i,j] = g[i,j] + currGamma*w[rlen,i]*D[i].V[rlen+1]
			# Computing the part c in the equation using the model part of (W)DR
			currGamma = 1.0
			for t in range(min(rlen, D[i].len - 1)+1):
				w2 = 1.0/float(n) if t == 0 else w[t-1,i]
				temp = w[t,i]*D[i].Q[t] - w2*D[i].V[t]
				g[i,j] = g[i,j] - currGamma*temp
				currGamma = currGamma*gamma
	# print g
	gVec = np.sum(g,0)
	# Computing the sample covariance matrix
	Omega = np.zeros((len(J), len(J)))
	for i in range(len(J)):
		for j in range(len(J)):
			Omega[i,j] = cov(g[:,i], g[:,j])
	# We haven't multiplied this covariance matrix by n as was there in the paper
	# print gVec
	# Now computing the bias vector
	gInf = DR(D, isWeighted, gamma)
	# print 'DR estimate' + str(gInf)
	# Using bootstrap resampling
	bS_est = np.zeros(kappa)
	for i in range(kappa):
		B = []
		for j in range(n):
			idx = np.random.randint(0,n)
			B.append(D[idx])
		bS_est[i] = DR(B, isWeighted, gamma)
		# print 'Curr DR estimate: ' + str(bS_est[i])
	bS_est = np.sort(bS_est)
	CI_low = min(gInf, bS_est[int(kappa*float(delta/2.0))])
	# print 'CI_low: ' + str(CI_low)
	CI_high = max(gInf, bS_est[int(kappa*float(1.0-delta/2.0))])
	# print 'CI_high: ' + str(CI_high)
	# Computing the bias vector
	b = np.zeros(len(J))
	for i in range(len(J)):
		if gVec[i] > CI_high:
			b[i] = gVec[i] - CI_high
		elif gVec[i] < CI_low:
			b[i] = CI_low - gVec[i]
	# Compute the weight vector x by minimizing MSE
	# print 'Bias'
	# print b
	b = np.resize(b, (len(J),1))
	# print 'Bias New'
	# print b
	A = Omega + np.multiply(b,b.T) + epsilon*np.eye(len(J))
	cond = LA.cond(A)
	# print A
	# raw_input('Press Enter to continue...')
	# Optimization objective via Gurobi model
	model = Model()
	x = {}
	for i in range(len(J)):
		x[i] = model.addVar(lb = 0.0, ub = 1.0, \
			vtype = GRB.CONTINUOUS, name = 'x_'+str(i))
	model.update()
	constr = LinExpr()
	for i in range(len(J)):
		constr = constr + x[i]
	model.addConstr(constr, GRB.EQUAL, 1) # Adding constraint of simplex
	model.update()
	obj = QuadExpr()
	for i in range(len(J)):
		for j in range(len(J)):
			obj = obj + x[i]*A[i,j]*x[j]
	model.setObjective(obj, GRB.MINIMIZE) # Set objective using the A matrix above
	model.update()
	model.optimize() # Optimize model
	val = np.zeros(len(J))
	for i in range(len(J)):
		val[i] = x[i].X
	# Now return the weighted average of the J-step returns
	print 'Condition Number: ' + str(cond)
	return np.dot(val, gVec), cond














