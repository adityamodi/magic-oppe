import policy
import modelwin
import modelfail
import gridworld
import hybrid
import model
import trajectory
import magic_traj
import algos
import numpy as np

def compute_data(mdp, pibIdx, pieIdx, filename, N, delta):
	unObserved = False
	numTrajs = np.zeros(N)
	numTrajs[0] = 2
	for i in range(1,N):
		numTrajs[i] = int(3 + 2**i)
	# print 'numTrajs array : ' + str(numTrajs)

	pib = mdp.getPolicy(pibIdx)
	pie = mdp.getPolicy(pieIdx)

	target = mdp.evaluatePolicy(pie) # Target value for MSE error evaluated using Monte Carlo

	gamma = 1.0 # All tasks are episodic

	J = np.zeros(mdp.getMaxTrajLength() + 2)
	for i in range(mdp.getMaxTrajLength()+2):
		J[i] = i - 2
	# print 'Vector J:'
	# print J
	# raw_input('Press Enter')

	numAlgos = 8
	numTrials = 50

	MSE = np.zeros((N,numAlgos))
	MSE_error = np.zeros((N,numAlgos))
	cond_tot = np.zeros(N)

	for trajIdx in range(N):
		n = int(numTrajs[trajIdx])
		outAM = np.zeros(numTrials)
		outIS = np.zeros(numTrials)
		outPDIS = np.zeros(numTrials)
		outWIS = np.zeros(numTrials)
		outCWPDIS = np.zeros(numTrials)
		outDR = np.zeros(numTrials)
		outWDR = np.zeros(numTrials)
		outMagic = np.zeros(numTrials)
		cond = np.zeros(numTrials)

		for trialIdx in range(numTrials):
			# print 'numTraj = '+str(n)
			data = []
			data = mdp.generateTrajectories(pib, n)
			# print data
			# exit()
			mod = model.Model(data, mdp.getNumStates(), mdp.getNumActions(), mdp.getMaxTrajLength(), unObserved)
			# print mod.P
			# print mod.R

			mod.loadEvalPolicy(pie, mdp.getMaxTrajLength())
			mag_data = []
			for i in range(n):
				# print 'Actions'
				# print data[i].actions
				# print 'Rewards'
				# print data[i].rewards
				mag_data.append(magic_traj.magicTrajectory(data[i], pie, mod.Q, mod.V))
				# print mag_data[i].r
				# print mag_data[i].len

			# print 'Eval Policy Value: ' + str(mod.evalPolicyvalue)
			# raw_input('Press Enter')
			outAM[trialIdx] = mod.evalPolicyvalue
			outIS[trialIdx] = algos.IS(mag_data, False, gamma)
			outPDIS[trialIdx] = algos.PDIS(mag_data, False, gamma)
			outWIS[trialIdx] = algos.IS(mag_data, True, gamma)
			outCWPDIS[trialIdx] = algos.PDIS(mag_data, True, gamma)
			outDR[trialIdx] = algos.DR(mag_data, False, gamma)
			outWDR[trialIdx] = algos.DR(mag_data, True, gamma)
			outMagic[trialIdx], cond[trialIdx] = algos.MAGIC(mag_data, gamma, J, mod.evalPolicyvalue, delta)
			print 'Completed trial no. ' + str(trialIdx) + ' for trajectory index ' + str(trajIdx)

		outAM = outAM - target
		outIS = outIS - target
		outPDIS = outPDIS - target
		outWIS = outWIS - target
		outCWPDIS = outCWPDIS - target
		outDR = outDR - target
		outWDR = outWDR - target
		outMagic = outMagic - target

		MSE[trajIdx, 0] = np.dot(outAM, outAM)/float(numTrials)
		MSE[trajIdx, 1] = np.dot(outIS, outIS)/float(numTrials)
		MSE[trajIdx, 2] = np.dot(outPDIS, outPDIS)/float(numTrials)
		MSE[trajIdx, 3] = np.dot(outWIS, outWIS)/float(numTrials)
		MSE[trajIdx, 4] = np.dot(outCWPDIS, outCWPDIS)/float(numTrials)
		MSE[trajIdx, 5] = np.dot(outDR, outDR)/float(numTrials)
		MSE[trajIdx, 6] = np.dot(outWDR, outWDR)/float(numTrials)
		MSE[trajIdx, 7] = np.dot(outMagic, outMagic)/float(numTrials)
		cond_tot[trajIdx] = np.mean(cond)

		MSE_error[trajIdx, 0] = np.std(np.multiply(outAM, outAM))/np.sqrt(numTrials)
		MSE_error[trajIdx, 1] = np.std(np.multiply(outIS, outIS))/np.sqrt(numTrials)
		MSE_error[trajIdx, 2] = np.std(np.multiply(outPDIS, outPDIS))/np.sqrt(numTrials)
		MSE_error[trajIdx, 3] = np.std(np.multiply(outWIS, outWIS))/np.sqrt(numTrials)
		MSE_error[trajIdx, 4] = np.std(np.multiply(outCWPDIS, outCWPDIS))/np.sqrt(numTrials)
		MSE_error[trajIdx, 5] = np.std(np.multiply(outDR, outDR))/np.sqrt(numTrials)
		MSE_error[trajIdx, 6] = np.std(np.multiply(outWDR, outWDR))/np.sqrt(numTrials)
		MSE_error[trajIdx, 7] = np.std(np.multiply(outMagic, outMagic))/np.sqrt(numTrials)
	# Let's now write the MSE results to a file
	# np.savetxt(filename + '_MSE.txt', MSE)
	# np.savetxt(filename+'_MSE_error.txt', MSE_error)
	np.savetxt('condition_number_hybrid.txt', cond_tot)

if __name__ == '__main__':
	print 'Running 50 trials for the given MDP model...'

	trueHorizon = False
	delta = 0.1

	# env = modelfail.ModelFail()
	# print 'Generating data for ModelFail MDP...'
	# compute_data(env, 1, 2, 'out_ModelFail', 15, delta)

	# env = modelwin.ModelWin()
	# print 'Generating data for ModelWin MDP...'
	# compute_data(env, 1, 2, 'out_MSE_ModelWin', 15, delta)

	env = hybrid.Hybrid()
	print 'Generating data for Hybrid Domain MDP...'
	compute_data(env, 1, 2, 'out_HybridDomain', 13, delta)

	# env = gridworld.Gridworld(trueHorizon)
	# print 'Generating data for Gridworld MDP p4 p5...'
	# compute_data(env, 4, 5, 'out_GridWorld_p4p5', 11, delta)

	print 'Done...'