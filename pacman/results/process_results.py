# -*- coding: utf-8 -* 

import glob
import sys
import os
sys.path.insert(1, os.path.join(sys.path[0], '..'))
import plot

import numpy as np
import matplotlib.pyplot as plt


files  = [file for file in glob.glob('*.txt') + glob.glob('results/*.txt') if file.split('/')[-1].split('-')[0] in ('test',)]

testRes = {}

for fileName in sorted(files):
	try:
		with open(fileName, 'r') as file:
			lines = [line for line in file]
		fileName = fileName.split('/')[-1].split('.')[0]
		learnStart = 4
		learnScores = [float(lines[learnStart][1:])]
		for line in lines[learnStart+1:]:
			if not line.startswith('aF'): break
			learnScores += [float(line[2:])]

		testStart = 4 + len(learnScores) + 3
		testScores = [float(lines[testStart][1:])]
		for line in lines[testStart+1:]:
			if not line.startswith('aF'): break
			testScores += [float(line[2:])]
		if fileName.split('-')[0] == 'test':
			testRes[tuple(fileName.split('-')[1].split('_'))] = (
				sum(testScores) / len(testScores),
			)
	except:
		print fileName,"failed"


agents = sorted(set([i[0] for i in testRes]))
nghosts = sorted(set([i[1] for i in testRes]))
colors = 'bgrcmykw'

npoints = None

for nghost in nghosts:
	for agentId in xrange(len(agents)):
		agent = agents[agentId]
		color = colors[agentId]
		x,y = [0],[0]
		for result in sorted(testRes):
			_agent,_nghost,_pm,_nlearn = result
			_testScore = testRes[result][0]
			if (result[:2] == (agent,nghost)) and (int(_nlearn) > 10):
				# print result
				x += [float(_nlearn)]
				y += [_testScore]
		if len(x) == npoints or npoints == None:
			npoints = len(x)

			z = np.polyfit(x, y, 1)
			# print x,y,z
			p = np.poly1d(z)
			s1 = np.linspace(0, 150, 1000)
			s2 = np.linspace(150, 250, 1000)
			# plt.plot(x, y, color+'.')
			plt.plot(s1, p(s1), color+'-', label=agent)
			plt.plot(s2, p(s2), color+'--', label='Previsao para ' + agent)

			for lgames in (200, 225, 250):
				print "estimated average score for %s %s after %d learn games: %f" % (agent, nghost, lgames, p(lgames))
	plt.xlabel('Jogos de aprendizagem')
	plt.ylabel('Pontuacao media nos jogos de teste')
	plt.legend()
	plt.ylim(-1000,5000)
	plt.show()

