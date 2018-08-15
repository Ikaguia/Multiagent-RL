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

for nghost in nghosts:
	for agentId in xrange(len(agents)):
		agent = agents[agentId]
		color = colors[agentId]
		x,y = [],[]
		for result in sorted(testRes):
			if result[:2] == (agent,nghost):
				x += [float(result[2])]
				y += [testRes[result][0]]
		z = np.polyfit(x, y, len(x)-1)
		# print x,y,z
		p = np.poly1d(z)
		s = np.linspace(0, x[-1] + 50, 100)
		plt.plot(x, y, color+'.')
		plt.plot(s, p(s), color+'--', label=agent)

		for lgames in (100, 150, 200, 250, 500, 1000):
			print "estimated score for %s %s after %d learn games: %f" % (agent, nghost, lgames, p(lgames))
	plt.xlabel('learn games')
	plt.ylabel('average score in test games')
	plt.legend()
	plt.show()

