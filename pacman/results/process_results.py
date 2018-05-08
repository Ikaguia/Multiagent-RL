import glob
import sys
import os
sys.path.insert(1, os.path.join(sys.path[0], '..'))
import plot


files  = [file for file in glob.glob('*.txt') + glob.glob('results/*.txt') if file.split('/')[-1].split('-')[0] in ('learn','test')]

learnRes = {}
testRes = {}

for fileName in sorted(files):
	print fileName
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
		# if fileName == 'learn-ai2_2g_fleet_100.txt':
		# 	print learnScores
		# 	print testScores
		if fileName.split('-')[0] == 'learn':
			learnRes[tuple(fileName.split('-')[1].split('_'))] = (
				plot.calculate_regression_coefficients(learnScores, degree=1)[1],
				plot.calculate_regression_coefficients(learnScores, degree=1)[0],
			)
		if fileName.split('-')[0] == 'test':
			testRes[tuple(fileName.split('-')[1].split('_'))] = (
				sum(testScores) / len(testScores),
			)
	except:
		print "failed"

print "learn results:"
for result in sorted(learnRes):
	print "\t",result,": %d + %fx" % learnRes[result]
print "test results:"
for result in sorted(testRes):
	print "\t",result,":",testRes[result][0]
