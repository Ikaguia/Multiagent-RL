#!/usr/bin/python

import subprocess

class Result:
    """
    Container for simulations data.
    """
    def __init__(self, name, avg_score=0, win_rate=0):
        self.name = name
        self.avg_score = avg_score
        self.win_rate = win_rate

    def __str__(self):
        return '%s - Avg. Score: %f Win Rate %f' % (self.name, self.avg_score, self.win_rate)

def parseOutput(output, game_name):
    """
    Parses output generated by the Pacman simulator in order to acquire relevant
    data.
    """
    result = Result(game_name)

    for line in output:
        if 'Average Score' in line:
            result.avg_score = float(line.split()[-1])
        if 'Win Rate' in line:
            result.win_rate = float(line.split()[-1][1:-1])

    return result

def runGame(pacmanAgent, ghostAgent, layout, repetitions):
    """
    Runs Pacman game with configured agents and layout, parsing its results.
    """
    print 'Running game'

    processArgs = [
        'python', 'pacman.py', # Game initialization command
        '-p', pacmanAgent, # Pacman controller class
        '-g', ghostAgent, # Ghosts controller class
        '-l', layout, # Game layout map file
        '-n', repetitions, # Number of times the game will be repeated
        '-q' # Suppress visual output
    ]
    game_name = '%s_%s_%s_%s' %(pacmanAgent, ghostAgent, layout, repetitions)
    proc = subprocess.Popen(processArgs, stdout=subprocess.PIPE)
    result = parseOutput(proc.stdout, game_name)
    print result

if __name__ == '__main__':
    game_definitions = [
        ['RandomAgent', 'RandomGhost', 'originalClassic', '3'],
        ['GreedyAgent', 'RandomGhost', 'originalClassic', '3'],
    ]

    for game_definition in game_definitions:
        print game_definition
        runGame(*game_definitions)