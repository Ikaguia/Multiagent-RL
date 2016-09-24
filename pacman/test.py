#!/usr/bin/python
# -*- coding: utf-8 -*-
"""Test simulations on the Berkeley Pacman Files.

Test simulations on the Berkeley Pacman Files and parse its results to,
aquire relevant data.
"""


import subprocess

__author__ = "Matheus Portela and Guilherme N. Ramos"
__credits__ = ["Matheus Portela", "Guilherme N. Ramos", "Renato Nobre",
               "Pedro Saman"]
__maintainer__ = "Guilherme N. Ramos"
__email__ = "gnramos@unb.br"


class Result:
    """Container for simulations data."""

    def __init__(self, name, avg_score=0, win_rate=0):
        """Constructor for the Result Class.

        Args:
            name (str): Name of the game test being created.
            avg_score (float): Average score for the test. Default = 0.
            win_rate (float): Win rate for the test. Default = 0.
        Attributes:
            name: Name of the game test being created.
            avg_score: Average score for the test.
            win_rate: Win rate for the test.
        """
        self.name = name
        self.avg_score = avg_score
        self.win_rate = win_rate

    def __str__(self):
        """Define Behavior for when str() is called.

        Print the Average score and the Win Rate of the Result object.
        """
        return '%s - Avg. Score: %f Win Rate %f' % (self.name,
                                                    self.avg_score,
                                                    self.win_rate)


def parseOutput(output, game_name):
    """Parse output generated by the Pacman simulator.

    Parse output generated by the Pacman simulator in order to acquire
    relevant data.

    Args:
        output (str): Standard output of subprocess to be printed.
        game_name (str): Name of the game being tested.
    Returns:
        result (str): Formated string from output.
    """
    result = Result(game_name)

    for line in output:
        if 'Average Score' in line:
            result.avg_score = float(line.split()[-1])
        if 'Win Rate' in line:
            result.win_rate = float(line.split()[-1][1:-1])

    return result


def runGame(pacmanAgent, ghostAgent, layout, repetitions):
    """Run Pacman game with configured agents and layouts.

    Run Pacman game with configured agents and layouts,
    and parse its results.

    Args:
        pacmanAgent (str): Name of the Pacman agent for running the tests.
        ghostAgent (str): Name of the Ghost agent for running the tests.
        layout (str): Name of the Layout agent for running the tests.
        repetitions (str): Number of games for the tests in string format for
        argument processing.
    """
    print 'Running game'

    processArgs = [
        'python', 'pacman.py',
        '-p', pacmanAgent,
        '-g', ghostAgent,
        '-l', layout,
        '-n', repetitions,
        '-q'
    ]
    game_name = '%s_%s_%s_%s' % (pacmanAgent, ghostAgent, layout, repetitions)
    proc = subprocess.Popen(processArgs, stdout=subprocess.PIPE,
                            cwd='./berkeley/')
    result = parseOutput(proc.stdout, game_name)
    print result

if __name__ == '__main__':
    game_definitions = [
        ['LeftTurnAgent', 'RandomGhost', 'originalClassic', '3'],
        ['GreedyAgent', 'RandomGhost', 'originalClassic', '3'],
    ]

    for game_definition in game_definitions:
        print game_definition
        runGame(*game_definition)
