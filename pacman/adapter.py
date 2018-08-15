#!/usr/bin/env python
#  -*- coding: utf-8 -*-
"""Adapts Communication.

Adapts Communication. between controller and the Berkeley Pac-man simulator.

Attributes:
	DEFAULT_GHOST_AGENT: The default ghost agent, 'ai'.
	DEFAULT_GHOST_AGENT2: The default ghost agent for the agent 2, 'a1'.
	DEFAULT_GHOST_AGENT3: The default ghost agent for the agent 3, 'a1'.
	DEFAULT_GHOST_AGENT4: The default ghost agent for the agent 4, 'a1'.
	DEFAULT_LAYOUT: The default map layout, 'classic'.
	DEFAULT_NUMBER_OF_GHOSTS: The default number of ghosts, 3.
	DEFAULT_NUMBER_OF_LEARNING_RUNS: The default number of learning runs 100.
	DEFAULT_NUMBER_OF_TEST_RUNS: The default number of test runs, 15.
	DEFAULT_OUTPUT_FILE: The default output file, 'results.txt'.
	DEFAULT_PACMAN_AGENT: The default pacman agent, 'random'.
	NUMBER_OF_BERKELEY_GAMES: Pacman game configuration of Berkeley, 1.
	RECORD_BERKELEY_GAMES: Pacman game configuration of Berkeley, False.
	DEFAULT_COMM: Type of communication, default is none.
"""

import pickle
import os

from berkeley.graphicsDisplay import PacmanGraphics as BerkeleyGraphics
from berkeley.layout import getLayout as get_berkeley_layout
from berkeley.pacman import runGames as run_berkeley_games
from berkeley.textDisplay import NullGraphics as BerkeleyNullGraphics

import agents
import communication as comm

import cliparser

import random

__author__ = "Matheus Portela and Guilherme N. Ramos"
__credits__ = ["Matheus Portela", "Guilherme N. Ramos", "Renato Nobre",
			   "Pedro Saman", "Cristiano K. Brust"]
__maintainer__ = "Guilherme N. Ramos"
__email__ = "gnramos@unb.br"


# Default settings (CLI parsing)
DEFAULT_GHOST_AGENT = 'ai'
DEFAULT_GHOST_AGENT2 = 'a1'
DEFAULT_GHOST_AGENT3 = 'a1'
DEFAULT_GHOST_AGENT4 = 'a1'
DEFAULT_LAYOUT = 'classic'
DEFAULT_NUMBER_OF_GHOSTS = 3
DEFAULT_NUMBER_OF_LEARNING_RUNS = 100
DEFAULT_NUMBER_OF_TEST_RUNS = 15
DEFAULT_OUTPUT_FILE = 'results.txt'
DEFAULT_PACMAN_AGENT = 'random'
DEFAULT_COMM = 'none'
DEFAULT_MSE = 0

# Pac-Man game configuration
NUMBER_OF_BERKELEY_GAMES = 1
RECORD_BERKELEY_GAMES = False


import sys
import time;
def log(msg):
	"""Print adapter message."""
	localtime = time.asctime( time.localtime(time.time()) )
	print localtime,'[  Adapter ] {}'.format(msg)
	sys.stdout.flush()


class Adapter(object):
	"""Some Stuff.

	Attributes:
		layout: The initialized layout.
		pacman_class: The initialized pacman.
		num_ghosts: The initialized number of ghosts.
		learn_runs: The initialized number of learning runs.
		test_runs: The initialized number of test runs.
		display: The graphics of the simulations if graphics were set to
			'True'.
		policy_file: The name of the policy_file.
		all_agents: The pacman and the ghosts agents.
		ghost_class: The initialized ghosts.
		ghosts: The identifier of all the ghosts
		pacman: Instance of PacmanAdapterAgent.
	Todo:
		* Define pacman-agent choices and ghost-agent choices from agents.py
			file.
		* Parse arguments outside class, pass values as arguments for
			constructor.
	"""

	def __init__(self,
				 pacman_agent=DEFAULT_PACMAN_AGENT,
				 ghost_agent=DEFAULT_GHOST_AGENT,
				 ghost_agent2=DEFAULT_GHOST_AGENT2,
				 ghost_agent3=DEFAULT_GHOST_AGENT3,
				 ghost_agent4=DEFAULT_GHOST_AGENT4,
				 num_ghosts=DEFAULT_NUMBER_OF_GHOSTS,
				 noise=agents.DEFAULT_NOISE,
				 policy_file=None,
				 layout=DEFAULT_LAYOUT,
				 learn_runs=DEFAULT_NUMBER_OF_LEARNING_RUNS,
				 test_runs=DEFAULT_NUMBER_OF_TEST_RUNS,
				 client=None,
				 output_file=DEFAULT_OUTPUT_FILE,
				 graphics=False,
				 comm=DEFAULT_COMM,
				 mse=DEFAULT_MSE,
				 random_seed=None):
		"""Constructor for the Adapter class.

		Setup the layout, the pacman agent, the ghosts agents, the policy file,
		the number of learning and tests, the output file and the graphics
		activation for the runs.

		Args:
			pacman_agent: Pacman agent to be initialized,
				default is 'random'.
			ghost_agent: Ghost agent to be initialized, default is 'ai'.
			num_ghosts: Number of ghosts to be initialized, default is 3.
			noise: Noise, default is 0.
			policy_file: Name of the file for saving or loading policies,
				default is None.
			layout: Layout of the game, default is 'Classic'.
			learn_runs: Number of learing simulations, default is 100.
			test_runs: Number of test simulations, default is 15.
			client: Client address to connect to adapter
			output_file: File for saving the simulations results, default
				is 'output.txt'.
			graphics: Enable or disable the simulations graphics, default
				is 'False'.
		Raises:
			ValueError: Layout file missing.
			ValueError: Pac-Man agent does not exist.
			ValueError: Unexpected ghosts number.
			ValueError: Ghost agent does not exist.
			ValueError: Unexpected number of learing simulations.
			ValueError: Unexpected number of test simulations.
		"""

		if(random_seed): random.seed(random_seed)
		agents.NOISE = noise
		# Setup layout
		LAYOUT_PATH = 'pacman/layouts'
		file_name = str(num_ghosts) + 'Ghosts'
		layout_file = '/'.join([LAYOUT_PATH, layout, file_name])
		self.layout = get_berkeley_layout(layout_file)
		if not self.layout:
			raise ValueError('Layout {} missing.'.format(layout_file))
		log('Loaded {}.'.format(layout_file))

		# Setup Pac-Man agent
		if pacman_agent == 'random':
			self.pacman_class = agents.RandomPacmanAgent
		elif pacman_agent == 'random2':
			self.pacman_class = agents.RandomPacmanAgentTwo
		elif pacman_agent == 'ai':
			self.pacman_class = agents.BehaviorLearningPacmanAgent
		elif pacman_agent == 'eater':
			self.pacman_class = agents.EaterPacmanAgent
		elif pacman_agent == 'BFS':
			self.pacman_class = agents.BFS_PacmanAgent
		elif pacman_agent == 'fleet':
			self.pacman_class = agents.FleetPacmanAgent
		else:
			raise ValueError
			('Pac-Man agent must be ai, random, random2, eater, BFS or fleet')

		self.pacman = agents.PacmanAdapterAgent(client=client)
		log('Created {} #{}.'.format(self.pacman_class.__name__,
									 self.pacman.agent_id))
		self.__register_agent__(self.pacman, 'pacman', self.pacman_class)
		self.client = client
		# Setup Communication

		if comm == 'pm':
			self.comm = 'pm'
		elif comm == 'sharedLearn':
			self.comm = 'sharedLearn'
		elif comm == 'both':
			self.comm = 'both'
		elif comm == 'none':
			self.comm = 'none'
		elif comm == 'mse':
			self.comm = 'mse'
		else:
			raise ValueError
			('Communication type must be none, pm, state or both')

		log('Communication defined to {}'.format(self.comm))

		if mse == 1:
			self.mse = True
		elif mse == 0:
			self.mse = False
		else:
			raise ValueError('MSE must be 1 for True or 0 for False.')

		# print comm

		# Setup Ghost agents
		self.num_ghosts = int(num_ghosts)
		if not (1 <= self.num_ghosts <= 4):
			raise ValueError('Must 1-4 ghost(s).')

		def get_ghost_class_agent(ghost_class, agent_id, ghost1_agent=None):
			if ghost_class == 'random':
				return agents.RandomGhostAgent
			elif ghost_class == 'ai':
				return agents.BehaviorLearningGhostAgent
			elif ghost_class == 'ai2':
				return agents.BehaviorLearningGhostAgentTwo
			elif ghost_class == 'ai3':
				return agents.BehaviorLearningGhostAgentThree
			elif ghost_class == 'ai4':
				return agents.BehaviorLearningGhostAgentFour
			elif ghost_class == 'ai5':
				return agents.BehaviorLearningGhostAgentFive
			elif ghost_class == 'fixedFlee':
				return agents.FixedFleeGhostAgent
			elif ghost_class == 'fixedSeek':
				return agents.FixedSeekGhostAgent
			elif ghost_class == 'fixedPursue':
				return agents.FixedPursueGhostAgent
			else:
				if ghost1_agent:
					if ghost_class == 'a1':
						return ghost1_agent
					else:
						raise ValueError('Ghost agent must be ai, ai2, ai3, ai4, ai5, random, fixedFlee, fixedSeek or fixedPursue.')
				else:
					raise ValueError('Ghost agent must be a1, ai, ai2, ai3, ai4, ai5, random, fixedFlee, fixedSeek or fixedPursue.')

		self.ghost_class  = [get_ghost_class_agent(ghost_agent, 1)]
		self.ghost_class += [get_ghost_class_agent(ghost_agent2, 2, self.ghost_class[0])]
		self.ghost_class += [get_ghost_class_agent(ghost_agent3, 3, self.ghost_class[0])]
		self.ghost_class += [get_ghost_class_agent(ghost_agent4, 4, self.ghost_class[0])]

		self.ghosts = []
		for x in xrange(num_ghosts):
			ghost_name = self.ghost_class[x].__name__
			ghost = agents.GhostAdapterAgent(x + 1, client=client,
											 comm=self.comm, mse=self.mse)
			log('Created {} #{}.'.format(ghost_name, ghost.agent_id))
			self.__register_agent__(ghost, 'ghost', self.ghost_class[x])
			self.ghosts.append(ghost)

		self.all_agents = [self.pacman] + self.ghosts

		# Setup policy file
		self.policy_file = str(policy_file) if policy_file else None

		# Setup MSE
		self.mseCount = 0
		self.mseCounters = []
		for ghost in self.ghosts:
			self.mseCounters.append(0)

		# Setup runs
		self.learn_runs = int(learn_runs)
		if self.learn_runs < 1:
			raise ValueError('Number of learning runs must be at least 1.')

		self.test_runs = int(test_runs)
		if self.test_runs < 1:
			raise ValueError('Number of test runs must be at least 1.')

		self.output_file = str(output_file)

		if graphics:
			self.display = BerkeleyGraphics()
		else:
			self.display = BerkeleyNullGraphics()

		log('Ready')

	def __initialize__(self, agent):
		"""Request the initialization message for the agent id.

		Args:
			agent: The agent to initialize communication.
		"""
		msg = comm.RequestInitializationMessage(agent_id=agent.agent_id)
		agent.communicate(msg)

	def __get_behavior_count__(self, agent):
		"""Request the behavior count from the agent id.

		Args:
			agent: The agent to get the behavior.
		Returns:
			The count of reply messages.
		"""
		msg = comm.RequestBehaviorCountMessage(agent_id=agent.agent_id)
		reply_msg = agent.communicate(msg)
		return reply_msg.count

	def __get_policy__(self, agent):
		"""Request the policy from the agent id.

		Args:
			agent: The agent to get the policy.
		Returns:
			The policy replied.
		"""
		msg = comm.RequestPolicyMessage(agent.agent_id)
		reply_msg = agent.communicate(msg)
		return reply_msg.policy

	def __load_policy__(self, agent, policy):
		"""Pass the policy message of the agent id.

		Args:
			agent: The agent to load the policy.
			policy: The policy to load to the agent.
		Returns:
			The receive of the communication message
		"""
		msg = comm.PolicyMessage(agent_id=agent.agent_id, policy=policy)
		return agent.communicate(msg)

	def __load_policies_from_file__(self, filename):
		"""Load policies from file.

		Args:
			filename: Name of the file to load policies from.
		"""
		policies = {}
		if filename and os.path.isfile(filename):
			log('Loading policies from {}.'.format(filename))
			with open(filename) as f:
				policies = pickle.loads(f.read())
		return policies

	def __log_behavior_count__(self, agent, results):
		"""Log the behaviors.

		Log the behaviors and the respective counts for each one of the agents

		Args:
			agent: The agent which will log the behavior.
			results: The results of the behaviors count.
		"""
		behavior_count = self.__get_behavior_count__(agent)

		for behavior, count in behavior_count.items():
			if behavior not in results['behavior_count'][agent.agent_id]:
				results['behavior_count'][agent.agent_id][behavior] = []
			results['behavior_count'][agent.agent_id][behavior].append(count)

		log('{} behavior count: {}.'.format(type(agent).__name__,
											behavior_count))

	def __update_mse_pm_count__(self, agent=None):
		"""..."""
		msg = comm.RequestMSECountMessage()
		reply_msg = agent.communicate(msg)

		self.mseCount += reply_msg.mse

	def __update_mse_count__(self, agent=None):
		"""..."""
		msg = comm.RequestMSEMessage(agent=agent.agent_id)
		reply_msg = agent.communicate(msg)

		self.mseCounters[agent.agent_id-1] += reply_msg.mse

	def __process_game__(self, policies, results):
		"""Process the game.

		Start new game, load policies to agents, update agentes rewards, log
		the behavior count and log score.

		Args:
			policies: The policies to be loaded.
			results: The results to be logged.
		"""
		# Start new game
		for agent in self.all_agents:
			agent.start_game(self.layout)

		# Load policies to agents
		if self.policy_file:
			for agent in self.all_agents:
				if agent.agent_id in policies:
					self.__load_policy__(agent, policies[agent.agent_id])

		log('Simulating game...')
		simulated_game = run_berkeley_games(self.layout, self.pacman,
											self.ghosts, self.display,
											NUMBER_OF_BERKELEY_GAMES,
											RECORD_BERKELEY_GAMES)[0]

		# Do this so as agents can receive the last reward
		for agent in self.all_agents:
			agent.update(simulated_game.state)

		"""Todo:
			* This as one list, probably by checking if agent is
				instance of BehaviorLearningAgent (needs refactoring).
		"""

		if self.mse is True:
			for ghost in self.ghosts:
				self.__update_mse_count__(agent=ghost)

		elif self.comm == 'mse':
			self.__update_mse_pm_count__(self.pacman)

		# Log behavior count
		if self.pacman_class == agents.BehaviorLearningPacmanAgent:
			self.__log_behavior_count__(self.pacman, results)

		for ghost in self.ghosts:
			if issubclass(self.ghost_class[ghost.agent_id-1],agents.BehaviorLearningGhostAgent):
				self.__log_behavior_count__(ghost, results)

		# Log score
		return simulated_game.state.getScore()

	def __register_agent__(self, agent, agent_team, agent_class):
		"""Request register message.

		Request register message for the agent id and log its action.

		Args:
			agent: The agent to register.
			agent_team: The agent team.
			agent_class: The agent class.
		Returns:
			The receive of the communication message.
		"""
		log('Request register for {} #{}.'.format(agent_class.__name__,
												  agent.agent_id))
		msg = comm.RequestRegisterMessage(agent_id=agent.agent_id,
										  agent_team=agent_team,
										  agent_class=agent_class)
		return agent.communicate(msg)

	def __save_policies__(self, policies):
		"""Save the policies from pacman and ghosts in the policy_file.

		Args:
			policies: The ghosts and pacman policies.
		Todo:
			* Keep policy in agent?
		"""
		if self.pacman_class == agents.BehaviorLearningPacmanAgent:
			policies[self.pacman.agent_id] = self.__get_policy__(self.pacman)

		for ghost in self.ghosts:
			if issubclass(self.ghost_class[ghost.agent_id-1],agents.BehaviorLearningGhostAgent):
				policies[ghost.agent_id] = self.__get_policy__(ghost)

		self.__write_to_file__(self.policy_file, policies)

	def __write_to_file__(self, filename, content):
		"""Write content to a file.

		Args:
			filename: Name of the file.
			content: content to be writen on the file.
		"""
		with open(filename, 'w+') as f:
			f.write(pickle.dumps(content))

	def run(self):
		"""Run the simulations.

		Load policies from file, initialize agents, process the game and save
		policies in file.
		"""
		log('Now running')

		results = {'learn_scores': [], 'test_scores': [], 'behavior_count': {}}

		"""Todo:
			* This as one list, probably by checking if agent is instance of
				BehaviorLearningAgent (needs refactoring).
		"""
		# Initialize Results
		if self.pacman_class == agents.BehaviorLearningPacmanAgent:
			results['behavior_count'][self.pacman.agent_id] = {}

		for ghost in self.ghosts:
			if issubclass(self.ghost_class[ghost.agent_id-1],agents.BehaviorLearningGhostAgent):
				results['behavior_count'][ghost.agent_id] = {}

		# Load policies from file
		policies = self.__load_policies_from_file__(self.policy_file)

		# Initialize agents
		for agent in self.all_agents:
			self.__initialize__(agent)

		for x in xrange(self.learn_runs):
			log('LEARN game {} (of {})'.format(x + 1, self.learn_runs))

			score = self.__process_game__(policies, results)
			results['learn_scores'].append(score)

		for agent in self.all_agents:
			agent.enable_test_mode()

		for x in xrange(self.test_runs):
			log('TEST game {} (of {})'.format(x + 1, self.test_runs))

			score = self.__process_game__(policies, results)
			results['test_scores'].append(score)

		if self.policy_file:
			self.__save_policies__(policies)

		if self.mse is True:
			total = 0
			for counter in self.mseCounters:
				total += (counter/(self.learn_runs+self.test_runs))
				log('Agent MSE: {}'.format(counter/(self.learn_runs +
													self.test_runs)))
			log('Total mse: {}'.format(total/3))
		elif self.comm == 'mse':
			log('Mean Square Error {}'.format(self.mseCount/(self.learn_runs +
															 self.test_runs)))
		log('Learn scores: {}'.format(results['learn_scores']))
		log('Test scores: {}'.format(results['test_scores']))

		self.__write_to_file__(self.output_file, results)

if __name__ == '__main__':
	try:
		adapter = cliparser.get_Adapter()
		adapter.run()
	except KeyboardInterrupt:
		print '\n\nInterrupted execution\n'
