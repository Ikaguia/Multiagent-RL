#!/usr/bin/env python
#  -*- coding: utf-8 -*-
"""Define features used by behavior learning agents."""

__author__ = "Matheus Portela and Guilherme N. Ramos"
__credits__ = ["Matheus Portela", "Guilherme N. Ramos", "Renato Nobre",
			   "Pedro Saman", "Cristiano K. Brust"]
__maintainer__ = "Guilherme N. Ramos"
__email__ = "gnramos@unb.br"


class Feature(object):
	"""Superclass for other features."""

	def __call__(self, state, action):
		"""Basic class __call__ method.

		This is only called when a subclass feature does not implement call.

		Args:
			state: The current game state.
			action: The action being made.
		Raises:
			NotImplementedError: A feature must implement __call__.
		"""
		raise NotImplementedError('Feature must implement __call__')


class EnemyDistanceFeature(Feature):
	"""Calculate the distance to the enemy."""

	def __init__(self, enemy_id):
		"""Constructor for the EnemyDistanceFeature Class.

		Args:
			enemy_id: The identifier of the enemy.
		Attributes:
			enemy_id: The identifier of the enemy.
		"""
		self.enemy_id = enemy_id

	def __call__(self, state, action):
		"""Method executed when EnemyDistanceFeature Class is called.

		Args:
			state: Agent state.
			action: Action done for the agent state.
		Returns:
			distance (float): The distance between the agent and the enemy.
		"""
		my_position = state.get_position()
		enemy_position = state.get_agent_position(self.enemy_id)
		distance = state.calculate_distance(my_position, enemy_position)

		if distance == 0.0:
			distance = 1.0

		distance = (1.0/distance)
		return distance


class FoodDistanceFeature(Feature):
	"""Calculate the distance between the agent and the food."""

	def __call__(self, state, action):
		"""Method executed when FoodDistanceFeature Class is called.

		Args:
			state: Agent state.
			action: Action done for the agent state.
		Returns:
			distance (float): The distance between the agent and the food.
		"""
		distance = state.get_food_distance()

		if distance == 0.0:
			distance = 1.0

		distance = (1.0 / distance)
		return distance


class FragileAgentFeature(Feature):
	"""Get the Fragile Agent for an agent identifier.

	A fragile agent is the when a pacman eat a pill and might be on danger.
	"""

	def __init__(self, agent_id):
		"""Constructor method for the FragileAgentFeature Class.

		Args:
			agent_id: The identifier of the agent.
		Attributes:
			agent_id: The identifier of the agent.
		"""
		self.agent_id = agent_id

	def __call__(self, state, action):
		"""Method executed when FragileAgentFeature Class is called.

		Args:
			state: Agent state.
			action: Action done for the agent state.
		Returns:
			Fragile Agent for the agent id.
		"""
		return state.get_fragile_agent(self.agent_id)


class AllybehaviorFeature(Feature):
	"""Gets if any ally was using 'behavior' in the last state."""

	def __init__(self, behavior, ally_ids):
		"""Constructor method for the AllybehaviorFeature Class.

		Args:
			behavior: The behavior to check for.
			ally_ids: A list of agent ids to check the behavior.
		Attributes:
			behavior: The behavior to check for.
			ally_ids: A list of agent ids to check the behavior.
		"""
		self.behavior = behavior
		self.ally_ids = ally_ids

	def __call__(self, state, action):
		"""Method executed when AllybehaviorFeature Class is called.

		Args:
			state: Agent state.
			action: Action done for the agent state.
		Returns:
			Boolean checking if any ally was using 'behavior' in the last state
		"""
		for ally in self.ally_ids:
			if state.get_agent_prev_behavior(ally) == self.behavior:
				return True
		return False


class AllybehaviorOldFeature(Feature):
	"""Gets which behavior 'ally' was using in the last state."""

	def __init__(self, ally, behavior_list):
		"""Constructor method for the AllybehaviorOldFeature Class.

		Args:
			ally: The id of the ally to check
			behavior_list: The behaviors to check
		Attributes:
			ally: The id of the ally to check
			behavior_list: The behaviors to check
		"""
		self.ally = ally
		self.behavior_list = behavior_list

	def __call__(self, state, action):
		"""Method executed when AllybehaviorFeature Class is called.

		Args:
			state: Agent state.
			action: Action done for the agent state.
		Returns:
			Integer id of the behavior 'ally' was using last state
		"""
		for i in xrange(len(self.behavior_list)):
			if state.get_agent_prev_behavior(self.ally) == self.behavior_list[i]:
				return i
		return len(self.behavior_list)



class AllyEnemyDistanceFeature(Feature):
	"""Calculate the distance from the enemy to the closest ally."""

	def __init__(self, enemy_id, ally_ids):
		"""Constructor for the AllyEnemyDistanceFeature Class.

		Args:
			enemy_id: The identifier of the enemy.
			ally_ids: The identifiers of the allies
		Attributes:
			enemy_id: The identifier of the enemy.
			ally_ids: The identifiers of the allies
		"""
		self.enemy_id = enemy_id
		self.ally_ids = ally_ids

	def __call__(self, state, action):
		"""Method executed when AllyEnemyDistanceFeature Class is called.

		Args:
			state: Agent state.
			action: Action done for the agent state.
		Returns:
			distance (float): The distance between the agent and the enemy.
		"""
		best_distance = None
		for ally_id in self.ally_ids:
			ally_position = state.get_agent_position(ally_id)
			enemy_position = state.get_agent_position(self.enemy_id)
			distance = state.calculate_distance(ally_position, enemy_position)
			if distance == 0.0: distance = 1.0
			distance = (1.0/distance)
			if not best_distance or distance > best_distance: best_distance = distance

		return best_distance


class ClosestToEnemyFeature(Feature):
	"""Calculate if 'agent' is the closest to the enemy."""

	def __init__(self, enemy_id, ally_ids):
		"""Constructor for the ClosestToEnemyFeature Class.

		Args:
			enemy_id: The identifier of the enemy.
			ally_ids: The identifiers of the allies
		Attributes:
			enemy_id: The identifier of the enemy.
			ally_ids: The identifiers of the allies
		"""
		self.enemy_id = enemy_id
		self.ally_ids = ally_ids

	def __call__(self, state, action):
		"""Method executed when ClosestToEnemyFeature Class is called.

		Args:
			state: Agent state.
			action: Action done for the agent state.
		Returns:
			closest (bool): Is the agent the closest to the enemy.
		"""

		def dist(agent_id):
			agent_position = state.get_agent_position(agent_id)
			enemy_position = state.get_agent_position(self.enemy_id)
			distance = state.calculate_distance(agent_position, enemy_position)
			if distance == 0.0: distance = 1.0
			distance = (1.0/distance)
			return distance

		my_distance = dist(state.agent_id)
		for ally_id in self.ally_ids:
			if dist(ally_id) < my_distance:
				return False
		return True


