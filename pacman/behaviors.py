#!/usr/bin/env python
#  -*- coding: utf-8 -*-

"""Define the behaviors an agents choose."""

import random

__author__ = "Matheus Portela and Guilherme N. Ramos"
__credits__ = ["Matheus Portela", "Guilherme N. Ramos", "Renato Nobre",
               "Pedro Saman"]
__maintainer__ = "Guilherme N. Ramos"
__email__ = "gnramos@unb.br"


class Behavior(object):
    """Base class for behaviors implementation."""

    def __str__(self):
        """Define the work of str() when its called.

        Returns:
            The class name.
        """
        return self.__class__.__name__

    def __call__(self, state, legal_actions):
        """Base __call__ method.

        This method needs to be overwrited by a __call__ function on a
        subclass.

        Args:
            state: A defined state.
            legal_actions: A list of legal actions.
        Raises:
            NotImplementedError: Behavior must implement __call__.
        """
        raise NotImplementedError('Behavior must implement __call__')


class RandomBehavior(Behavior):
    """Class for the Random Behavior."""

    def __call__(self, state, legal_actions):
        """Choose a random action.

        If not legal action avaiable stay still.

        Args:
            state: A defined state.
            legal_actions: A list of legal actions.
        Returns:
            Choose a random action.
        """
        if legal_actions == []:
            return 'Stop'
        else:
            return random.choice(legal_actions)


class EatBehavior(Behavior):
    """Class for the eat behavior."""

    def __call__(self, state, legal_actions):
        """Calculate the best action to get a food.

        Args:
            state: A defined state.
            legal_actions: A list of legal actions.
        Returns:
            The best action for this behavior.
        """
        agent_position = state.get_position()
        agent_map = state.get_map()
        food_map = state.food_map
        food_prob_threshold = food_map.max() / 2.0
        best_action = None
        min_dist = None

        random.shuffle(legal_actions)

        for action in legal_actions:
            diff = agent_map.action_to_pos[action]
            new_position = (agent_position[0] + diff[0],
                            agent_position[1] + diff[1])

            for x in range(food_map.width):
                for y in range(food_map.height):
                    new_distance = state.calculate_distance(new_position,
                                                            (y, x))

                    if (best_action is None) or (food_map[y][x] >
                                                 food_prob_threshold and
                                                 new_distance < min_dist):
                        min_dist = new_distance
                        best_action = action

        return best_action


class FleeBehavior(Behavior):
    """Class for the flee behavior."""

    def __call__(self, state, legal_actions):
        """Calculate the best action to flee from the enemy.

        Args:
            state: A defined state.
            legal_actions: A list of legal actions.
        Returns:
            The best action for this behavior.
        """
        agent_position = state.get_position()
        enemy_position = state.get_agent_position
        (state.get_closest_enemy(state))

        agent_map = state.get_map()

        best_action = None
        max_distance = None

        random.shuffle(legal_actions)

        for action in legal_actions:
            diff = agent_map.action_to_pos[action]
            new_position = (agent_position[0] + diff[0],
                            agent_position[1] + diff[1])
            new_distance = state.calculate_distance
            (new_position, enemy_position)

            if (best_action is None) or (agent_map._is_valid_position
                                         (new_position) and
                                         new_distance > max_distance):
                best_action = action
                max_distance = new_distance

        return best_action


class SeekBehavior(Behavior):
    """Class for the Seek behavior."""

    def __call__(self, state, legal_actions):
        """Calculate the best action to Seek an enemy.

        Args:
            state: A defined state.
            legal_actions: A list of legal actions.
        Returns:
            The best action for this behavior.
        """
        agent_position = state.get_position()
        enemy_position = state.get_agent_position
        (state.get_closest_enemy(state))

        agent_map = state.get_map()

        best_action = None
        min_distance = None

        random.shuffle(legal_actions)

        for action in legal_actions:
            diff = agent_map.action_to_pos[action]
            new_position = (agent_position[0] + diff[0],
                            agent_position[1] + diff[1])
            new_distance = state.calculate_distance
            (new_position, enemy_position)

            if (best_action is None) or (agent_map._is_valid_position
                                         (new_position) and
                                         new_distance < min_distance):
                best_action = action
                min_distance = new_distance

        return best_action


class PursueBehavior(Behavior):
    """Class for the Seek behavior.

    Attributes:
        n: Steps that will be simulated, default 2.
        enemy_previous_position: The previous position of an enemy.
    """

    def __init__(self, n=2):
        """Constructor for the PursueBehaviorClass.

        Args:
            n: Steps that will be simulated, default = 2.
        """
        self.n = n
        self.enemy_previous_position = None

    def _estimate_enemy_future_position(self, current_position, agent_map):
        """Estimate the future position of the enemy.

        Args:
            current_position: The current position of the enemy.
            agent_map: The layout being played.
        Returns:
            enemy_position: The estimate position of the enemy.
        """
        enemy_position = current_position

        if not self.enemy_previous_position:
            enemy_diff = (0, 0)
        else:
            enemy_diff = (enemy_position[0] - self.enemy_previous_position[0],
                          enemy_position[1] - self.enemy_previous_position[1])
        self.enemy_previous_position = enemy_position
        simulated_steps = 0

        while agent_map._is_valid_position(
                enemy_position) and simulated_steps < self.n:

            enemy_position = (enemy_position[0] + enemy_diff[0],
                              enemy_position[1] + enemy_diff[1])
            simulated_steps += 1

        return enemy_position

    def __call__(self, state, legal_actions):
        """Calculate the best action to pursue an enemy.

        Args:
            state: A defined state.
            legal_actions: A list of legal actions.
        Returns:
            The best action for this behavior.
        """
        agent_map = state.get_map()
        agent_position = state.get_position()
        enemy_position = self._estimate_enemy_future_position(
            state.get_agent_position(state.get_closest_enemy(state)),
            agent_map)

        best_action = None
        min_distance = None

        random.shuffle(legal_actions)

        for action in legal_actions:
            diff = agent_map.action_to_pos[action]
            new_position = (agent_position[0] + diff[0],
                            agent_position[1] + diff[1])

            new_distance = state.calculate_distance
            (new_position, enemy_position)

            if (best_action is None) or (agent_map._is_valid_position
                                         (new_position) and
                                         new_distance < min_distance):
                best_action = action
                min_distance = new_distance

        return best_action
