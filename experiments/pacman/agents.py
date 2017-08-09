#  -*- coding: utf-8 -*-

"""Define the agents."""

from __future__ import absolute_import

import logging
import random

from .berkeley.game import Agent as BerkeleyGameAgent, Directions

from . import behaviors
from . import features
from multiagentrl import core
from multiagentrl import exploration
from multiagentrl import learning


# Logging configuration
logger = logging.getLogger(__name__)


# Berkeley simulator actions
GHOST_ACTIONS = [Directions.NORTH, Directions.SOUTH, Directions.EAST,
                 Directions.WEST]
PACMAN_ACTIONS = GHOST_ACTIONS + [Directions.STOP]

# Berkeley simulator indices
PACMAN_INDEX = 0


class PacmanAgent(core.BaseControllerAgent):
    def __init__(self, agent_id):
        assert agent_id == PACMAN_INDEX
        super(PacmanAgent, self).__init__(agent_id)

    def start_game(self):
        pass

    def finish_game(self):
        pass


class GhostAgent(core.BaseControllerAgent):
    def __init__(self, agent_id):
        assert agent_id != PACMAN_INDEX
        super(GhostAgent, self).__init__(agent_id)

    def start_game(self):
        pass

    def finish_game(self):
        pass


class RandomPacmanAgent(PacmanAgent):
    """Agent that randomly selects an action."""
    def __init__(self, agent_id, ally_ids, enemy_ids):
        super(RandomPacmanAgent, self).__init__(agent_id)

    def learn(self, state, action, reward):
        pass

    def act(self, state, legal_actions, explore):
        if legal_actions:
            return random.choice(legal_actions)
        else:
            return Directions.STOP


class RandomPacmanAgentTwo(PacmanAgent):
    """Agent that after choosing a random direction will follow that direction
    until it reaches a wall or have more than three possible moves. In these
    case, continue to follow the previous directions have twice the chance of
    happening then the other possible movements"""
    def __init__(self, agent_id, ally_ids, enemy_ids):
        super(RandomPacmanAgentTwo, self).__init__(agent_id)
        self.last_action = None

    def learn(self, state, action, reward):
        self.last_action = action

    def act(self, state, legal_actions, explore):
        if self.last_action == 'Stop' or self.last_action not in legal_actions:
            if 'Stop' in legal_actions:
                legal_actions.remove('Stop')
            if len(legal_actions) > 0:
                return random.choice(legal_actions)
        else:
            if len(legal_actions) > 3:
                if len(legal_actions) == 4:
                    number = random.choice([1, 2, 3, 4, 5])
                else:
                    number = random.choice([1, 2, 3, 4, 5, 6])
                if number == 1 or number == 2:
                    return self.last_action
                else:
                    aux = 3
                    legal_actions.remove(self.last_action)
                    for possible_action in legal_actions:
                        if number == aux:
                            return possible_action
                        else:
                            aux += 1
                    else:
                        return random.choice(legal_actions)
            else:
                return self.last_action

class RandomGhostAgent(GhostAgent):
    """Agent that randomly selects an action."""
    def __init__(self, agent_id, ally_ids, enemy_ids):
        super(RandomGhostAgent, self).__init__(agent_id)

    def learn(self, state, action, reward):
        pass

    def act(self, state, legal_actions, explore):
        if legal_actions:
            return random.choice(legal_actions)
        else:
            return Directions.STOP


class SeekerGhostAgent(GhostAgent):
    """Agent that randomly selects an action."""
    def __init__(self, agent_id, ally_ids, enemy_ids):
        super(SeekerGhostAgent, self).__init__(agent_id)
        self.behavior = behaviors.SeekBehavior()

    def learn(self, state, action, reward):
        pass

    def act(self, state, legal_actions, explore):
        action = self.behavior(state, legal_actions)

        if action in legal_actions:
            return action
        elif legal_actions:
            return random.choice(legal_actions)
        else:
            return Directions.STOP


class BFSPacmanAgent(PacmanAgent):
    """Agent that search for the shortest food using BFS algorithm."""
    def __init__(self, agent_id, ally_ids, enemy_ids):
        super(BFSPacmanAgent, self).__init__(agent_id)

    def learn(self, state, action, reward):
        pass

    def act(self, state, legal_actions, explore):
        """Choose the action that brigs Pacman to the neartest food.
        Args:
            state: Current game state.
            action: Last executed action.
            reward: Reward for the previous action.
            legal_actions: List of currently allowed actions.
            explore: Boolean whether agent is allowed to explore.
        Returns:
            Sugested action
        """
        queue = Queue.Queue()
        visited = []

        initial_position = state.get_position()

        food_map = state.food_map

        agent_map = state.get_map()

        queue.put(initial_position)
        visited.append(initial_position)

        closest_food = None
        while not queue.empty():

            if(closest_food is not None):
                break

            current_edge = queue.get()
            (k, l) = current_edge

            random.shuffle(PACMAN_ACTIONS)
            for actions in PACMAN_ACTIONS:

                diff = agent_map.action_to_pos[actions]
                new_edge = (k + diff[0],
                            l + diff[1])

                if agent_map._is_valid_position(new_edge):
                    if new_edge not in visited:
                        (i, j) = new_edge
                        if food_map[i][j] > 0.0:
                            if closest_food is None:
                                closest_food = new_edge
                        else:
                            queue.put(new_edge)
                            visited.append(new_edge)

        if closest_food is None:
            return Directions.STOP

            return random.choice(legal_actions)

        best_action = None
        min_dist = float('inf')

        (f, p) = (0, 0)

        for actions in legal_actions:

            diff = agent_map.action_to_pos[actions]
            new_edge = (initial_position[0] + diff[0],
                        initial_position[1] + diff[1])

            new_dist = state.calculate_distance(new_edge, closest_food)

            if new_dist <= min_dist:
                min_dist = new_dist
                best_action = actions
                (f, p) = new_edge

        food_map[f][p] = 0.0
        return best_action


class FleetPacmanAgent(PacmanAgent):
    """Pacman that run away from ghosts and get food.

    Attributes:
        agent_id: The identifier of an agent.
        ally_ids: The identifier of all allies agents.
        enemy_ids: The identifier of all enemies agents.
    """

    def __init__(self, agent_id, ally_ids, enemy_ids):
        """Extend the constructor from the PacmanAgent superclass.

        Args:
            agent_id: The identifier of an agent.
            ally_ids: The identifier of all allies agents.
            enemy_ids: The identifier of all enemies agents.
        """
        super(FleetPacmanAgent, self).__init__(agent_id)
        self.eat_behavior = behaviors.EatBehavior()

    def learn(self, state, action, reward):
        pass

    def act(self, state, legal_actions, explore):
        """Choose the best action.

        Args:
            state: Current game state.
            action: Last executed action.
            reward: Reward for the previous action.
            legal_actions: List of currently allowed actions.
            test: Boolean whether agent is allowed to explore.
        """
        agent_map = state.get_map()
        (x, y) = state.get_position()

        nearby_enemies = []
        enemies_locations = []
        fragile_enemies_position = []

        for p in state.enemy_ids:
            q = state.get_agent_position(p)
            enemies_locations.append(q)
            if state.get_fragile_agent(p):
                fragile_enemies_position.append(q)
                FragileFlag = True
            else:
                FragileFlag = False

        for enemy_position in enemies_locations:
            distance = state.calculate_distance((x, y), enemy_position)
            if distance < 4:
                nearby_enemies.append(enemy_position)

        if len(nearby_enemies) == 0:
            suggested_action = self.eat_behavior(state, legal_actions)
            if suggested_action in legal_actions:
                return suggested_action
            elif legal_actions == []:
                return Directions.STOP
            else:
                return random.choice(legal_actions)

        elif FragileFlag is True:
            min_distance = float('inf')
            best_action = None
            for enemie in fragile_enemies_position:
                for actions in legal_actions:
                    diff = agent_map.action_to_pos[actions]
                    new_position = (diff[0]+x, diff[1]+y)
                    new_distance = state.calculate_distance(new_position,
                                                            enemie)
                    if(new_distance < min_distance):
                        min_distance = new_distance
                        best_action = actions
            if(best_action is not None):
                return best_action

        else:
            max_distance = (-1)*float('inf')
            best_action = None
            for actions in legal_actions:
                new_distance = 0
                for enemie in nearby_enemies:
                    diff = agent_map.action_to_pos[actions]
                    new_position = (diff[0]+x, diff[1]+y)
                    new_distance += state.calculate_distance(new_position,
                                                             enemie)
                if new_distance > max_distance:
                    max_distance = new_distance
                    best_action = actions
            return best_action


class EaterPacmanAgent(PacmanAgent):
    def __init__(self, agent_id, ally_ids, enemy_ids):
        super(EaterPacmanAgent, self).__init__(agent_id)
        self.eat_behavior = behaviors.EatBehavior()

    def learn(self, state, action, reward):
        pass

    def act(self, state, legal_actions, explore):
        if not legal_actions:
            return Directions.STOP

        suggested_action = self.eat_behavior(state, legal_actions)

        if suggested_action in legal_actions:
            return suggested_action
        else:
            return random.choice(legal_actions)


class TDLearningPacmanAgent(PacmanAgent):
    def __init__(self, agent_id, ally_ids, enemy_ids, learning_algorithm,
                 exploration_algorithm):
        super(TDLearningPacmanAgent, self).__init__(agent_id)
        self.game_number = 1
        self.game_step = 1
        self.exploration_rate = 0.1

        if enemy_ids:
            self.features = [
                features.ClosestEnemyDistanceFeature(enemy_ids),
                features.FragileAgentFeature(enemy_ids[0])
            ]
        else:
            self.features = []

        self.learning = learning_algorithm
        self.exploration = exploration_algorithm
        self.agent_state = None

    def get_policy(self):
        return self.learning.q_values

    def set_policy(self, weights):
        self.learning.q_values = weights

    def start_game(self):
        self.game_step = 1

    def finish_game(self):
        self.game_number += 1

    def _get_state(self, state):
        if not self.agent_state:
            self.agent_state = tuple([
                feature(state) for feature in self.features])
        return self.agent_state

    def learn(self, state, action, reward):
        self.agent_state = self._get_state(state)
        self.learning.learn(self.agent_state, action, reward)

    def act(self, state, legal_actions, explore):
        if not legal_actions:
            return Directions.STOP

        self.agent_state = self._get_state(state)
        action = self.learning.act(self.agent_state)

        if explore and legal_actions:
            action = self.exploration.explore(action, legal_actions)

        self.game_step += 1

        # Reset agent state
        self.agent_state = None

        return self._select_valid_action(action, legal_actions)

    def _select_valid_action(self, action, legal_actions):
        if action in legal_actions:
            return action
        else:
            return random.choice(legal_actions)


class QLearningPacmanAgent(TDLearningPacmanAgent):
    def __init__(self, agent_id, ally_ids, enemy_ids):
        super(QLearningPacmanAgent, self).__init__(
            agent_id, ally_ids, enemy_ids,
            learning.QLearning(learning_rate=0.3, discount_factor=0.7,
                               actions=PACMAN_ACTIONS),
            exploration.EGreedy(exploration_rate=0.1))


class SARSALearningPacmanAgent(TDLearningPacmanAgent):
    def __init__(self, agent_id, ally_ids, enemy_ids):
        super(SARSALearningPacmanAgent, self).__init__(
            agent_id, ally_ids, enemy_ids,
            learning.SARSALearning(learning_rate=0.5, discount_factor=0.9,
                                   actions=PACMAN_ACTIONS),
            exploration.EGreedy(exploration_rate=0.1))


class BehaviorLearningPacmanAgent(TDLearningPacmanAgent):
    def __init__(self, agent_id, ally_ids, enemy_ids, learning_algorithm,
                 exploration_algorithm):
        TDLearningPacmanAgent.__init__(
            self, agent_id, ally_ids, enemy_ids, learning_algorithm,
            exploration_algorithm)

        self.num_behavior_steps = 5
        self.previous_behavior = None
        self.K = 1.0  # Learning rate

    def learn(self, state, action, reward):
        self.agent_state = self._get_state(state)

        if self.previous_behavior:
            self.learning.learning_rate = self.K / (self.K + self.game_step)
            self.learning.learn(self.agent_state, self.previous_behavior,
                                reward)

    def act(self, state, legal_actions, explore):
        if not legal_actions:
            return Directions.STOP

        self.agent_state = self._get_state(state)

        # Select a new behavior every `num_behavior_steps` steps.
        if (self.previous_behavior is None or
                self.game_step % self.num_behavior_steps == 0):
            behavior = self.learning.act(self.agent_state)
            self.previous_behavior = behavior
        else:
            behavior = self.previous_behavior

        action = behavior(state, legal_actions)

        if explore and legal_actions:
            action = self.exploration.explore(action, legal_actions)

        self.game_step += 1

        # Reset agent state
        self.agent_state = None

        return self._select_valid_action(action, legal_actions)


class BehaviorRandomPacmanAgent(TDLearningPacmanAgent):
    def __init__(self, agent_id, ally_ids, enemy_ids):
        TDLearningPacmanAgent.__init__(
            self, agent_id, ally_ids, enemy_ids, None, None
        )

        self.behaviors = [
            behaviors.EatBehavior(),
            behaviors.FleeBehavior(),
            behaviors.SeekBehavior(),
            behaviors.PursueBehavior()
        ]
        self.num_behavior_steps = 5
        self.previous_behavior = None

    def learn(self, state, action, reward):
        pass

    def act(self, state, legal_actions, explore):
        if not legal_actions:
            return Directions.STOP

        if (self.previous_behavior is None or
                self.game_step % self.num_behavior_steps == 0):
            behavior = random.choice(self.behaviors)
            self.previous_behavior = behavior
        else:
            behavior = self.previous_behavior

        action = behavior(state, legal_actions)
        self.game_step += 1
        return self._select_valid_action(action, legal_actions)


class BehaviorQLearningPacmanAgent(BehaviorLearningPacmanAgent):
    def __init__(self, agent_id, ally_ids, enemy_ids):
        super(BehaviorQLearningPacmanAgent, self).__init__(
            agent_id, ally_ids, enemy_ids,
            learning.QLearning(
                learning_rate=0.3, discount_factor=0.7,
                actions=[
                    behaviors.EatBehavior(),
                    behaviors.FleeBehavior(),
                    behaviors.SeekBehavior(),
                ]),
            exploration.EGreedy(exploration_rate=0.1))


class BayesianBehaviorQLearningPacmanAgent(BehaviorLearningPacmanAgent):
    def __init__(self, agent_id, ally_ids, enemy_ids):
        super(BayesianBehaviorQLearningPacmanAgent, self).__init__(
            agent_id, ally_ids, enemy_ids,
            learning.QLearningWithApproximation(
                learning_rate=0.1, discount_factor=0.9,
                actions=[
                    behaviors.EatBehavior(),
                    behaviors.FleeBehavior(),
                    behaviors.SeekBehavior(),
                ]),
            exploration.EGreedy(exploration_rate=0.1))
        self.learning.features = self.features
