# baselineTeam.py
# ---------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


# baselineTeam.py
# ---------------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

import random
import util

from captureAgents import CaptureAgent
from game import Directions
from util import nearestPoint
from math import log10, log2


#################
# Team creation #
#################

def create_team(first_index, second_index, is_red,
                first='OffensiveReflexAgent', second='DefensiveReflexAgent', num_training=0):
    """
    This function should return a list of two agents that will form the
    team, initialized using firstIndex and secondIndex as their agent
    index numbers.  isRed is True if the red team is being created, and
    will be False if the blue team is being created.

    As a potentially helpful development aid, this function can take
    additional string-valued keyword arguments ("first" and "second" are
    such arguments in the case of this function), which will come from
    the --redOpts and --blueOpts command-line arguments to capture.py.
    For the nightly contest, however, your team will be created without
    any extra arguments, so you should make sure that the default
    behavior is what you want for the nightly contest.
    """
    return [eval(first)(first_index), eval(second)(second_index)]


##########
# Agents #
##########

class ReflexCaptureAgent(CaptureAgent):
    """
    A base class for reflex agents that choose score-maximizing actions
    """

    def __init__(self, index, time_for_computing=.1):
        super().__init__(index, time_for_computing)
        self.start = None

    def register_initial_state(self, game_state):
        self.start = game_state.get_agent_position(self.index)
        CaptureAgent.register_initial_state(self, game_state)

    def choose_action(self, game_state):
        """
        Picks among the actions with the highest Q(s,a).
        """
        actions = game_state.get_legal_actions(self.index)

        # You can profile your evaluation time by uncommenting these lines
        # start = time.time()
        values = [self.evaluate(game_state, a) for a in actions]
        # print 'eval time for agent %d: %.4f' % (self.index, time.time() - start)

        max_value = max(values, default=0)
        best_actions = [a for a, v in zip(actions, values) if v == max_value]

        food_left = len(self.get_food(game_state).as_list())

        if food_left <= 2:
            best_dist = 9999
            best_action = None
            for action in actions:
                successor = self.get_successor(game_state, action)
                pos2 = successor.get_agent_position(self.index)
                dist = self.get_maze_distance(self.start, pos2)
                if dist < best_dist:
                    best_action = action
                    best_dist = dist
            return best_action

        return random.choice(best_actions)

    def get_successor(self, game_state, action):
        """
        Finds the next successor which is a grid position (location tuple).
        """
        successor = game_state.generate_successor(self.index, action)
        pos = successor.get_agent_state(self.index).get_position()
        if pos != nearestPoint(pos):
            # Only half a grid position was covered
            return successor.generate_successor(self.index, action)
        else:
            return successor

    def evaluate(self, game_state, action):
        """
        Computes a linear combination of features and feature weights
        """
        features = self.get_features(game_state, action)
        weights = self.get_weights(game_state, action)
        return features * weights

    def get_features(self, game_state, action):
        """
        Returns a counter of features for the state
        """
        features = util.Counter()
        successor = self.get_successor(game_state, action)
        features['successor_score'] = self.get_score(successor)
        return features

    def get_weights(self, game_state, action):
        """
        Normally, weights do not depend on the game state.  They can be either
        a counter or a dictionary.
        """
        return {'successor_score': 1.0}



class OffensiveReflexAgent(ReflexCaptureAgent): # DANNY: make defensive pacman be defensive if scared, make offensive pacman be offensive if scared
    """
  A reflex agent that seeks food. This is an agent
  we give you to get an idea of what an offensive agent might look like,
  but it is by no means the best or only way to build an offensive agent.
  """

    def get_features(self, game_state, action):
        features = util.Counter()
        successor = self.get_successor(game_state, action)
        food_list = self.get_food(successor).as_list()
        capsules_list = self.get_capsules(successor)
        score = self.get_score(successor)
        our_food = self.get_food_you_are_defending(successor).as_list()
        
        my_state = successor.get_agent_state(self.index)
        my_pos = my_state.get_position()
        food_eaten = my_state.num_carrying
        am_scared = my_state.scared_timer > 0
        
        enemies = [successor.get_agent_state(i) for i in self.get_opponents(successor)]
        friends = [successor.get_agent_state(i) for i in self.get_team(successor)]
        
        features['score'] = score # increase the score
        
        friend_distance = max([self.get_maze_distance(my_pos, friend.get_position()) for friend in friends], default=0)
        features['friend_dist'] = log10(friend_distance+1)
        
        if self.red and my_pos[0] == 1:
            features['stay_away_from_base'] = 1
        elif my_pos[0] == game_state.data.layout.width-2:
            features['stay_away_from_base'] = 1
            
        if self.red:
            center_width = (game_state.data.layout.width / 2) - 1
        else:
            center_width = (game_state.data.layout.width / 2)
        heights = range(1, game_state.data.layout.height)

        if score > 0: # be defensive
            
            lower_food_col = [food[0] for food in our_food if food[1] <= game_state.data.layout.height/2]
            
            if self.red:
                closest_food_col = max(lower_food_col, default=0)
                closest_foods = [food for food in our_food if food[1] <= game_state.data.layout.height/2 and food[0] == closest_food_col]
                closest_food = closest_foods[0]
            else:
                closest_food_col = min(lower_food_col, default=0)
                closest_foods = [food for food in our_food if food[1] <= game_state.data.layout.height/2 and food[0] == closest_food_col]
                closest_food = closest_foods[0]
                
            if closest_food[1] == 1:
                viable_spots =  [float(item) for item in heights if not game_state.has_wall(closest_food[0], item)]
                closest_food = (closest_food[0], sorted(viable_spots)[1])
            
            features['defend_food'] = self.get_maze_distance(my_pos, closest_food)
                
            features['on_side'] = int(not my_state.is_pacman)
            
            invaders = [a for a in enemies if a.is_pacman and a.get_position() is not None]
            
            if len(invaders) > 0:
                dists = [self.get_maze_distance(my_pos, a.get_position()) for a in invaders]
            else:
                dists = [0]
                
            if am_scared: # if scared
    
                min_dist_to_home = min([self.get_maze_distance((center_width, float(item)), my_pos) for item in heights if not game_state.has_wall(int(center_width), item)], default=0)
                
                features['min_dist_to_home'] = min_dist_to_home # go home when you're scared
                
                features['invader_distance'] = min(dists, default=0) # stay away from the invader when you're scared
            else:
                features['invader_distance'] = -min(dists, default=0) # go to the invader when you're not scared
                if features['invader_distance'] == 0:
                    features['invader_distance'] = 9999999999

            if action == Directions.STOP:
                features['stop'] = 1 # don't stop
            rev = Directions.REVERSE[game_state.get_agent_state(self.index).configuration.direction]
            
            if action == rev:
                features['reverse'] = 1 # don't reverse
                
            features['friends_dist'] = features['friends_dist']/500 # worry about distance less if playing defense
            
            # if there are no invaders, hang out near the border (north)
            
            if len(invaders) == 0:
                if self.red:
                    features['hang_by_border'] = min([self.get_maze_distance((center_width-1, float(item)), my_pos) for item in heights if not game_state.has_wall(int(center_width-1), item)], default=0)
                else:
                    features['hang_by_border'] = min([self.get_maze_distance((center_width+1, float(item)), my_pos) for item in heights if not game_state.has_wall(int(center_width+1), item)], default=0)
            
            
        else: # be offensive
            features['on_side'] = int(my_state.is_pacman)
            
            features['neg_food_list'] = -len(food_list) # decrease remaining food

            if self.red:
                center_width = (game_state.data.layout.width / 2) - 1
            else:
                center_width = (game_state.data.layout.width / 2)
            
            heights = range(1, game_state.data.layout.height)
            min_dist_to_home = min([self.get_maze_distance((center_width, float(item)), my_pos) for item in heights if not game_state.has_wall(int(center_width), item)], default=0)

            if min_dist_to_home > 0:
                inv_dist_to_home = 1/min_dist_to_home
            else:
                inv_dist_to_home = 999999
            features['full_stomach'] = (food_eaten)*inv_dist_to_home # go home when you're full, MAKE THIS EXPONENTIAL

            min_dist_to_food = min([self.get_maze_distance(my_pos, food) for food in food_list if food[1] < game_state.data.layout.height/2], default=0)
            features['distance_to_food'] = min_dist_to_food # get close to food

            enemy_distances = [self.get_maze_distance(my_pos, enemy.get_position()) for enemy in enemies if
                                not enemy.is_pacman and enemy.get_position() is not None and enemy.scared_timer == 0]
            distance_to_enemy = min(enemy_distances, default=0)
            features['distance_to_enemy'] = distance_to_enemy # stay away from enemies who are not pacman and not scared
            capsule_distances = [self.get_maze_distance(my_pos, capsule) for capsule in capsules_list]
            min_capsule_distance = min(capsule_distances, default=0)
            features['distance_to_capsule'] = min_capsule_distance*(food_eaten+1) # get close to capsules, especially if you're full
            
            if action == Directions.STOP: 
                features['stop'] = 1 # don't stop
                
            rev = Directions.REVERSE[game_state.get_agent_state(self.index).configuration.direction]
            if action == rev:
                features['reverse'] = 1 # don't reverse

        return features

    def get_weights(self, game_state, action):
        big_number = 999999999999999
        return {'score': big_number,
                'stay_away_from_base': -big_number, 'friend_dist': 1,
                'on_side': 10, 'invader_distance': 100,
                'neg_food_list': big_number, 'distance_to_food': -100, 'distance_to_enemy': 10,
                'distance_to_capsule': -10, 'full_stomach': 10000,
                'reverse': -5, 'stop': -5,
                'min_dist_to_home': -10, "hang_by_border": -10, 'defend_food': -75}


class DefensiveReflexAgent(ReflexCaptureAgent): # DANNY: make defensive pacman be defensive if scared, make offensive pacman be offensive if scared

    """
  A reflex agent that seeks food. This is an agent
  we give you to get an idea of what an offensive agent might look like,
  but it is by no means the best or only way to build an offensive agent.
  """

    def get_features(self, game_state, action):
        features = util.Counter()
        successor = self.get_successor(game_state, action)
        food_list = self.get_food(successor).as_list()
        capsules_list = self.get_capsules(successor)
        score = self.get_score(successor)
        our_food = self.get_food_you_are_defending(successor).as_list()
        
        my_state = successor.get_agent_state(self.index)
        my_pos = my_state.get_position()
        food_eaten = my_state.num_carrying
        am_scared = my_state.scared_timer > 0
        
        enemies = [successor.get_agent_state(i) for i in self.get_opponents(successor)]
        friends = [successor.get_agent_state(i) for i in self.get_team(successor)]
        
        features['score'] = score # increase the score
        
        friend_distance = max([self.get_maze_distance(my_pos, friend.get_position()) for friend in friends], default=0)
        features['friend_dist'] = log10(friend_distance+1)
        
        if self.red and my_pos[0] == 1:
            features['stay_away_from_base'] = 1
        elif my_pos[0] == game_state.data.layout.width-2:
            features['stay_away_from_base'] = 1
            
        if self.red:
            center_width = (game_state.data.layout.width / 2) - 1
        else:
            center_width = (game_state.data.layout.width / 2)
        heights = range(1, game_state.data.layout.height)

        if score > 0 and not am_scared: # be defensive
            
            upper_food_col = [food[0] for food in our_food if food[1] > game_state.data.layout.height/2]
            if self.red:
                closest_food_col = max(upper_food_col, default=0)
                closest_foods = [food for food in our_food if food[1] > game_state.data.layout.height/2 and food[0] == closest_food_col]
                closest_food = closest_foods[0]
            else:
                closest_food_col = min(upper_food_col, default=0)
                closest_foods = [food for food in our_food if food[1] > game_state.data.layout.height/2 and food[0] == closest_food_col]
                closest_food = closest_foods[0]
            
            if closest_food[1] == game_state.data.layout.height-2:
                viable_spots =  [float(item) for item in heights if not game_state.has_wall(closest_food[0], item)]
                closest_food = (closest_food[0], sorted(viable_spots)[-2])

            features['defend_food'] = self.get_maze_distance(my_pos, closest_food)
                
                
            features['on_side'] = int(not my_state.is_pacman)
            
            invaders = [a for a in enemies if a.is_pacman and a.get_position() is not None]
            
            if len(invaders) > 0:
                dists = [self.get_maze_distance(my_pos, a.get_position()) for a in invaders]
            else:
                dists = [0]
                
            if am_scared: # if scared
    
                min_dist_to_home = min([self.get_maze_distance((center_width, float(item)), my_pos) for item in heights if not game_state.has_wall(int(center_width), item)])
                
                features['min_dist_to_home'] = min_dist_to_home # go home when you're scared
                
                features['invader_distance'] = min(dists, default=0) # stay away from the invader when you're scared
            else:
                features['invader_distance'] = -min(dists, default=0) # go to the invader when you're not scared
                if features['invader_distance'] == 0:
                    features['invader_distance'] = 9999999999

            if action == Directions.STOP:
                features['stop'] = 1 # don't stop
            rev = Directions.REVERSE[game_state.get_agent_state(self.index).configuration.direction]
            
            if action == rev:
                features['reverse'] = 1 # don't reverse
                
            features['friends_dist'] = features['friends_dist']/500 # worry about distance less if playing defense
            
            # if there are no invaders
            
            if len(invaders) == 0:
                if self.red:
                    features['hang_by_border'] = min([self.get_maze_distance((center_width-2, float(item)), my_pos) for item in heights if not game_state.has_wall(int(center_width-2), item)], default=0)
                else:
                    features['hang_by_border'] = min([self.get_maze_distance((center_width+2, float(item)), my_pos) for item in heights if not game_state.has_wall(int(center_width+2), item)], default=0)
            
            
        else: # be offensive
            features['on_side'] = int(my_state.is_pacman)
            
            features['neg_food_list'] = -len(food_list) # decrease remaining food

            if self.red:
                center_width = (game_state.data.layout.width / 2) - 1
            else:
                center_width = (game_state.data.layout.width / 2)
            
            min_dist_to_home = min([self.get_maze_distance((center_width, float(item)), my_pos) for item in heights if not game_state.has_wall(int(center_width), item)], default=0)

            if min_dist_to_home > 0:
                inv_dist_to_home = 1/min_dist_to_home
            else:
                inv_dist_to_home = 999999
            features['full_stomach'] = (food_eaten)*inv_dist_to_home # go home when you're full, MAKE THIS EXPONENTIAL

            min_dist_to_food = min([self.get_maze_distance(my_pos, food) for food in food_list if food[1] > game_state.data.layout.height/2], default=0)
            features['distance_to_food'] = min_dist_to_food # get close to food

            enemy_distances = [self.get_maze_distance(my_pos, enemy.get_position()) for enemy in enemies if
                                not enemy.is_pacman and enemy.get_position() is not None and enemy.scared_timer == 0]
            distance_to_enemy = min(enemy_distances, default=0)
            features['distance_to_enemy'] = distance_to_enemy # stay away from enemies who are not pacman and not scared
            capsule_distances = [self.get_maze_distance(my_pos, capsule) for capsule in capsules_list]
            min_capsule_distance = min(capsule_distances, default=0)
            features['distance_to_capsule'] = min_capsule_distance*(food_eaten+1) # get close to capsules, especially if you're full
            
            if action == Directions.STOP: 
                features['stop'] = 1 # don't stop
                
            rev = Directions.REVERSE[game_state.get_agent_state(self.index).configuration.direction]
            if action == rev:
                features['reverse'] = 1 # don't reverse

        return features

    def get_weights(self, game_state, action):
        big_number = 999999999999999
        return {'score': big_number,
                'stay_away_from_base': -big_number, 'friend_dist': 1,
                'on_side': 10, 'invader_distance': 100,
                'neg_food_list': big_number, 'distance_to_food': -100, 'distance_to_enemy': 10,
                'distance_to_capsule': -10, 'full_stomach': 10000,
                'reverse': -5, 'stop': -5,
                'min_dist_to_home': -10, "hang_by_border": -1, "defend_food": -75}