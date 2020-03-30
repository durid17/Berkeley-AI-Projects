# qlearningAgents.py
# ------------------
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


from game import *
from learningAgents import ReinforcementAgent
from featureExtractors import *

import random,util,math

class QLearningAgent(ReinforcementAgent):
    """
      Q-Learning Agent

      Functions you should fill in:
        - computeValueFromQValues
        - computeActionFromQValues
        - getQValue
        - getAction
        - update

      Instance variables you have access to
        - self.epsilon (exploration prob)
        - self.alpha (learning rate)
        - self.discount (discount rate)

      Functions you should use
        - self.getLegalActions(state)
          which returns legal actions for a state
    """
    def __init__(self, **args):
        "You can initialize Q-values here..."
        ReinforcementAgent.__init__(self, **args)
        self.qvalues = util.Counter() #dictionary for qvalues
        self.minus_infinity = -10000000000000 #minus infinity

    # return qvalue for state and action from dictionary
    def getQValue(self, state, action):
        """
          Returns Q(state,action)
          Should return 0.0 if we have never seen a state
          or the Q node value otherwise
        """
        return self.qvalues[(state , action)]

    # computes values from qvalues
    def computeValueFromQValues(self, state):
        """
          Returns max_action Q(state,action)
          where the max is over legal actions.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return a value of 0.0.
        """
        # get all legal actions from this state
        actions = self.getLegalActions(state)
        if len(actions) == 0: return 0.0 # if there is no legal action value is zero

        # value is max between qvalues
        value = self.minus_infinity
        for a in actions:
          value = max(value , self.getQValue(state ,a))
        return value

    # computes action from qvalues 
    def computeActionFromQValues(self, state):
        """
          Compute the best action to take in a state.  Note that if there
          are no legal actions, which is the case at the terminal state,
          you should return None.
        """
        # get all legal actions from this state
        actions = self.getLegalActions(state)
        action = None
        value = self.minus_infinity
        # computes action which has best qvalue
        for a in actions:
          if self.getQValue(state ,a) > value: 
            action = a
            value = self.getQValue(state ,a)
        return action

    # gets new action
    def getAction(self, state):
        """
          Compute the action to take in the current state.  With
          probability self.epsilon, we should take a random action and
          take the best policy action otherwise.  Note that if there are
          no legal actions, which is the case at the terminal state, you
          should choose None as the action.

          HINT: You might want to use util.flipCoin(prob)
          HINT: To pick randomly from a list, use random.choice(list)
        """
        # Pick Action
        # gets all legal actions from state
        legalActions = self.getLegalActions(state)
        # with probability of self.epsilon returns random action from list
        # otherwise return best action
        if( not util.flipCoin(self.epsilon)):
          return self.computeActionFromQValues(state)
        else: 
          return random.choice(legalActions)

    # updates qvalue after one sample
    def update(self, state, action, nextState, reward):
        """
          The parent class calls this to observe a
          state = action => nextState and reward transition.
          You should do your Q-Value update here

          NOTE: You should never call this function,
          it will be called on your behalf
        """
        # new qvalue is (1 - alpa) * previos valiue + alpa * sample
        self.qvalues[(state , action)] =  (1 - self.alpha) * self.qvalues[(state , action)] + self.alpha * (reward + self.discount * self.computeValueFromQValues(nextState))

    # return bests action drom state , policy
    def getPolicy(self, state):
        return self.computeActionFromQValues(state)

    # gets states value
    def getValue(self, state):
        return self.computeValueFromQValues(state)


class PacmanQAgent(QLearningAgent):
    "Exactly the same as QLearningAgent, but with different default parameters"

    def __init__(self, epsilon=0.05,gamma=0.8,alpha=0.2, numTraining=0, **args):
        """
        These default parameters can be changed from the pacman.py command line.
        For example, to change the exploration rate, try:
            python pacman.py -p PacmanQLearningAgent -a epsilon=0.1

        alpha    - learning rate
        epsilon  - exploration rate
        gamma    - discount factor
        numTraining - number of training episodes, i.e. no learning after these many episodes
        """
        args['epsilon'] = epsilon
        args['gamma'] = gamma
        args['alpha'] = alpha
        args['numTraining'] = numTraining
        self.index = 0  # This is always Pacman
        QLearningAgent.__init__(self, **args)

    def getAction(self, state):
        """
        Simply calls the getAction method of QLearningAgent and then
        informs parent of action for Pacman.  Do not change or remove this
        method.
        """
        action = QLearningAgent.getAction(self,state)
        self.doAction(state,action)
        return action


class ApproximateQAgent(PacmanQAgent):
    """
       ApproximateQLearningAgent

       You should only have to overwrite getQValue
       and update.  All other QLearningAgent functions
       should work as is.
    """
    def __init__(self, extractor='IdentityExtractor', **args):
        self.featExtractor = util.lookup(extractor, globals())()
        PacmanQAgent.__init__(self, **args)
        self.weights = util.Counter()

    # return all weights as dictionary
    def getWeights(self):
        return self.weights

    # returns value for this action
    def getQValue(self, state, action):
        """
          Should return Q(state,action) = w * featureVector
          where * is the dotProduct operator
        print(state)
        """
        sum = 0.0
        # get all features
        features = self.featExtractor.getFeatures(state , action)
        # counts value with sum of feature[i] * w[i]
        for key in features:
          sum += self.weights[key]  * features[key]
        return sum

    # updates weights
    def update(self, state, action, nextState, reward):
        """
           Should update your weights based on transition
        """
        # get all features
        features = self.featExtractor.getFeatures(state , action)
        # difference is sample minus last value
        difference = (reward + self.discount * self.getValue(nextState) - self.getQValue(state , action))
        for key in features.keys():
          value = features[key]
          # wieght is last weight + alpa * difference * value, from formula
          self.weights[key] = self.weights[key] + self.alpha * difference * value

    def final(self, state):
        "Called at the end of each game."
        # call the super-class final method
        PacmanQAgent.final(self, state)

        # did we finish training?
        if self.episodesSoFar == self.numTraining:
          pass