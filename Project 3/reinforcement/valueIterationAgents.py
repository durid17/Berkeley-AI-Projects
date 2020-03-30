# valueIterationAgents.py
# -----------------------
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


import mdp, util

from learningAgents import ValueEstimationAgent

class ValueIterationAgent(ValueEstimationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A ValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100):
        """
          Your value iteration agent should take an mdp on
          construction, run the indicated number of iterations
          and then act according to the resulting policy.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state, action, nextState)
              mdp.isTerminal(state)
        """
        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations
        self.values = util.Counter() # A Counter is a dict with default 0
        self.minus_infinity = -1000000000000
        states = mdp.getStates()
        newValues = util.Counter()
        # this is a standart value iteration 
        # values are last computed values in value iteration
        # new values are new values for this iteration
        # after iteration is finished values become new values
        # and the proccess continues for number of iterations
        # finaly results are stored in value dictionary
        for _ in range(0 , iterations):
            for state in states:
                newValues[state] = self.minus_infinity
                actions = mdp.getPossibleActions(state)
                for action in actions:
                    newValues[state] = max(newValues[state] , self.computeQValueFromValues(state , action))
                if(newValues[state] == self.minus_infinity): newValues[state] = 0
            self.values = newValues.copy()

    # return value for state from dictionary
    def getValue(self, state):
        """
          Return the value of the state (computed in __init__).
        """
        return self.values[state]

    # this method counts qvalue for state and action
    def computeQValueFromValues(self, state, action):
        """
          Compute the Q-value of action in state from the
          value function stored in self.values.
        """
        # get all states and its probabilities where pacman will go with this action
        probs = self.mdp.getTransitionStatesAndProbs(state , action)
        qvalue = 0
        for prob in probs:
            # add qvalue probability of new state times value of new state
            qvalue += prob[1] * (self.mdp.getReward(state , action , prob[0]) + self.discount * self.values[prob[0]])
        # qvalue is our answer
        return qvalue

    # return best action from state
    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        qvalues = util.Counter()
        # get all possible actions from this state
        actions = self.mdp.getPossibleActions(state)
        for action in actions:
            # store qvalue of state action in dictionary
            qvalues[action] = self.computeQValueFromValues(state , action)
        # return arg max beetween qvalues dictionary
        return qvalues.argMax()

    # return best action from state , policy
    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    # return best action from state
    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    # return qvalue for state and action
    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)
