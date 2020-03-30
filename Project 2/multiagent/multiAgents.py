# multiAgents.py
# --------------
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


from util import manhattanDistance
from game import Directions
import random, util
import sys

from game import Agent

class ReflexAgent(Agent):
    """
      A reflex agent chooses an action at each choice point by examining
      its alternatives via a state evaluation function.

      The code below is provided as a guide.  You are welcome to change
      it in any way you see fit, so long as you don't touch our method
      headers.
    """


    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {North, South, West, East, Stop}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """

        """
        sum counts sum of manhattan Distances from pacmans position to ghosts
        mn is minimum manhattan Distances from pacmans position to food
        finally this function returns sum - 4 * mn
        also if successorGameState position is one step away from ghost , this function returns -inf
        """

        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        food_positions = newFood.asList()
        sum = 0
        mn = sys.maxsize
        for elem in newGhostStates:
            if(util.manhattanDistance(newPos,elem.getPosition()) <= 1) : return - sys.maxsize / 100
            sum += 3 * util.manhattanDistance(newPos,elem.getPosition())
        for elem in food_positions:
            mn = min(mn , util.manhattanDistance(newPos,elem))
        if len(newFood.asList()) != currentGameState.getFood().count(): mn = 0
        return sum - 4 * mn

def scoreEvaluationFunction(currentGameState):
    """
      This default evaluation function just returns the score of the state.
      The score is the same one displayed in the Pacman GUI.

      This evaluation function is meant for use with adversarial search agents
      (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
      This class provides some common elements to all of your
      multi-agent searchers.  Any methods defined here will be available
      to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

      You *do not* need to make any changes here, but you can if you want to
      add functionality to all your adversarial search agents.  Please do not
      remove anything, however.

      Note: this is an abstract class: one that should not be instantiated.  It's
      only partially specified, and designed to be extended.  Agent (game.py)
      is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
      Your minimax agent (question 2)
    """
    def getAction(self, gameState):

        """
        This is a recursive function , which takes 4 arguments, 
        gameState
        how many nodes are left to consider
        index from start point
        total num_agents
        from index and num_agents we can count is its pacman or ghost
        """
        def search(gameState , left, index , num_agents):
            if(left == 0): return self.evaluationFunction(gameState)
            actions = gameState.getLegalActions(index)
            if len(actions) == 0: return self.evaluationFunction(gameState)
            mn = 100000000
            mx = -100000000
            for action in actions:
                successor = gameState.generateSuccessor(index, action)
                res = search(successor , left - 1 , (index + 1) % num_agents , num_agents)
                mn = min(mn , res)
                mx = max(mx , res)
            if(index == 0): return mx
            return mn
        """
          Returns the minimax action from the current gameState using self.depth
          and self.evaluationFunction.

          Here are some method calls that might be useful when implementing minimax.

          gameState.getLegalActions(agentIndex):
            Returns a list of legal actions for an agent
            agentIndex=0 means Pacman, ghosts are >= 1

          gameState.generateSuccessor(agentIndex, action):
            Returns the successor game state after an agent takes an action

          gameState.getNumAgents():
            Returns the total number of agents in the game
        """

        """
        we count which action gives best answer to pacman and move that way
        """
        num_agents = gameState.getNumAgents()
        actions = gameState.getLegalActions(0)
        ans = actions[0]
        mx = -sys.maxsize - 1
        for action in actions:
            successor = gameState.generateSuccessor(0, action)
            res = search(successor , num_agents * self.depth - 1, 1 % num_agents, num_agents)
            if(res > mx):
                mx = res
                ans = action
        return ans

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """        
    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """

        """
        This is a recursive function which takes 6 arguments
        gameState , how many nodes are left to consider, index from start point , total num_agents , 
        mn_value best value for ghosts so far , mx_value best value for pacman so far
        from index and num_agents we can count is its pacman or ghost
        and rest is typical alfa beta turing solution
        """
        def search(gameState , left, index , num_agents , mn_value , mx_value):
            if(left == 0): return self.evaluationFunction(gameState)
            actions = gameState.getLegalActions(index)
            if len(actions) == 0: return self.evaluationFunction(gameState)
            mn = sys.maxsize
            mx = -sys.maxsize -1
            for action in actions:
                successor = gameState.generateSuccessor(index, action)
                if index == 0:
                    res = search(successor , left - 1 , (index + 1) % num_agents , num_agents , mn_value , mx_value)
                    mx = max(mx , res)
                    if mx > mn_value: break
                    mx_value = max(mx_value , mx)
                else:
                    res = search(successor , left - 1 , (index + 1) % num_agents , num_agents , mn_value , mx_value)
                    mn = min(mn , res)
                    if mn < mx_value: break
                    mn_value = min(mn_value , mn)
            if(index == 0): return mx
            return mn
        
        """
        we count which action gives best answer to pacman and move that way
        """
        num_agents = gameState.getNumAgents()
        actions = gameState.getLegalActions(0)
        ans = actions[0]
        mn = sys.maxsize
        mx = -sys.maxsize - 1
        for action in actions:
            successor = gameState.generateSuccessor(0, action)
            res = search(successor , num_agents * self.depth - 1, 1 % num_agents, num_agents , mn , mx)
            if(res > mx):
                mx = res
                ans = action
        return ans

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
          Returns the expectimax action using self.depth and self.evaluationFunction

          All ghosts should be modeled as choosing uniformly at random from their
          legal moves.
        """

        """
        This is a recursive function which takes 4 arguments
        gameState , how many nodes are left to consider, index from start point , total num_agents
        from index and num_agents we can count is its pacman or ghost
        if its pacman we count max from childs , if not we count average
        """

        def search(gameState , left, index , num_agents):
            if(left == 0): return self.evaluationFunction(gameState)
            actions = gameState.getLegalActions(index)
            if len(actions) == 0: return self.evaluationFunction(gameState)
            mx = -sys.maxsize - 1
            sum = 0.0
            for action in actions:
                successor = gameState.generateSuccessor(index, action)
                res = search(successor , left - 1 , (index + 1) % num_agents , num_agents)
                sum += float(res) / float(len(actions))
                mx = max(mx , res)
            if(index == 0): return mx
            return sum
        
        """
        we count which action gives best answer to pacman and move that way
        """
        num_agents = gameState.getNumAgents()
        actions = gameState.getLegalActions(0)
        res = search(gameState , num_agents * self.depth, 0 , num_agents)
        for action in actions:
            successor = gameState.generateSuccessor(0, action)
            if search(successor , num_agents * self.depth - 1, 1 % num_agents , num_agents) == res: return action
        

def betterEvaluationFunction(currentGameState):
    
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).

      DESCRIPTION: <write something here so we know what you did>
    """

    """
    sum counts sum of 3 times manhattan Distances from pacmans position to ghosts , only if manhattan Distance <= 4
    if manhattan Distance <= 1 we subtract  sys.maxsize / 100 to the result
    if state is win state we add  sys.maxsize / 1000 to the result
    if state is lose state we subtract  sys.maxsize / 1000 to the result
    finally this function returns sum - 4 * mn
    also if successorGameState position is one step away from ghost , this function returns -inf
    """
    successorGameState = currentGameState
    newPos = successorGameState.getPacmanPosition()
    newGhostStates = successorGameState.getGhostStates()
    sum = 0
    if successorGameState.isLose(): sum -= sys.maxsize / 1000
    if successorGameState.isWin(): sum += sys.maxsize / 1000
    for elem in newGhostStates:
        if(util.manhattanDistance(newPos,elem.getPosition()) <= 1) : sum -= sys.maxsize / 100
        if(util.manhattanDistance(newPos,elem.getPosition()) <= 4) :
            sum += 3 * util.manhattanDistance(newPos,elem.getPosition())
    return currentGameState.getScore() + sum

# Abbreviation
better = betterEvaluationFunction

