# search.py
# ---------
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


"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
"""

import util

class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return  [s, s, w, s, w, w, s, w]

def depthFirstSearch(problem):
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    """
    # print "Start:", problem.getStartState()
    # print "Is the start a goal?", problem.isGoalState(problem.getStartState())
    # print "Start's successors:", problem.getSuccessors(problem.getStartState())

    """
        It's a simple DFS , stack contains tuples , first argument is position of pacman and second argument is
        path how he got there
        been contains position where pacman has already been
    """
    s = util.Stack()
    been = set()
    s.push((problem.getStartState() , []))
    while not s.isEmpty():
        elem = s.pop()
        state = elem[0]
        path = elem[1]
        if state in been: continue
        been.add(state)
        if problem.isGoalState(state): return path
        successors = problem.getSuccessors(state)
        for successor in successors:
            new_path = path[:]
            new_path.append(successor[1])
            s.push((successor[0] , new_path))
    return []

def breadthFirstSearch(problem):
    """Search the shallowest nodes in the search tree first."""
    """
        It's a simple BFS , queue contains tuples , first argument is position of pacman and second argument is
        path how he got there
        been contains position where pacman has already been
    """
    q = util.Queue()
    been = set()
    q.push((problem.getStartState() , []))
    while not q.isEmpty():
        elem = q.pop()
        state = elem[0]
        path = elem[1]
        if state in been: continue
        been.add(state)
        if problem.isGoalState(state): 
            return path
        successors = problem.getSuccessors(state)
        for successor in successors:
            new_path = path[:]
            new_path.append(successor[1])
            q.push((successor[0] , new_path))

    
    return []

def uniformCostSearch(problem):
    """Search the node of least total cost first."""
    """
        It's a simple UCS , Priority Queue contains tuples , first argument is position of pacman and second argument is
        path how he got there , priority while adding node to the queue is length of the path + cost of this move
        been contains position where pacman has already been
    """
    q = util.PriorityQueue()
    been = set()
    costs = {}
    q.push((problem.getStartState() , []) , 0)
    costs[problem.getStartState()] = 0
    while not q.isEmpty():
        elem = q.pop()
        state = elem[0]
        path = elem[1]
        if state in been: continue
        been.add(state)
        if problem.isGoalState(state): return path
        successors = problem.getSuccessors(state)
        for successor in successors:
            new_path = path[:]
            new_path.append(successor[1])
            cost = costs[state] + successor[2]
            if not successor[0] in costs: costs[successor[0]] = cost
            q.push((successor[0] , new_path) , cost)
    return []

def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def aStarSearch(problem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    """
        It's a simple UCS , Priority Queue contains tuples , first argument is position of pacman and second argument is
        path how he got there , priority while adding node to the queue is length of the path + cost of this move
        + heuristic functions value for this position
        been contains position where pacman has already been
    """
    q = util.PriorityQueue()
    been = set()
    q.push((problem.getStartState() , []) , 0)
    costs = {}
    costs[problem.getStartState()] = heuristic(problem.getStartState() , problem)
    while not q.isEmpty():
        elem = q.pop()
        state = elem[0]
        path = elem[1]
        if state in been: continue
        been.add(state)
        if problem.isGoalState(state): return path
        successors = problem.getSuccessors(state)
        for successor in successors:
            new_path = path[:]
            new_path.append(successor[1])
            cost = costs[state] + successor[2]
            if not successor[0] in costs: costs[successor[0]] = cost
            q.push((successor[0] , new_path) , cost + heuristic(successor[0] , problem))
    return []


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
