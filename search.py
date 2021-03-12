"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
"""

import util
import sys
from time import sleep

from game import Directions

n = Directions.NORTH
s = Directions.SOUTH
e = Directions.EAST
w = Directions.WEST


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


def depthFirstSearch(problem):
    '''
    return a path to the goal
    '''
    # TODO 05
    # initialization: create stack and visited list
    stack = util.Stack()
    visited = []
    # push starting state into stack
    stack.push((problem.getStartState(), []))
    # pop 1st state
    (state, toDirection) = stack.pop()
    # add to visited states
    visited.append(state)

    # loop until found goal
    while not problem.isGoalState(state):
        # get state's successor
        successors = problem.getSuccessors(state)
        for successor in successors:
            # successor: (state, toDirection, toCost = 1)
            # if successor has not been visited then push into stack
            if not successor[0] in visited:
                stack.push((successor[0], toDirection + [successor[1]]))
                visited.append(successor[0])
        (state, toDirection) = stack.pop()

    return toDirection


def breadthFirstSearch(problem):
    '''
    return a path to the goal
    '''
    # TODO 06
    # initialization: create queue and visited list
    queue = util.Queue()
    visited = []
    # push starting state into queue
    queue.push((problem.getStartState(), []))
    # pop the 1st state
    (state, toDirection) = queue.pop()
    # add to visited states
    visited.append(state)

    # while not found goal
    while not problem.isGoalState(state):
        # get point's successor
        successors = problem.getSuccessors(state)
        for successor in successors:
            # (state, toDirection, toCost = 1)
            # if successor has not been visited then push into stack
            if not successor[0] in visited:
                queue.push((successor[0], toDirection + [successor[1]]))
                visited.append(successor[0])
        (state, toDirection) = queue.pop()

    return toDirection


def uniformCostSearch(problem):
    '''
    return a path to the goal
    '''
    # TODO 07
    from game import Directions
    # initialization
    queue = util.PriorityQueue()
    visited = []

    # push starting point
    queue.push((problem.getStartState(), [], 0), 0)
    # pop the point
    (state, toDirection, toCost) = queue.pop()
    # add to visited
    visited.append((state, toCost))

    # while goal not found
    while not problem.isGoalState(state):
        # get successors
        successors = problem.getSuccessors(state)
        for successor in successors:
            visitedExist = False
            total_cost = toCost + successor[2]
            for (visitedState, visitedToCost) in visited:
                # add state only if successor not visited, or visited but lower cost
                if (successor[0] == visitedState) and (total_cost >= visitedToCost):
                    visitedExist = True
                    break

            if not visitedExist:
                # push state with its total cost as priority
                queue.push((successor[0], toDirection + [successor[1]], total_cost), total_cost)
                visited.append((successor[0], total_cost))  # add this point to visited list
        (state, toDirection, toCost) = queue.pop()

    return toDirection


def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0


# TODO 08 + 09
'''
students propose at least two heuristic functions for A*
'''


def manhattanHeuristic(state, problem=None):
    import math
    # print state[1][0][0], problem
    heuristic = 0.0
    foodCor = []
    xCor = 0
    for x in state[1]:
        yCor = 0
        for y in x:
            if y == True:
                # print "found", xCor, yCor
                foodCor.append((xCor, yCor))
            yCor += 1
        xCor += 1
    # cal manhattan distance from current position to food
    for (xFood, yFood) in foodCor:
        (xState, yState) = state[0]
        heuristicCur = math.fabs((xState - xFood)) + math.fabs((yState - yFood))
        if heuristicCur > heuristic: heuristic = heuristicCur
    # print heuristic
    return heuristic


def euclideanHeuristic(state, problem=None):
    import math
    # print state[1][0][0], problem
    heuristic = 0.0
    foodCor = []
    xCor = 0
    for x in state[1]:
        yCor = 0
        for y in x:
            if y == True:
                # print "found", xCor, yCor
                foodCor.append((xCor, yCor))
            yCor += 1
        xCor += 1
    # cal euclidean distance from current position to food
    for (xFood, yFood) in foodCor:
        (xState, yState) = state[0]
        heuristicCur = math.sqrt(math.pow((xState - xFood), 2) + math.pow((yState - yFood), 2))
        if heuristicCur > heuristic: heuristic = heuristicCur
    # print heuristic
    return heuristic


def aStarSearch(problem, heuristic=nullHeuristic):
    '''
    return a path to the goal
    '''
    # TODO 10
    # initialization
    pq = util.PriorityQueue()
    visited = []

    # push the starting point into the queue
    pq.push((problem.getStartState(), [], 0), heuristic(problem.getStartState(), problem))
    # pop out the point
    (state, toDirection, toCost) = pq.pop()
    # add to visited
    visited.append((state, toCost + heuristic(problem.getStartState())))
    # while not found goal
    while not problem.isGoalState(state):
        # get successors
        successors = problem.getSuccessors(state)
        for successor in successors:
            visited_exist = False
            cost = toCost + successor[2]
            for (visitedState, visitedCost) in visited:
                # if successor has not been visited, or has a lower cost than previous one
                if (successor[0] == visitedState) and (cost >= visitedCost):
                    visited_exist = True
                    break
            if not visited_exist:
                # push the point with priority number of its total cost
                pq.push((successor[0], toDirection + [successor[1]], cost),
                        cost + heuristic(successor[0], problem))
                # add to visited
                visited.append((successor[0], cost))
        (state, toDirection, toCost) = pq.pop()

    return toDirection


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
