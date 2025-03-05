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
from math import sqrt, log

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
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
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
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"
        newFoodPositions = newFood.asList()
        foodDistances = [manhattanDistance(newPos, food) for food in newFoodPositions]
        if len(foodDistances) != 0:
            minFoodDistance = min(foodDistances)
        else:
            minFoodDistance = 1

        if action == 'Stop':
            return -1 # living cost - we don't want to stop moving

        ghostPosition = newGhostStates[0].getPosition() # only works with a single ghost
        if manhattanDistance(ghostPosition, newPos) < 4:
            return -100 # don't get close to ghosts
        else:
            return successorGameState.getScore() + 1 / minFoodDistance # closer food -> better eval


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

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"
        return self.value(gameState, 0, 0)[1]

    def value(self, gameState, agentIndex, depth):
        if (gameState.isWin() or gameState.isLose() or depth == self.depth):
            return self.evaluationFunction(gameState), ""
        if agentIndex == 0:
            return self.maxValue(gameState, agentIndex, depth)
        else:
            return self.minValue(gameState, agentIndex, depth)

    def maxValue(self, gameState, agentIndex, depth):
        legalActions = gameState.getLegalActions(agentIndex)
        bestAction = ""
        v = float("-inf")
        for action in legalActions:
            successor = gameState.generateSuccessor(agentIndex, action)
            v2 = self.value(successor, 1, depth)[0]
            if v2 > v:
                v = v2
                bestAction = action
        return v, bestAction

    def minValue(self, gameState, agentIndex, depth):
        legalActions = gameState.getLegalActions(agentIndex)
        bestAction = ""
        v = float("inf")
        for action in legalActions:
            successor = gameState.generateSuccessor(agentIndex, action)
            if agentIndex == gameState.getNumAgents() - 1:
                v2 = self.value(successor, 0, depth + 1)[0]
            else:
                v2 = self.value(successor, agentIndex + 1, depth)[0]
            if v2 < v:
                v = v2
                bestAction = action
        return v, bestAction

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        return self.value(gameState, 0, 0, float("-inf"), float("inf"))[1]

    def value(self, gameState, agentIndex, depth, alpha, beta):
        if (gameState.isWin() or gameState.isLose() or depth == self.depth):
            return self.evaluationFunction(gameState), ""
        if agentIndex == 0:
            return self.maxValue(gameState, agentIndex, depth, alpha, beta)
        else:
            return self.minValue(gameState, agentIndex, depth, alpha, beta)

    def maxValue(self, gameState, agentIndex, depth, alpha, beta):
        legalActions = gameState.getLegalActions(agentIndex)
        bestAction = ""
        v = float("-inf")
        for action in legalActions:
            successor = gameState.generateSuccessor(agentIndex, action)
            v2 = self.value(successor, 1, depth, alpha, beta)[0]
            if v2 > v:
                v = v2
                bestAction = action
            if v > beta:
                return v, bestAction
            alpha = max(alpha, v)
        return v, bestAction

    def minValue(self, gameState, agentIndex, depth, alpha, beta):
        legalActions = gameState.getLegalActions(agentIndex)
        bestAction = ""
        v = float("inf")
        for action in legalActions:
            successor = gameState.generateSuccessor(agentIndex, action)
            if agentIndex == gameState.getNumAgents() - 1:
                v2 = self.value(successor, 0, depth + 1, alpha, beta)[0]
            else:
                v2 = self.value(successor, agentIndex + 1, depth, alpha, beta)[0]
            if v2 < v:
                v = v2
                bestAction = action
            if v < alpha:
                return v, bestAction
            beta = min(beta, v)
        return v, bestAction

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
        "*** YOUR CODE HERE ***"
        return self.value(gameState, 0, 0)[1]

    def value(self, gameState, agentIndex, depth):
        if (gameState.isWin() or gameState.isLose() or depth == self.depth):
            return self.evaluationFunction(gameState), ""
        if agentIndex == 0:
            return self.maxValue(gameState, agentIndex, depth)
        else:
            return self.expValue(gameState, agentIndex, depth)

    def maxValue(self, gameState, agentIndex, depth):
        legalActions = gameState.getLegalActions(agentIndex)
        bestAction = ""
        v = float("-inf")
        for action in legalActions:
            successor = gameState.generateSuccessor(agentIndex, action)
            v2 = self.value(successor, 1, depth)[0]
            if v2 > v:
                v = v2
                bestAction = action
        return v, bestAction

    def expValue(self, gameState, agentIndex, depth):
        legalActions = gameState.getLegalActions(agentIndex)
        bestAction = ""
        v = 0
        for action in legalActions:
            successor = gameState.generateSuccessor(agentIndex, action)
            p = 1 / len(legalActions)
            if agentIndex == gameState.getNumAgents() - 1:
                v += p*self.value(successor, 0, depth + 1)[0]
                bestAction = action
            else:
                v += p*self.value(successor, agentIndex + 1, depth)[0]
                bestAction = action
        return v, bestAction


class MonteCarloAgent(MultiAgentSearchAgent):
    def __init__(self, **kwargs):
        self.enableRandomPolicy = int(kwargs.get('enableRandomPolicy', 0))
        self.simulation_depth = int(kwargs.get('simulation_depth', 20))
        self.simulations_per_move = int(kwargs.get('simulations_per_move', 50))
        self.c = 500 * sqrt(2)

    def getAction(self, state):
        root = mctsnode(state, None, agent_index=0)
        for _ in range(self.simulations_per_move):
            leaf = self.treePolicy(root)
            simulation_result = self.simulation(leaf.state, leaf.agent_index)
            self.backupNegMax(leaf, simulation_result)
        best_move = self.bestChild(root, 0).action
        return best_move

    def treePolicy(self, node):
        while not node.is_terminal_node():
            if not node.is_fully_expanded():
                return self.expand(node)
            else:
                c = node.state.getWalls().width * node.state.getWalls().height * sqrt(2)
                node = self.bestChild(node, self.c)
        return node

    def expand(self, node):
        action = random.choice(node.untried_actions)
        node.untried_actions.remove(action)
        index = node.agent_index
        if index == node.state.getNumAgents() - 1:
            index = 0
        else:
            index += 1
        next_state = node.state.generateSuccessor(node.agent_index, action)
        child_node = mctsnode(state=next_state, parent=node, action=action, agent_index=index)
        node.children.append(child_node)
        return child_node

    def simulation(self, state, agent_index):
        sim = 0
        index = agent_index
        while not (state.isWin() or state.isLose() or sim > self.simulation_depth):
            sim += 1

            legalActions = state.getLegalActions(index)
            if self.enableRandomPolicy == 0 or index != 0:
                action = random.choice(legalActions)
            else:
                scores = [self.evaluationFunctionPacman(state, action) for action in legalActions]
                bestScore = max(scores)
                bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
                chosenIndex = random.choice(bestIndices)  # Pick randomly among the best
                action = legalActions[chosenIndex]

            state = state.generateSuccessor(index, action)
            index = (index + 1) % state.getNumAgents()

        return state.getScore()

    def backupNegMax(self, node, result):
        while node is not None:
            node.visits += 1
            if node.parent is None or node.parent.agent_index != 0:
                node.q_sum -= result
            else:
                node.q_sum += result
            node = node.parent

    def bestChild(self, root, c):
        bestChild = root.children[0]
        maxVal = -float("inf")
        for child in root.children:
            val = child.q_sum / (child.visits + 1) + c * sqrt(2 * log(root.visits) / (child.visits + 1))
            if val > maxVal:
                maxVal = val
                bestChild = child
        return bestChild

    def evaluationFunctionPacman(self, currentGameState, action):

        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()

        newFoodPositions = newFood.asList()
        foodDistances = [manhattanDistance(newPos, food) for food in newFoodPositions]
        if len(foodDistances) != 0:
            minFoodDistance = min(foodDistances)
        else:
            minFoodDistance = 1

        if action == 'Stop':
            return -10000000 # living cost - we don't want to stop moving

        return successorGameState.getScore() + 1 / minFoodDistance  # closer food -> better eval

class mctsnode:
    def __init__(self, state, parent, agent_index, action=None):
        self.agent_index = agent_index
        self.state = state
        self.parent = parent
        self.children = []
        good_actions = state.getLegalActions(agent_index)
        if Directions.STOP in good_actions:
            good_actions.remove(Directions.STOP)
        self.untried_actions = good_actions
        self.action = action
        self.q_sum = 0
        self.visits = 0

    def is_fully_expanded(self):
        return len(self.untried_actions) == 0

    def is_terminal_node(self):
        return self.state.isWin() or self.state.isLose()