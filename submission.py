from util import manhattanDistance
from game import Directions
import random, util

from game import Agent


class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
  """

    def __init__(self):
        self.lastPositions = []
        self.dc = None

    def getAction(self, gameState):
        """
    getAction chooses among the best options according to the evaluation function.

    getAction takes a GameState and returns some Directions.X for some X in the set {North, South, West, East, Stop}
    ------------------------------------------------------------------------------
    Description of GameState and helper functions:

    A GameState specifies the full game state, including the food, capsules,
    agent configurations and score changes. In this function, the |gameState| argument
    is an object of GameState class. Following are a few of the helper methods that you
    can use to query a GameState object to gather information about the present state
    of Pac-Man, the ghosts and the maze.

    gameState.getLegalActions():
        Returns the legal actions for the agent specified. Returns Pac-Man's legal moves by default.

    gameState.generateSuccessor(agentIndex, action):
        Returns the successor state after the specified agent takes the action.
        Pac-Man is always agent 0.

    gameState.getPacmanState():
        Returns an AgentState object for pacman (in game.py)
        state.configuration.pos gives the current position
        state.direction gives the travel vector

    gameState.getGhostStates():
        Returns list of AgentState objects for the ghosts

    gameState.getNumAgents():
        Returns the total number of agents in the game

    gameState.getScore():
        Returns the score corresponding to the current state of the game


    The GameState class is defined in pacman.py and you might want to look into that for
    other helper methods, though you don't need to.
    """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        # For each move in legalMoves, get the corresponding score
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        # Take the best score
        bestScore = max(scores)
        # Store indices of all moves which produce bestScore (one or more)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices)  # Pick randomly among the best

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
    The evaluation function takes in the current and proposed successor
    GameStates (pacman.py) and returns a number, where higher numbers are better.

    The code below extracts some useful information from the state, like the
    remaining food (oldFood) and Pacman position after moving (newPos).
    newScaredTimes holds the number of moves that each ghost will remain
    scared because of Pacman having eaten a power pellet.
    """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        oldFood = currentGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        return successorGameState.getScore()


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

    def __init__(self, evalFn='scoreEvaluationFunction', depth='2'):
        self.index = 0  # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)


######################################################################################
# Problem 1b: implementing minimax

class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (problem 1)
  """

    def getAction(self, gameState):
        """
      Returns the minimax action from the current gameState using self.depth
      and self.evaluationFunction. Terminal states can be found by one of the following:
      pacman won, pacman lost or there are no legal moves.

      Here are some method calls that might be useful when implementing minimax.

      gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

      gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

      gameState.getNumAgents():
        Returns the total number of agents in the game

      gameState.getScore():
        Returns the score corresponding to the current state of the game

      gameState.isWin():
        Returns True if it's a winning state

      gameState.isLose():
        Returns True if it's a losing state

      self.depth:
        The depth to which search should continue

    """

        # BEGIN_YOUR_CODE (our solution is 20 lines of code, but don't worry if you deviate from this)
        topOptions = []
        valueMoves = []

        def nextAgent(agentIndex, gameDepth):
            if gameState.getNumAgents() == agentIndex + 1:
                gameDepth -= 1
                nextAgentIndex = 0
            else:
                nextAgentIndex = agentIndex + 1
            return nextAgentIndex, gameDepth

        def baseCases(gameState, legalMoves, gameDepth):
            if gameState.isWin() or gameState.isLose() or len(legalMoves) == 0:
                return 'Game Over'
            elif gameDepth == 0:
                return 'Max Depth'

        # Return (minimax value of state, optimal action)
        def minimaxValueRecurs(gameState, gameDepth, agentIndex):

            legalMoves = gameState.getLegalActions(agentIndex)

            # Address two base cases
            if baseCases(gameState, legalMoves, gameDepth) == 'Game Over':
                return gameState.getScore()
            elif baseCases(gameState, legalMoves, gameDepth) == 'Max Depth':
                return self.evaluationFunction(gameState)

            else:
                options = []
                # If we've dealt with all agents, need to go one layer deeper and reset agent counter
                # Otherwise, bump up the agent counter
                nextAgentIndex, gameDepth = nextAgent(agentIndex, gameDepth)
                # Recursive step
                for action in legalMoves:
                    succState = gameState.generateSuccessor(agentIndex, action)
                    options.append([minimaxValueRecurs(succState,
                                                       gameDepth, nextAgentIndex), action])

                # Pacman
                if agentIndex == 0:
                    return max(options)[0]
                # Ghost
                return min(options)[0]

        allMoves = gameState.getLegalActions()
        for move in allMoves:
            nextState = gameState.generateSuccessor(self.index, move)
            valueMoves.append([minimaxValueRecurs(nextState,
                                                  self.depth, self.index + 1), move])

        # take the top score and move associated
        topScore = max(valueMoves)[0]
        for index in range(len(valueMoves)):
            if valueMoves[index][0] == topScore:
                topOptions.append(index)
        # Select randomly from top options
        return allMoves[random.choice(topOptions)]
        # END_YOUR_CODE


######################################################################################
# Problem 2a: implementing alpha-beta

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (problem 2)
  """

    def getAction(self, gameState):
        """
      Returns the minimax action using self.depth and self.evaluationFunction
    """

        # BEGIN_YOUR_CODE (our solution is 36 lines of code, but don't worry if you deviate from this)

        topOptions = []
        valueMoves = []

        # def alphaPruning(agentIndex, possMoves, standard, gameDepth, alpha, beta):
        #     for action in possMoves:
        #         choices = minimaxValueRecurs(gameState.generateSuccessor(agentIndex, action), gameDepth, nextAgent,
        #                                      alpha, beta)
        #         if choices > standard:
        #             standard = choices
        #         alpha = max(alpha, standard)
        #         if beta <= alpha:
        #             break
        #     return standard
        #
        # def betaPruning(agentIndex, possMoves, standard, gameDepth, alpha, beta):
        #     for action in possMoves:
        #         choices = minimaxValueRecurs(gameState.generateSuccessor(agentIndex, action), gameDepth, nextAgent,
        #                                      alpha, beta)
        #         if choices < standard:
        #             standard = choices
        #         beta = min(beta, standard)
        #         if beta <= alpha:
        #             break
        #     return standard

        def nextAgent(agentIndex, gameDepth):
            if gameState.getNumAgents() == agentIndex + 1:
                gameDepth -= 1
                nextAgentIndex = 0
            else:
                nextAgentIndex = agentIndex + 1
            return nextAgentIndex, gameDepth

        def baseCases(gameState, possMoves, gameDepth):
            if gameState.isWin() or gameState.isLose() or len(possMoves) == 0:
                return 'Game Over'
            elif gameDepth == 0:
                return 'Max Depth'

        # Return (minimax value of state, optimal action)
        def minimaxValueRecurs(gameState, gameDepth, agentIndex, alpha, beta):
            standard = 0
            possMoves = gameState.getLegalActions(agentIndex)

            # Address two base cases
            if baseCases(gameState, possMoves, gameDepth) == 'Game Over':
                return gameState.getScore()
            elif baseCases(gameState, possMoves, gameDepth) == 'Max Depth':
                return self.evaluationFunction(gameState)

            else:
                # If we've dealt with all agents, need to go one layer deeper and reset agent counter
                # Otherwise, bump up the agent counter
                nextAgentIndex, gameDepth = nextAgent(agentIndex, gameDepth)
                # Recursive step
                upper = float('inf')
                lower = float('-inf')
                for action in possMoves:
                    succState = gameState.generateSuccessor(agentIndex, action)
                    values = minimaxValueRecurs(succState, gameDepth, nextAgentIndex, alpha, beta)
                    if values < upper:
                        upper = values
                    beta = min(beta, upper)
                    if values > lower:
                        lower = values
                    alpha = max(alpha, lower)
                    if beta <= alpha:
                        break
                if agentIndex == 0:
                    return lower
                return upper

                # if agentIndex > 0:
                #     return betaPruning(agentIndex, possMoves, standard, gameDepth, alpha, beta)
                # else:
                #     return alphaPruning(agentIndex, possMoves, standard, gameDepth, alpha, beta)

        allMoves = gameState.getLegalActions()
        for move in allMoves:
            nextState = gameState.generateSuccessor(self.index, move)
            valueMoves.append(
                [minimaxValueRecurs(nextState, self.depth, self.index + 1, float('-inf'), float('inf')), move])

        # take the top score and move associated
        topScore = max(valueMoves)[0]
        for index in range(len(valueMoves)):
            if valueMoves[index][0] == topScore:
                topOptions.append(index)
        # Select randomly from top options
        return allMoves[random.choice(topOptions)]

        # END_YOUR_CODE


######################################################################################
# Problem 3b: implementing expectimax

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
    Your expectimax agent (problem 3)
  """

    def getAction(self, gameState):
        """
      Returns the expectimax action using self.depth and self.evaluationFunction

      All ghosts should be modeled as choosing uniformly at random from their
      legal moves.
    """

        # BEGIN_YOUR_CODE (our solution is 20 lines of code, but don't worry if you deviate from this)
        topOptions = []
        valueMoves = []

        def nextAgent(agentIndex, gameDepth):
            if gameState.getNumAgents() == agentIndex + 1:
                gameDepth -= 1
                nextAgentIndex = 0
            else:
                nextAgentIndex = agentIndex + 1
            return nextAgentIndex, gameDepth

        def baseCases(gameState, legalMoves, gameDepth):
            if gameState.isWin() or gameState.isLose() or len(legalMoves) == 0:
                return 'Game Over'
            elif gameDepth == 0:
                return 'Max Depth'

        # Return (minimax value of state, optimal action)
        def minimaxValueRecurs(gameState, gameDepth, agentIndex):

            legalMoves = gameState.getLegalActions(agentIndex)

            # Address two base cases
            if baseCases(gameState, legalMoves, gameDepth) == 'Game Over':
                return gameState.getScore()
            elif baseCases(gameState, legalMoves, gameDepth) == 'Max Depth':
                return self.evaluationFunction(gameState)

            else:
                options = []
                # If we've dealt with all agents, need to go one layer deeper and reset agent counter
                # Otherwise, bump up the agent counter
                nextAgentIndex, gameDepth = nextAgent(agentIndex, gameDepth)
                # Recursive step
                for action in legalMoves:
                    succState = gameState.generateSuccessor(agentIndex, action)
                    options.append([minimaxValueRecurs(succState, gameDepth, nextAgentIndex), action])

                if agentIndex == 0:
                    return max(options)[0]
                return random.choice(options)[0]

        allMoves = gameState.getLegalActions()
        for move in allMoves:
            nextState = gameState.generateSuccessor(self.index, move)
            valueMoves.append([minimaxValueRecurs(nextState,
                                                  self.depth, self.index + 1), move])

        # take the top score and move associated
        topScore = max(valueMoves)[0]
        for index in range(len(valueMoves)):
            if valueMoves[index][0] == topScore:
                topOptions.append(index)
        # Select randomly from top options
        return allMoves[random.choice(topOptions)]
        # END_YOUR_CODE


######################################################################################
# Problem 4a (extra credit): creating a better evaluation function

def betterEvaluationFunction(currentGameState):
    """
    Your extreme, unstoppable evaluation function (problem 4). Note that you can't fix a seed in this function.

    DESCRIPTION: <write something here so we know what you did>
  """

    # BEGIN_YOUR_CODE (our solution is 13 lines of code, but don't worry if you deviate from this)
    raise Exception("Not implemented yet")
    # END_YOUR_CODE


# Abbreviation
better = betterEvaluationFunction
