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

        # A Counter is a dict with default 0. Easy way of accesing a set of instances, list[instance]
        self.values = util.Counter()

        # Write value iteration code here

        # Our aim here is to get a list of the states with the best value possible given by any of the actions that
        # can be taken from that state. We save that list in newValues, which we later save in self.values.
        "*** Our Code Starts Here ***"


        for i in range(0, self.iterations):
            newValues = self.values.copy()  # copy the values, doesn't create a reference to them


            # getStates returns a list of all the posible states (x,y) coordinates of the agent

            for state in self.mdp.getStates():  # go through every state

                # If a state is a terminal, we skip the for iteration
                if self.mdp.isTerminal(state):
                    continue

                # We get the possible actions given a state and decide whichone has the best value

                bestValue = -999999999

                possibleActionsInState = self.mdp.getPossibleActions(state)

                for action in possibleActionsInState:

                    valueOfAction = self.getQValue(state, action)

                    if valueOfAction > bestValue:
                        bestValue = valueOfAction

                # Assigns to the State in the list newValues the best Value found for that state.
                #  {(0, 1): 1.0, (3, 2): -100.0, ...}
                newValues[state] = bestValue


            self.values = newValues


    def getValue(self, state):
        """
          Return the value of the state (computed in __init__).
        """
        return self.values[state]


    def computeQValueFromValues(self, state, action):
        """
          Compute the Q-value of action in state from the
          value function stored in self.values.
        """
        "*** Our Code Starts Here ***"

        qValue = 0

        # for every possible outcome of the action
        for nextState, probability in self.mdp.getTransitionStatesAndProbs(state, action):

            # add reward & future reward (=V) * probability of the outcome
            reward = self.mdp.getReward(state, action, nextState)
            qValue += probability * (reward + self.discount * self.values[nextState])

        return qValue


        #util.raiseNotDefined()

    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        "*** Our Code Starts Here ***"

        # get the best possible action for the state
        policies = util.Counter()
        for action in self.mdp.getPossibleActions(state):

            # how good is an action = q-value (which considers all possible outcomes)
            policies[action] = self.getQValue(state, action)

        # return the best action, e.g. 'north'
        return policies.argMax()

        # util.raiseNotDefined()

    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)
