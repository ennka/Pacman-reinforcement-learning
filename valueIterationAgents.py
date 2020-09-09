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
import collections

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
        self.runValueIteration()

    def runValueIteration(self):
        # Write value iteration code here
        "*** YOUR CODE HERE ***"
        for i in range(self.iterations):
            #print(self.values)
            newValues = self.values.copy()

            for state in self.mdp.getStates():
                actions=self.mdp.getPossibleActions(state)
                temp=[self.computeQValueFromValues(state,action) for action in actions]
                if len(temp)==0:
                    continue
                maxValue=max(temp)
                newValues[state]=maxValue
                # #self.values[state]=maxValue
                # print(state)
                # print(self.values)
                # print(newValues)
                # print("?")
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
        "*** YOUR CODE HERE ***"
        qValue=0
        for s, t in self.mdp.getTransitionStatesAndProbs(state, action):
            qValue+= t*(self.mdp.getReward(state,action,s) + self.discount* self.getValue(s))
        return qValue

    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        import operator
        action = {}
        for a in self.mdp.getPossibleActions(state):
            action[a] = self.computeQValueFromValues(state, a)
        if len(action)==0:
            return None
        return sorted(action.items(), key=operator.itemgetter(1), reverse=True)[0][0]#return the max key value


    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)

class AsynchronousValueIterationAgent(ValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        An AsynchronousValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs cyclic value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 1000):
        """
          Your cyclic value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy. Each iteration
          updates the value of only one state, which cycles through
          the states list. If the chosen state is terminal, nothing
          happens in that iteration.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state)
              mdp.isTerminal(state)
        """
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        "*** YOUR CODE HERE ***"

        states = [state for state in self.mdp.getStates()]
        for i in range(self.iterations):
            newValues = self.values.copy()
            state=states[i % len(states)]
            actions = self.mdp.getPossibleActions(state)
            temp = [self.computeQValueFromValues(state, action) for action in actions]
            if len(temp) == 0:
                continue
            maxValue = max(temp)
            newValues[state] = maxValue
            self.values = newValues



class PrioritizedSweepingValueIterationAgent(AsynchronousValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A PrioritizedSweepingValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs prioritized sweeping value iteration
        for a given number of iterations using the supplied parameters.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100, theta = 1e-5):
        """
          Your prioritized sweeping value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy.
        """
        self.theta = theta
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        "*** YOUR CODE HERE ***"
        pq=util.PriorityQueue()
        states = self.mdp.getStates()

        dict={}
        for state in states:
            if self.mdp.isTerminal(state):
                continue
            maxQ=max([ self.computeQValueFromValues(state, action)for action in self.mdp.getPossibleActions(state) ])
            #Get the best Q-value
            diff = abs(self.values[state] - maxQ)
            pq.update(state, -diff)
            for action in self.mdp.getPossibleActions(state):
                for newState, t in self.mdp.getTransitionStatesAndProbs(state, action):
                    if newState in dict.keys():
                        dict[newState].append(state)
                    else:
                        dict[newState]=[state]







        for i in range(self.iterations):
            if pq.isEmpty():
                break
            state=pq.pop()
            if not self.mdp.isTerminal(state):
                maxQ = max(
                    [self.computeQValueFromValues(state, action) for action in self.mdp.getPossibleActions(state)])
                self.values[state]=maxQ

            for newState in dict[state]:
                    if self.mdp.isTerminal(newState):
                        continue

                    maxQ = max(
                        [self.computeQValueFromValues(newState, a) for a in
                         self.mdp.getPossibleActions(newState)])
                    # Get the best Q-value
                    difference = abs(self.values[newState] - maxQ)
                    if self.theta < difference:
                        pq.update(newState, -difference)