import copy as cp
import random as rd


class QAgent:

    def __init__(self, name="no_name"):
        self.name = name

        self.Q = {}  # Q-functions
        self.preferences = []  # [a, b, c] <=> Q_a > Q_b > Q_c

        self.actions = []

        self.decay_method = "linear"
        self.epsilon = 1.0
        self.epsilon_decay = 0
        self.alpha = 0.1
        self.gamma = 0.9

        self.isRandom = False

        self.selection_method = "lex"

    def setActions(self, actions):
        self.actions = actions
        
    def getQValues(self, qfunction, state):
        return self.Q[qfunction].get(state, {})
    
    def initDecay(self, steps):
        if self.decay_method == "linear":
            self.epsilon_decay = self.epsilon / steps
        elif self.decay_method == "exponential":
            self.epsilon_decay = 0.99 ** (1 / steps)
        else:
            raise ValueError(f"Unknown decay method: {self.decay_method}")

    def setPreferences(self, preferences):
        self.preferences = preferences
        for q in preferences:
            if q not in self.Q:
                raise ValueError(f"Q-function '{q}' does not exist. Existing Q-functions: {list(self.Q.keys())}")

    def addQFunction(self, name):
        if name not in self.Q:
            self.Q[name] = {}
            self.preferences.append(name)
        else:
            raise ValueError(f"Q-function '{name}' already exists.")

    def getBestActions(self, qfunction, state, actions=None, tolerance=0):
        if actions is None:
            actions = cp.deepcopy(self.actions)
        qvalues = self.getQValues(qfunction, state)
        if not qvalues:
            return None
        max_value = max([qvalues[action] for action in actions if action in qvalues])
        # apply tolerance
        max_value = max_value - (tolerance / 100) * abs(max_value - min(qvalues.values()))
        return [action for action, value in qvalues.items() if value >= max_value and action in actions]
    
    def getActionsAboveThreshold(self, qfunction, state, actions=None, threshold=0):
        if actions is None:
            actions = cp.deepcopy(self.actions)
        qvalues = self.getQValues(qfunction, state)
        if not qvalues:
            return None
        possible = [action for action, value in qvalues.items() if value >= threshold and action in actions]
        if len(possible) > 0:
            return possible
        # if no actions above threshold, return the best actions
        max_value = max([qvalues[action] for action in actions if action in qvalues])
        return [action for action, value in qvalues.items() if value == max_value and action in actions]

    def selectBestAction(self, state):
        if self.selection_method == "lex":
            return self.lexicographic(state)
        if self.selection_method == "tlex":
            return self.thresholdLexicographic(state)
        if self.selection_method == "dlex":
            return self.deltaLexicographic(state)

    def lexicographic(self, state):
        actions = cp.deepcopy(self.actions)
        for q in self.preferences:
            # print(actions, end="->")
            actions = self.getBestActions(q, state, actions)
        # print(actions)
        return actions

    def thresholdLexicographic(self, state):
        # TODO: doesnt seem to work for now - fix later
        actions = cp.deepcopy(self.actions)
        for q in self.preferences:
            # print(actions, end="->")
            actions = self.getActionsAboveThreshold(q, state, actions, -0.5)
        # print(actions)
        return actions

    def deltaLexicographic(self, state, tolerance=10):
        # tolerance in percent
        actions = cp.deepcopy(self.actions)
        for q in self.preferences:
            # print(actions, end="->")
            actions = self.getBestActions(q, state, actions, tolerance)
        # print(actions)
        return actions

    def getAction(self, state):
        if rd.random() < self.epsilon or self.isRandom:
            return rd.choice(self.actions)
        else:
            best_actions = self.selectBestAction(state)
            if best_actions:
                return best_actions[0]
            else:
                return rd.choice(self.actions)

    def updateQValue(self, q, state, action, reward, next_state, optimal_action=None):
        # Hash the state if it's a list (to use as a dict key)
        # Flatten state and next_state if they are lists of lists, then hash as tuple
        qvalues = self.getQValues(q, state)
        if action not in qvalues:
            for a in self.actions:
                qvalues[a] = 0.0
        # max_next_q accounts for optimal_action if not None, else takes max
        max_next_q = max(self.getQValues(q, next_state).values(), default=0)
        if optimal_action is not None:
            max_next_q = max(max_next_q, self.getQValues(q, next_state).get(optimal_action, 0))
        qvalues[action] += self.alpha * (reward + self.gamma * max_next_q - qvalues[action])
        self.Q[q][state] = qvalues

        if self.decay_method == "linear":
            self.epsilon -= self.epsilon_decay
        elif self.decay_method == "exponential":
            self.epsilon *= self.epsilon_decay
        self.epsilon = max(0, self.epsilon)

    def updateQFunctions(self, state, action, signals, next_state):
        # print(signals, action)
        for q in self.preferences:
            if q not in signals:
                raise ValueError(f"Signal '{q}' not found in signals. Available signals: {list(signals.keys())}")
            self.updateQValue(q, state, action, signals[q], next_state)