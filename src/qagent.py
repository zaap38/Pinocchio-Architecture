import copy as cp
import random as rd


class QAgent:

    def __init__(self, name):
        self.name = name

        self.qfunctions = {}  # Q-functions
        self.preferences = []  # [Q_a, Q_b, Q_c] <=> Q_a > Q_b > Q_c

        self.actions = []

        self.decay_method = "linear"
        self.epsilon = 1.0
        self.epsilon_decay = 0
        self.alpha = 0.1
        self.gamma = 0.9

        self.selection_method = "lex"

    def set_actions(self, actions):
        self.actions = actions
    
    def get_qvalues(self, qfunction, state):
        return self.qvalues.get(state, {})
    
    def set_preferences(self, preferences):
        self.preferences = preferences
        for q in preferences:
            if q not in self.qfunctions:
                raise ValueError(f"Q-function '{q}' does not exist. Existing Q-functions: {list(self.qfunctions.keys())}")
    
    def add_qfunction(self, name):
        if name not in self.Q:
            self.Q[name] = {}
            self.preferences.append(name)
        else:
            raise ValueError(f"Q-function '{name}' already exists.")
        
    def get_best_actions(self, qfunction, state, actions=None):
        if actions is None:
            actions = cp.deepcopy(self.actions)
        # return the set of optimal actions
        qvalues = self.get_qvalues(qfunction, state)
        if not qvalues:
            return None  # No actions found for this state
        max_value = max(qvalues.values())
        return [action for action, value in qvalues.items() if value == max_value]
    
    def select_best_action(self, state):
        if self.selection_method == "lex":  # lexicographic
            return self.lexicographic(state)
        if self.selection_method == "tlex":  # threshold-lexicographic
            return self.threshold_lexicographic(state)
        if self.selection_method == "dlex":  # delta-lexicographic
            return self.delta_lexicographic(state)
    
    def lexicographic(self, state):
        actions = cp.deepcopy(self.actions)
        for q in self.preferences:
            actions = self.get_best_actions(self.qfunctions[q], state, actions)
        return actions

    def threshold_lexicographic(self, state):
        NotImplementedError("Threshold-lexicographic selection method is not implemented.")

    def delta_lexicographic(self, state):
        NotImplementedError("Delta-lexicographic selection method is not implemented.")
    
    def get_action(self, state):
        if rd.random() < self.epsilon:
            # Explore: choose a random action
            return rd.choice(self.actions)
        else:
            # Exploit: choose the best action based on Q-values
            best_actions = self.select_best_action(state)
            if best_actions:
                return best_actions[0]  # always the first action to avoid stochasticity
            else:
                return rd.choice(self.actions)  # fallback to random action if no best actions found
    

    
    