from qagent import QAgent
import copy as cp
import random as rd
from af import *


class ConstitutiveNorm:

    def __init__(self):
        # C(a, b | c): In context 'c', 'a' counts as 'b'
        self.context = []  # 'c', empty is tautology
        self.premise = []  # 'a'
        self.conclusion = []  # 'b'
        self.description = ""  # description of the norm

    def __str__(self):
        if not self.context:
            return f"{self.type}({self.premise}, {self.conclusion})"
        else:
            return f"{self.type}({self.premise}, {self.conclusion} | {self.context})"


class RegulativeNorm:

    def __init__(self):
        # X(a | b): In context 'b', it is X to do 'a'
        self.type = "F"  # F/P/O
        self.context = []  # 'b', empty is tautology
        self.premise = []  # 'a'

    def isProhibition(self):
        return self.type == "F"
    
    def isPermission(self):
        return self.type == "P"
    
    def isObligation(self):
        return self.type == "O"
    
    def __str__(self):
        if not self.context:
            return f"{self.type}({self.premise})"
        else:
            return f"{self.type}({self.premise} | {self.context})"


class Stakeholder:

    def __init__(self, name="no_name"):
        self.name = name
        self.c_norms = {}
        self.afs = {}

    def addNorm(self, norm):
        self.c_norms[norm.name] = []
        self.afs[norm.name] = AF()

    def setConstitutiveNorms(self, rnorm, cnorms):
        normName = str(rnorm)
        if normName in self.c_norms:
            self.c_norms[normName].extend(cnorms)
        else:
            raise ValueError(f"Norm '{normName}' does not exist in stakeholder '{self.name}'. (In setConstitutiveNorms)")

    def setArguments(self, rnorm, arguments):
        normName = str(rnorm)
        if normName in self.afs:
            for arg in arguments:
                self.afs[normName].addArgument(arg)
        else:
            raise ValueError(f"Norm '{normName}' does not exist in stakeholder '{self.name}'. (In setArguments)")

    def setAttacks(self, rnorm, attacks):
        normName = str(rnorm)
        if normName in self.afs:
            for attacker, attacked in attacks.items():
                for attacked_arg in attacked:
                    self.afs[normName].addAttack(attacker, attacked_arg)
        else:
            raise ValueError(f"Norm '{normName}' does not exist in stakeholder '{self.name}'. (In setAttacks)")


class Pinocchio:

    def __init__(self, name="no_name"):
        self.name = name
        self.agent = QAgent(name)
        self.stakeholders = []
        self.norms = []

    def getQValues(self, qfunction, state):
        return self.agent.getQValues(qfunction, state)

    def addStakeholder(self, stakeholder):
        self.stakeholders.append(stakeholder)

    def getAction(self, state):
        return self.agent.getAction(state)
    
    def selectBestAction(self, state):
        return self.agent.selectBestAction(state)
                                           
    def updateQValue(self, q, state, action, reward, next_state, optimal_action=None):
        self.agent.updateQValue(q, state, action, reward, next_state, optimal_action)

    def updateQFunctions(self, state, action, signals, next_state):
        self.agent.updateQFunctions(state, action, signals, next_state)

    def setActions(self, actions):
        self.agent.setActions(actions)

    def loadOptimalAgent(self, steps):
        self.agent = QAgent(self.name)
        self.agent.addQFunction("R")
        self.agent.initDecay(steps)

    def setSteps(self, steps):
        self.agent.initDecay(steps)

    def addNorm(self, norm):
        # add regulative norm
        self.norms.append(norm)