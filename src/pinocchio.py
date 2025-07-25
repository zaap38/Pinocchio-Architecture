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


class RegulativeNorm:

    def __init__(self):
        # X(a | b): In context 'b', it is X to do 'a'
        self.name = "no_name"
        self.type = "F"  # F/P/O
        self.context = []  # 'b', empty is tautology
        self.conclusion = []  # 'a'


class Stakeholder:

    def __init__(self):
        self.name = ""
        self.c_norms = {}
        self.afs = {}

    def addNorm(self, norm):
        self.c_norms[norm.name] = []
        self.afs[norm.name] = AF()

    def setConstitutiveNorms(self, normName, cnorms):
        if type(normName) == ConstitutiveNorm:
            normName = normName.name
        if normName in self.c_norms:
            self.c_norms[normName] = cnorms
        else:
            raise ValueError(f"Norm '{normName}' does not exist in stakeholder '{self.name}'. (In setConstitutiveNorms)")

    def setArguments(self, normName, arguments):
        if type(normName) == RegulativeNorm:
            normName = normName.name
        if normName in self.afs:
            self.afs[normName].arguments = arguments
        else:
            raise ValueError(f"Norm '{normName}' does not exist in stakeholder '{self.name}'. (In setArguments)")

    def setAttacks(self, normName, attacks):
        if type(normName) == RegulativeNorm:
            normName = normName.name
        if normName in self.afs:
            for attacker, attacked in attacks.items():
                for attacked_arg in attacked:
                    self.afs[normName].addAttack(attacker, attacked_arg)
        else:
            raise ValueError(f"Norm '{normName}' does not exist in stakeholder '{self.name}'. (In setAttacks)")


class Pinocchio:

    def __init__(self):
        self.agent = QAgent()
        self.stakeholders = []
        self.norms = []

    def addStakeholder(self, stakeholder):
        self.stakeholders.append(stakeholder)