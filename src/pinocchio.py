from qagent import QAgent
import copy as cp
import random as rd
from af import *


class ConstitutiveNorm:

    def __init__(self, prem=[], conc=[], context=[]):
        # C(a, b | c): In context 'c', 'a' counts as 'b'

        if type(prem) is str:
            prem = [prem]
        if type(conc) is str:
            conc = [conc]
        if type(context) is str:
            context = [context]

        self.context = context  # 'c', empty is tautology
        self.premise = prem  # 'a'
        self.conclusion = conc  # 'b'
        self.description = ""  # description of the norm

    def __str__(self):
        if not self.context:
            return f"C({self.premise}, {self.conclusion})"
        else:
            return f"C({self.premise}, {self.conclusion} | {self.context})"


class RegulativeNorm:

    def __init__(self, t='F', prem=[], context=[]):
        # X(a | b): In context 'b', it is X to do 'a'

        if type(prem) is str:
            prem = [prem]
        if type(context) is str:
            context = [context]

        self.type = t  # F/P/O
        self.context = context  # 'b', empty is tautology
        self.premise = prem  # 'a'

        self.weight = 1.0

    def isProhibition(self):
        return self.type == "F"
    
    def isPermission(self):
        return self.type == "P"
    
    def isObligation(self):
        return self.type == "O"
    
    def comply(self, facts):
        premisesInFacts = True
        for premise in self.premise:
            if premise not in facts:
                premisesInFacts = False
                break
        if self.isProhibition():
            return not premisesInFacts
        if self.isObligation():
            return premisesInFacts
    
    def __str__(self):
        if not self.context:
            return f"{self.type}({self.premise})"
        else:
            return f"{self.type}({self.premise} | {self.context})"


class Stakeholder:

    def __init__(self, name="no_name"):
        self.name = name
        self.c_norms = {}  # cnorms for each regulative norm
        self.afs = {}  # afs for each regulative norm

    def addNorm(self, rnorm):
        normName = str(rnorm)
        self.c_norms[normName] = []
        self.afs[normName] = AF()

    def addConstitutiveNorm(self, rnorm, cnorm):
        normName = str(rnorm)
        if normName in self.c_norms:
            self.c_norms[normName].append(cnorm)
        else:
            raise ValueError(f"Norm '{normName}' does not exist in stakeholder '{self.name}'. (In addConstitutiveNorm)")

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
            for attack in attacks:
                self.afs[normName].addAttack(attack[0], attack[1])
        else:
            raise ValueError(f"Norm '{normName}' does not exist in stakeholder '{self.name}'. (In setAttacks)")
        
    def closure(self, rnorm, facts):
        normName = str(rnorm)
        factSize = len(facts)
        stop = False
        while not stop:
            facts = self.closureStep(normName, facts)
            if len(facts) == factSize:
                stop = True
            factSize = len(facts)
        return facts

    def closureStep(self, rnorm, facts):
        normName = str(rnorm)
        new_facts = cp.deepcopy(facts)
        for cnorm in self.c_norms[normName]:
            premiseInFacts = True
            for premise in cnorm.premise:
                if premise not in facts:
                    premiseInFacts = False
                    break
            if premiseInFacts:
                new_facts.extend(cnorm.conclusion)
        return list(set(new_facts))
    
    def getActiveArguments(self, rnorm, facts):
        normName = str(rnorm)
        if normName in self.afs:
            arguments = []
            for arg in self.afs[normName].arguments:
                if arg in facts:
                    arguments.append(arg)
            return arguments
        else:
            raise ValueError(f"Norm '{normName}' does not exist in stakeholder '{self.name}'. (In getArguments)")


class Pinocchio:

    def __init__(self, name="no_name"):
        self.name = name
        self.agent = QAgent(name)
        self.stakeholders = []
        self.norms = []
        self.facts = {}
        self.override = {}
        self.responsible = {}  # whether the agent is responsible for its actions

    def judge(self, state, flags, debug=False):
        # add all rnorms to the facts
        facts = []
        for rnorm in self.norms:
            facts.append(str(rnorm))
        # apply the epsilon function to get the facts
        facts.extend(self.epsilon(state, flags))
        if debug:
            # print("JUDGES", self.name, id(self.agent))
            # print("Inv.:", self.getInventory())
            # print("Last action:", self.getLastAction())
            # print("Flags:", flags)
            # print("Brute:", facts)
            inst = {}
            for rnorm in self.norms:
                inst[str(rnorm)] = []
                for stakeholder in self.stakeholders:
                    inst[str(rnorm)].extend(stakeholder.closure(rnorm, facts))
                inst[str(rnorm)] = list(set(inst[str(rnorm)]))
                # print("Inst.", str(rnorm), ":", inst[str(rnorm)])
            # for each norm
        # get the activate arguments, and combine the AFs of each stakeholders
        # then judges
        violations = {}
        noncompliances = {}
        defeats = {}
        for rnorm in self.norms:

            violations[str(rnorm)] = 0
            af = AF()
            all_facts = []
            all_attacks = []
            all_active = []
            for stakeholder in self.stakeholders:
                fact_closure = stakeholder.closure(rnorm, facts)
                all_facts.extend(fact_closure)
                active_args = stakeholder.getActiveArguments(rnorm, fact_closure)
                for arg in active_args:
                    af.addArgument(arg)
                    if arg not in all_active:
                        all_active.append(arg)
                for attack in stakeholder.afs[str(rnorm)].getAttacks():
                    if attack not in all_attacks:
                        all_attacks.append(attack)
            for attack in all_attacks:
                if attack[0] in all_active and attack[1] in all_active:
                    af.addAttack(attack)
            
            # compute the extension
            extension = af.computeExtension("grounded")
            didViolation = False
            # print(all_facts, extension, "Comply:", rnorm.comply(all_facts))
            normActive = str(rnorm) in extension
            if str(rnorm) in self.override:
                normActive = self.override[str(rnorm)]

            defeats[str(rnorm)] = -1 if not normActive else 0  # maybe add the condition for the header to be in the facts
            if not rnorm.comply(all_facts):
                noncompliances[str(rnorm)] = -1

            if normActive and not rnorm.comply(all_facts):
                violations[str(rnorm)] = -rnorm.weight
                didViolation = True
            if debug:
                print("Avoidance condition:", rnorm.comply(all_facts), end=' | ')
                print("Responsible:", self.responsible[str(rnorm)])
                print("Violates", str(rnorm), ":", didViolation, '| Extension:', extension)
                pass
            
        return {'V': violations, 'A': noncompliances, 'D': defeats}
    
    def addFact(self, fact_name, fun):
        if fact_name not in self.facts:
            self.facts[fact_name] = fun
        else:
            raise ValueError(f"Fact '{fact_name}' already exists in Pinocchio '{self.name}'.")
    
    def epsilon(self, state, flags):
        facts = []
        for fact_item in self.facts:
            if self.facts[fact_item](state, flags):
                facts.append(fact_item)
        return facts
    
    def overrideJudgement(self, norm_name, value):
        self.override[norm_name] = value

    def clearOverrides(self):
        self.override = {}

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

    def updateQFunctions(self, state, action, signals, next_state, optimal_action=None):
        self.agent.updateQFunctions(state, action, signals, next_state, optimal_action)

    def setActions(self, actions):
        self.agent.setActions(actions)

    def loadOptimalAgent(self, steps):
        self.agent = QAgent(self.name)
        self.agent.addQFunction("R")
        self.agent.initDecay(steps)

    def loadNormativeAgent(self, steps):
        self.agent = QAgent(self.name)
        self.agent.addQFunction("V")
        self.agent.addQFunction("R")
        self.agent.initDecay(steps)
        self.agent.selection_method = "dlex"

    def loadNonAvoidantAgent(self, steps):
        self.agent = QAgent(self.name)
        self.agent.addQFunction("V")
        self.agent.addQFunction("A")
        self.agent.addQFunction("R")
        self.agent.addQFunction("D", no_ordering=True)
        self.agent.initDecay(steps)
        self.agent.selection_method = "lex"

    def loadAvoidantAgent(self, steps):
        self.agent = QAgent(self.name)
        self.agent.addQFunction("V")
        self.agent.addQFunction("A", no_ordering=True)
        self.agent.addQFunction("R")
        self.agent.addQFunction("D", no_ordering=True)
        self.agent.initDecay(steps)
        self.agent.selection_method = "lex"

    def setSteps(self, steps):
        self.agent.initDecay(steps)

    def addNorm(self, norm):
        # add regulative norm
        self.norms.append(norm)
        self.responsible[str(norm)] = False

    def getInventory(self):
        return self.agent.getInventory()
    
    def addItemToInventory(self, item):
        self.agent.addItemToInventory(item)

    def removeItemFromInventory(self, item):
        self.agent.removeItemFromInventory(item)

    def addItemsToInventory(self, items):
        for item in items:
            self.addItemToInventory(item)

    def removeItemsFromInventory(self, items):
        for item in items:
            self.removeItemFromInventory(item)

    def resetInventory(self):
        self.agent.resetInventory()

    def getLastAction(self):
        return self.agent.getLastAction()
    
    def setLastAction(self, action):
        self.agent.setLastAction(action)

    def getLastSignal(self):
        return self.agent.getLastSignal()
    
    def setLastSignal(self, signal):
        self.agent.setLastSignal(signal)

    def printQFunctions(self, state, rounding=3):
        rnorms = []
        for norm in self.norms:
            if type(norm) is RegulativeNorm:
                rnorms.append(str(norm))
        self.agent.printQFunctions(state, rnorms)

    def setOptimal(self, optimal):
        self.agent.optimal = optimal

    def setLearning(self, learning):
        self.agent.learning = learning

    def isOptimal(self):
        return self.agent.optimal
    
    def has(self, item):
        return self.agent.has(item)
    
    def updateResponsible(self, state, action):
        possible = []
        qvs = self.getQValues('D', state)
        for q in qvs:
            if qvs[q] == max(qvs.values()):
                possible.append(q)
        if action not in possible:
            self.responsible[state[1]] = True
