from pinocchio import *

import time
import copy as cp
import random as rd

import matplotlib.pyplot as plt  # type: ignore
from tqdm import tqdm  # type: ignore

import facts as funfacts

WALL = 0
ROAD = 1
PLAIN = 2
PACMAN = 3
OBJECT = 4


SYMBOLS = {
    WALL: "#",
    ROAD: " ",
    PLAIN: "-",
    PACMAN: "O"
}


class Cell:

    def __init__(self):
        self.pos = [0, 0]  # [x, y]
        self.type = PLAIN
        self.symbol = "?"  # only for objects
        self.labels = []

    def setType(self, type):
        self.type = type

    def setPos(self, pos):
        self.pos = pos

    def addLabel(self, label):
        if label not in self.labels:
            self.labels.append(label)

    def removeLabel(self, label):
        if label in self.labels:
            self.labels.remove(label)

    def getSymbol(self):
        if self.type in SYMBOLS:
            return SYMBOLS[self.type]
        return self.symbol


class Environment:

    def __init__(self):
        self.width = 0
        self.height = 0
        self.grid = []
        self.objects = {}

        self.stochasticity = 0.1  # probability of random action

        self.agents = []
        self.pos = {}  # position of each agent

        self.steps = 1000
        self.timeout = 30  # steps
        self.loadedPreset = ""
        self.iterations = 0  # number of steps since last reset

        self.historic = []

        self.doAction = self.doAction_1  # default action method

        self.debug = False
        self.debug_judgement = False  # debug the judgement of the agents

    def addAgent(self, agent):
        self.agents.append(agent)
        self.pos[agent.name] = [1, 1]  # default position, can be changed later

    def setSize(self, width, height):
        self.width = width
        self.height = height
        self.grid = [[Cell() for _ in range(width)] for _ in range(height)]
        for y in range(height):
            for x in range(width):
                self.grid[y][x].setPos([x, y])

    def loadFile(self, filename):
        with open(filename, 'r') as file:
            lines = file.readlines()
            self.setSize(len(lines[0].strip()), len(lines))
            for y, line in enumerate(lines):
                for x, char in enumerate(line.strip()):
                    if char == SYMBOLS[WALL]:
                        self.grid[y][x].setType(WALL)
                    elif char == SYMBOLS[ROAD]:
                        self.grid[y][x].setType(ROAD)
                    elif char == SYMBOLS[PLAIN]:
                        self.grid[y][x].setType(PLAIN)

    def loadPreset(self, presetName, reset_agent=True):
        self.loadedPreset = presetName
        if presetName == "taxi":
            self.loadTaxi(reset_agent)
        elif presetName == "pacman":
            self.loadPacman(reset_agent)
        elif presetName == "adam":
            self.loadAdam(reset_agent)
        elif presetName == "mini_taxi":
            self.loadMiniTaxi(reset_agent)

    def loadMiniTaxi(self, reset_agent=True):

        self.window = 40
        
        if reset_agent:
            self.steps = 20000
            self.timeout = 30
            self.loadFile("src/environments/taxi_5x5.txt")
            taxi = Pinocchio("Taxi")
            # taxi.loadOptimalAgent(self.steps)
            taxi.loadNormativeAgent(self.steps)
            self.agents.append(taxi)

            # r norms
            r1 = RegulativeNorm("F", "pavement")
            r2 = RegulativeNorm("F", "speeding")
            r3 = RegulativeNorm("F", "stop", "road")

            taxi.addNorm(r1)
            taxi.addNorm(r2)
            taxi.addNorm(r3)

            # c norms Taxi
            ct1 = ConstitutiveNorm("role(taxi)")
            ct2 = ConstitutiveNorm("not_service", "not_service")
            ct3 = ConstitutiveNorm("distance_>_time", "late")
            ct4 = ConstitutiveNorm("no_traffic", "no_traffic")

            # c norms Law
            cl1 = ConstitutiveNorm("dist_parking_<_2", "parking_near")
            cl2 = ConstitutiveNorm("in_city", "no_exception")

            # facts
            taxi.addFact("pavement", lambda state, flags:
                state["grid"][state["pos"]["Taxi"][1]][state["pos"]["Taxi"][0]] == PLAIN)
            taxi.addFact("road", lambda state, flags:
                state["grid"][state["pos"]["Taxi"][1]][state["pos"]["Taxi"][0]] == ROAD)
            taxi.addFact("speeding", lambda state, flags: state["actions"].get(taxi.name)[1] == "fast")
            taxi.addFact("stop", lambda state, flags: "pick" in flags or "drop" in flags)
            taxi.addFact("role(taxi)", lambda state, flags: True)
            taxi.addFact("not_service", lambda state, flags: state["daytime"] in ["morning", "night"])
            taxi.addFact("distance_>_time", lambda state, flags: "passenger" in state["inventory"][taxi.name] and \
                         state["daytime"] == "evening")
            # taxi.addFact("no_traffic", lambda state, flags: len(state["pos"]) < 3)
            taxi.addFact("dist_parking_<_2", funfacts.parking_close)
            # taxi.addFact("in_city", lambda state, flags: True)

            attacks_r1 = []
            attacks_r2 = [("late", str(r2)), ("no_traffic", str(r2)), ("no_exception", "late"), ("no_exception", "no_traffic")]
            attacks_r3 = [("role(taxi)", str(r3)), ("not_service", "role(taxi)"), ("parking_near", "role(taxi)")]

            # stakeholders
            taxi_sh = Stakeholder("Taxi")
            taxi_sh.addNorm(r1)
            taxi_sh.addNorm(r2)
            taxi_sh.addNorm(r3)
            taxi_sh.addConstitutiveNorm(r2, ct3)
            taxi_sh.addConstitutiveNorm(r2, ct4)
            taxi_sh.addConstitutiveNorm(r3, ct1)
            taxi_sh.addConstitutiveNorm(r3, ct2)

            taxi_sh.afs[str(r1)] = AF()
            taxi_sh.afs[str(r2)] = AF()
            taxi_sh.afs[str(r3)] = AF()

            taxi_sh.setArguments(str(r1), [str(r1)])
            taxi_sh.setArguments(str(r2), [str(r2), "no_traffic", "late"])
            taxi_sh.setAttacks(str(r2), attacks_r2)
            taxi_sh.setArguments(str(r3), [str(r3), "not_service", "role(taxi)"])
            taxi_sh.setAttacks(str(r3), attacks_r3)

            law = Stakeholder("Law")
            law.addNorm(r1)
            law.addNorm(r2)
            law.addNorm(r3)
            law.addConstitutiveNorm(r2, cl2)
            law.addConstitutiveNorm(r3, cl1)

            law.afs[str(r1)] = AF()
            law.afs[str(r2)] = AF()
            law.afs[str(r3)] = AF()
            law.setArguments(str(r1), [str(r1)])
            law.setArguments(str(r2), [str(r2), "no_exception"])
            law.setAttacks(str(r2), attacks_r2)
            law.setArguments(str(r3), [str(r3), "parking_near"])
            law.setAttacks(str(r3), attacks_r3)

            taxi.addStakeholder(taxi_sh)
            taxi.addStakeholder(law)

        movements = ["up", "down", "left", "right"]
        speeds = ["slow", "fast"]
        # speeds = ["slow"]
        actions = []
        for m in movements:
            for s in speeds:
                actions.append((m, s))

        self.doAction = self.doAction_2  # change action method for taxi

        self.objects["parking"] = self.makeObject()
        self.objects["parking"]["pos"] = [3, 1]
        self.objects["parking"]["symbol"] = "P"
        self.objects["parking"]["flags"] = ["parked", "pick"]
        self.objects["parking"]["reward"] = -5
        self.objects["parking"]["inv_add"] = ["passenger"]
        self.objects["parking"]["condition"] = ["not-passenger"]

        self.objects["street"] = self.makeObject()
        self.objects["street"]["pos"] = [2, 1]
        self.objects["street"]["symbol"] = "S"
        self.objects["street"]["flags"] = ["pick"]
        self.objects["street"]["inv_add"] = ["passenger"]
        self.objects["street"]["condition"] = ["not-passenger"]

        self.objects["destination"] = self.makeObject()
        self.objects["destination"]["pos"] = [1, 3]
        self.objects["destination"]["symbol"] = "D"
        self.objects["destination"]["reward"] = 100
        self.objects["destination"]["inv_rem"] = ["passenger"]
        self.objects["destination"]["condition"] = ["passenger"]
        self.objects["destination"]["flags"] = ["drop"]

        for agent in self.agents:
            agent.resetInventory()
            agent.setActions(actions)
            # agent.isRandom = True  # comment this
            self.setPos(agent, [1, 1])

    def loadTaxi(self, reset_agent=True):

        self.window = 600
        
        if reset_agent:
            self.steps = 2000000
            self.timeout = 60
            self.loadFile("src/environments/taxi_10x10.txt")
            taxi = Pinocchio("Taxi")
            # taxi.loadOptimalAgent(self.steps)
            taxi.loadNormativeAgent(self.steps)
            self.agents.append(taxi)

            # r norms
            r1 = RegulativeNorm("F", "pavement")
            r2 = RegulativeNorm("F", "speeding")
            r3 = RegulativeNorm("F", "stop", "road")

            taxi.addNorm(r1)
            taxi.addNorm(r2)
            taxi.addNorm(r3)

            # c norms Taxi
            ct1 = ConstitutiveNorm("role(taxi)")
            ct2 = ConstitutiveNorm("morning", "not_service")
            ct3 = ConstitutiveNorm("night", "not_service")
            ct4 = ConstitutiveNorm(["evening", "has_passenger"], "late")
            ct5 = ConstitutiveNorm("no_traffic", "no_traffic")
            ct6 = ConstitutiveNorm("time_0-10", "morning")
            ct7 = ConstitutiveNorm("time_11-20", "day")
            ct8 = ConstitutiveNorm("time_21-30", "evening")
            ct9 = ConstitutiveNorm("time_31-40", "evening")
            ct10 = ConstitutiveNorm("time_41-50", "night")
            ct11 = ConstitutiveNorm("time_51-60", "night")

            # c norms Law
            cl1 = ConstitutiveNorm("dist_parking_<_4", "parking_near")
            cl2 = ConstitutiveNorm("in_city", "no_exception")

            # facts
            taxi.addFact("pavement", lambda state, flags:
                state["grid"][state["pos"]["Taxi"][1]][state["pos"]["Taxi"][0]] == PLAIN)
            taxi.addFact("road", lambda state, flags:
                state["grid"][state["pos"]["Taxi"][1]][state["pos"]["Taxi"][0]] == ROAD)
            taxi.addFact("speeding", lambda state, flags: state["actions"].get(taxi.name)[1] == "fast")
            taxi.addFact("stop", lambda state, flags: "pick" in flags or "drop" in flags)
            taxi.addFact("role(taxi)", lambda state, flags: True)
            taxi.addFact("has_passenger", lambda state, flags: "passenger" in state["inventory"][taxi.name])
            taxi.addFact("time_0-10", lambda state, flags: state["iterations"] <= 10)
            taxi.addFact("time_11-20", lambda state, flags: 10 < state["iterations"] <= 20)
            taxi.addFact("time_21-30", lambda state, flags: 20 < state["iterations"] <= 30)
            taxi.addFact("time_31-40", lambda state, flags: 30 < state["iterations"] <= 40)
            taxi.addFact("time_41-50", lambda state, flags: 40 < state["iterations"] <= 50)
            taxi.addFact("time_51-60", lambda state, flags: 50 < state["iterations"])
            taxi.addFact("dist_parking_<_4", funfacts.parking_close)
            # taxi.addFact("in_city", lambda state, flags: True)

            attacks_r1 = []
            attacks_r2 = [("late", str(r2)), ("no_traffic", str(r2)), ("no_exception", "late"), ("no_exception", "no_traffic")]
            attacks_r3 = [("role(taxi)", str(r3)), ("not_service", "role(taxi)"), ("parking_near", "role(taxi)")]

            # stakeholders
            taxi_sh = Stakeholder("Taxi")
            taxi_sh.addNorm(r1)
            taxi_sh.addNorm(r2)
            taxi_sh.addNorm(r3)

            taxi_sh.addConstitutiveNorm(r2, ct4)
            taxi_sh.addConstitutiveNorm(r2, ct5)
            taxi_sh.addConstitutiveNorm(r2, ct8)
            taxi_sh.addConstitutiveNorm(r2, ct9)

            taxi_sh.addConstitutiveNorm(r3, ct1)
            taxi_sh.addConstitutiveNorm(r3, ct2)
            taxi_sh.addConstitutiveNorm(r3, ct3)
            taxi_sh.addConstitutiveNorm(r3, ct6)
            taxi_sh.addConstitutiveNorm(r3, ct10)
            taxi_sh.addConstitutiveNorm(r3, ct11)

            taxi_sh.afs[str(r1)] = AF()
            taxi_sh.afs[str(r2)] = AF()
            taxi_sh.afs[str(r3)] = AF()

            taxi_sh.setArguments(str(r1), [str(r1)])
            taxi_sh.setArguments(str(r2), [str(r2), "no_traffic", "late"])
            taxi_sh.setAttacks(str(r2), attacks_r2)
            taxi_sh.setArguments(str(r3), [str(r3), "not_service", "role(taxi)"])
            taxi_sh.setAttacks(str(r3), attacks_r3)

            law = Stakeholder("Law")
            law.addNorm(r1)
            law.addNorm(r2)
            law.addNorm(r3)
            law.addConstitutiveNorm(r2, cl2)
            law.addConstitutiveNorm(r3, cl1)

            law.afs[str(r1)] = AF()
            law.afs[str(r2)] = AF()
            law.afs[str(r3)] = AF()
            law.setArguments(str(r1), [str(r1)])
            law.setArguments(str(r2), [str(r2), "no_exception"])
            law.setAttacks(str(r2), attacks_r2)
            law.setArguments(str(r3), [str(r3), "parking_near"])
            law.setAttacks(str(r3), attacks_r3)

            taxi.addStakeholder(taxi_sh)
            taxi.addStakeholder(law)

        movements = ["up", "down", "left", "right"]
        speeds = ["slow", "fast"]
        actions = []
        for m in movements:
            for s in speeds:
                actions.append((m, s))

        self.doAction = self.doAction_2  # change action method for taxi

        self.objects["parking"] = self.makeObject()
        self.objects["parking"]["pos"] = [7, 3]
        self.objects["parking"]["symbol"] = "P"
        self.objects["parking"]["flags"] = ["parked", "pick"]
        self.objects["parking"]["reward"] = -5
        self.objects["parking"]["inv_add"] = ["passenger"]
        self.objects["parking"]["condition"] = ["not-passenger", "not-dropped"]

        self.objects["street"] = self.makeObject()
        self.objects["street"]["pos"] = [5, 3]
        self.objects["street"]["symbol"] = "S"
        self.objects["street"]["flags"] = ["pick"]
        self.objects["street"]["inv_add"] = ["passenger"]
        self.objects["street"]["condition"] = ["not-passenger", "not-dropped"]

        self.objects["destination"] = self.makeObject()
        self.objects["destination"]["pos"] = [6, 8]
        self.objects["destination"]["symbol"] = "D"
        self.objects["destination"]["reward"] = 100
        self.objects["destination"]["inv_add"] = ["dropped"]
        self.objects["destination"]["inv_rem"] = ["passenger"]
        self.objects["destination"]["condition"] = ["passenger"]
        self.objects["destination"]["flags"] = ["drop"]

        for agent in self.agents:
            agent.resetInventory()
            agent.setActions(actions)
            # agent.isRandom = True  # comment this
            self.setPos(agent, [1, 1])
            # self.setPos(agent, [rd.randint(1, self.width - 2), rd.randint(1, self.height - 2)])  # default position

    def loadPacman(self, reset_agent=True):

        self.window = 50
        
        if reset_agent:
            self.steps = 10000
            self.timeout = 20
            self.loadFile("src/environments/basic_5x5.txt")
            pacman = Pinocchio("Pacman")
            pacman.loadOptimalAgent(self.steps)
            self.agents.append(pacman)

        actions = ["up", "down", "left", "right"]
        for agent in self.agents:
            agent.resetInventory()
            agent.setActions(actions)
            # agent.isRandom = True  # comment this
            self.setPos(agent, [1, 1])  # default position

    def loadAdam(self, reset_agent=True):

        self.window = 200

        if reset_agent:
            self.steps = 30000
            self.timeout = 10
            self.loadFile("src/environments/apple_7x7.txt")
            adam = Pinocchio("Adam")
            #adam.loadOptimalAgent(self.steps)
            adam.loadNormativeAgent(self.steps)

            # regulative norms
            r1 = RegulativeNorm("F", "knowledge")
            adam.addNorm(r1)

            # add fact functions
            adam.addFact("eat", lambda state, flags: "eat" in flags)
            adam.addFact("longtime", lambda state, flags: state["iterations"] > 5)

            # constitutive norms
            c1 = ConstitutiveNorm("eat", "knowledge")
            c2 = ConstitutiveNorm("longtime", "hungry")

            sh = []  # stakeholders

            god = Stakeholder("God")
            god.addNorm(r1)
            god.addConstitutiveNorm(r1, c1)
            god.afs[str(r1)] = AF()
            god.setArguments(str(r1), [str(r1)])
            sh.append(god)

            user = Stakeholder("User")
            user.addNorm(r1)
            user.addConstitutiveNorm(r1, c2)
            user.afs[str(r1)] = AF()
            user.setArguments(str(r1), [str(r1), "hungry"])
            user.setAttacks(str(r1), [("hungry", str(r1))])
            sh.append(user)

            for s in sh:
                adam.addStakeholder(s)
            self.agents.append(adam)

        self.objects["apple"] = self.makeObject()
        self.objects["apple"]["pos"] = [3, 3]
        self.objects["apple"]["symbol"] = "A"
        self.objects["apple"]["flags"] = ["eat"]
        self.objects["apple"]["reward"] = 10

        actions = ["up", "down", "left", "right"]
        for agent in self.agents:
            agent.resetInventory()
            agent.setActions(actions)
            # agent.isRandom = True  # comment this
            self.setPos(agent, [2, 2])

    def makeObject(self):
        obj = {}
        obj["pos"] = [0, 0]
        obj["symbol"] = "?"
        obj["flags"] = []
        obj["global_flags"] = []
        obj["reward"] = 0
        obj["permanent"] = False
        obj["inv_add"] = []
        obj["inv_rem"] = []
        obj["condition"] = []
        return obj

    def setSteps(self, steps):
        self.steps = steps
        for agent in self.agents:
            agent.setSteps(steps)

    def setPos(self, agent_name, pos):
        if type(agent_name) != str:
            agent_name = agent_name.name
        self.pos[agent_name] = pos

    def display(self):
        cp_grid = cp.deepcopy(self.grid)
        for agent in self.agents:
            pos = self.pos[agent.name]
            cp_grid[pos[1]][pos[0]].setType(PACMAN)
        for obj_name, obj in self.objects.items():
            pos = obj["pos"]
            cp_grid[pos[1]][pos[0]].setType(OBJECT)
            cp_grid[pos[1]][pos[0]].symbol = obj["symbol"]
        for row in cp_grid:
            print("".join(cell.getSymbol() for cell in row))

    def run(self, display=False, run_title=""):

        if run_title == "":
            run_title = f"Run {len(self.historic) + 1}"
        logs = []

        start_time = time.time()

        self.iterations = 0
        reset = True
        i = 0
        pbar = None
        if not display:
            pbar = tqdm(total=self.steps, desc=run_title)

        while i < self.steps or not reset:
            i += 1
            reset = False
            if pbar is not None:
                pbar.update(1)
            if display and self.iterations == 0:
                print("=========VVVVV=========VVVVV=========")
            log = self.step()
            logs.append(log)
            if display:
                print(f"Run '{run_title}': Iteration {i + 1}/{self.steps} - Step {self.iterations + 1}/{self.timeout}")
                for agent in self.agents:
                    print(f"{agent.name}: Action={agent.getLastAction()}  Signal={agent.getLastSignal()}  Inv: {agent.getInventory()}")
                self.display()
            self.iterations += 1
            if self.iterations >= self.timeout:  # reset the agent every X steps
                self.loadPreset(self.loadedPreset, reset_agent=False)
                self.iterations = 0
                reset = True

        end_time = time.time()

        # movingAverage of the tracked Q-Functions
        evolution = {}
        qfunctions = ['R', 'V']
        window = self.window
        if window > self.steps:
            window = int(self.steps / 20)
        for q in qfunctions:
            evolution[q] = self.movingAverage([log[q] for log in logs if isinstance(log, dict) and q in log], window)

        run_hist = {}
        run_hist["title"] = run_title
        run_hist["id"] = len(self.historic)
        run_hist["steps"] = self.steps
        run_hist["logs"] = logs
        run_hist["evolution"] = evolution
        run_hist["time"] = round(end_time - start_time, 1)
        self.historic.append(run_hist)

        # self.printRunHistoric(run_hist)

    def printRunHistoric(self, run_hist):
        signals = {}
        for q in run_hist["evolution"].keys():
            signals[q] = run_hist["evolution"][q]

        print(f"Run {run_hist['id']}: {run_hist['title']}")
        print(f"Steps: {run_hist['steps']}, Time: {run_hist['time']}s")

        for q, values in signals.items():
            plt.plot(values, label=q)
        plt.legend()
        plt.title(run_hist["title"])
        plt.show()

    def printHistoric(self):
        for run in self.historic:
            self.printRunHistoric(run)

    def movingAverage(self, data, window_size):
        if window_size <= 0:
            raise ValueError("Window size must be positive")
        if not data:
            return []
        cumsum = [0] * (len(data) + 1)
        for i, x in enumerate(data):
            cumsum[i + 1] = cumsum[i] + x
        return [(cumsum[i + window_size] - cumsum[i]) / window_size for i in range(len(data) - window_size + 1)]

    def step(self):
        all_signals = []
        all_states = []
        all_states_dict = []
        all_actions = []
        all_next_states = []
        all_next_states_dict = []
        all_flags = []
        all_gflags = []

        # sequential
        for agent in self.agents:
            state = self.getState()
            if self.debug:
                agent.printQFunctions(state)  # print Q-Functions for debugging
            state_dict = self.getStateDict()
            all_states.append(state)
            all_states_dict.append(state_dict)
            
            action = agent.getAction(state)
            agent.setLastAction(action)
            all_actions.append(action)

            signals, flags, gflags = self.doAction(agent, action)
            all_signals.append(signals)
            # print(flags, gflags)

            all_flags.append(flags)
            all_gflags.append(gflags)

            next_state = self.getState()
            next_state_dict = self.getStateDict()
            all_next_states.append(next_state)
            all_next_states_dict.append(next_state_dict)

        # append global flags to all agents' flags
        for gflags in all_gflags:
            for flag in gflags:
                for i in range(len(all_flags)):
                    if flag not in all_flags[i]:
                        all_flags[i].append(flag)

        for i, agent in enumerate(self.agents):
            state = all_states[i]
            state_dict = all_states_dict[i]
            next_state = all_next_states[i - 1]
            next_state_dict = all_next_states_dict[i - 1]
            if self.debug_judgement:
                print("State:",state)
                print("Q-Functions:", agent.printQFunctions(state))
            all_signals[i]['V'] = agent.judge(next_state_dict, all_flags[i], self.debug_judgement)  # judges the consequences
            optimalAction = agent.selectBestAction(next_state)
            if optimalAction is not None:
                if agent.isOptimal:
                    optimalAction = optimalAction[0]
                else:
                    optimalAction = rd.choice(optimalAction)
            agent.updateQFunctions(state, all_actions[i], all_signals[i], next_state, optimalAction)
            agent.setLastSignal(all_signals[i])
        
        return all_signals[-1]
    
    def getStateDict(self):
        state = {}

        state["grid"] = [[cell.type for cell in row] for row in self.grid]
        state["pos"] = {agent.name: self.pos[agent.name] for agent in self.agents}
        state["objects"] = {name: obj for name, obj in self.objects.items()}
        state["inventory"] = {agent.name: agent.getInventory() for agent in self.agents}
        state["iterations"] = self.iterations
        state["actions"] = {agent.name: agent.getLastAction() for agent in self.agents}

        return state

    def getState(self):
        # state of the map
        grid_state = ""
        for row in self.grid:
            grid_state += "".join([str(cell.type) for cell in row])
        grid_state = hash(grid_state)

        # agent positions
        agent_pos_state = tuple([pos[0] + pos[1] * self.width for pos in self.pos.values()])

        # inventory
        agent_inventory = tuple(sorted((agent.name, tuple(agent.getInventory())) for agent in self.agents))

        # objects
        objects_state = tuple(sorted((name, obj["pos"][0] + obj["pos"][1] * self.width)
                                     for name, obj in self.objects.items()))
        
        # iteration
        iteration_state = self.iterations // 5#10

        # final state
        state = [grid_state, agent_pos_state, objects_state, agent_inventory, iteration_state]
        return tuple(state)
    
    def setOptimal(self, value):
        for agent in self.agents:
            agent.setOptimal(value)
    
    def getCondition(self, condition):
        # if starts with 'not-', it is a negation
        negation = False
        if condition.startswith("not-"):
            condition = condition[4:]
            negation = True
        return condition, negation

    def processObject(self, obj_name, obj, agent):
        reward = 0
        flags = []
        global_flags = []
        inventory = agent.getInventory()
        toRemove = []

        for item in obj["condition"]:
            str_item, negation = self.getCondition(item)
            if (negation and str_item in inventory) or (not negation and str_item not in inventory):
                return 0, [], [], []
        
        agent.addItemsToInventory(obj["inv_add"])
        agent.removeItemsFromInventory(obj["inv_rem"])
        reward += obj["reward"]
        flags.extend(obj["flags"])
        global_flags.extend(obj["global_flags"])
        if not obj["permanent"]:
            toRemove.append(obj_name)

        return reward, flags, global_flags, toRemove
    
    def handleObjectsOnPosition(self, agent):
        reward = 0
        flags = []
        global_flags = []
        toRemove = []
        pos = self.pos[agent.name]

        for obj_name, obj in self.objects.items():
            if obj["pos"] == pos:
                reward_p, flags_p, global_flags_p, toRemove_p = self.processObject(obj_name, obj, agent)
                reward += reward_p
                flags.extend(flags_p)
                global_flags.extend(global_flags_p)
                toRemove.extend(toRemove_p)
        for obj_name in toRemove:
            del self.objects[obj_name]

        return reward, flags, global_flags

    def doAction_1(self, agent, action):
        signals = {}
        pos = self.pos[agent.name]
        reward = -1
        if rd.random() < self.stochasticity:
            possible = ["up", "down", "left", "right"]
            possible.remove(action)
            action = rd.choice(possible)
        if action == "up":
            if pos[1] > 0 and self.grid[pos[1] - 1][pos[0]].type != WALL:
                pos[1] -= 1
                reward = 0
        elif action == "down":
            if pos[1] < self.height - 1 and self.grid[pos[1] + 1][pos[0]].type != WALL:
                pos[1] += 1
                reward = 0
        elif action == "left":
            if pos[0] > 0 and self.grid[pos[1]][pos[0] - 1].type != WALL:
                pos[0] -= 1
                reward = 0
        elif action == "right":
            if pos[0] < self.width - 1 and self.grid[pos[1]][pos[0] + 1].type != WALL:
                pos[0] += 1
                reward = 0
        self.pos[agent.name] = pos

        reward_handle, flags, global_flags = self.handleObjectsOnPosition(agent)
        reward += reward_handle

        signals = {"R": reward, "V": 0}
        return signals, flags, global_flags
    
    def doAction_2(self, agent, action):
        signals = {}
        pos = self.pos[agent.name]
        reward = 0
        movement = action[0]
        speed = action[1]

        if speed == "fast":
            reward -= 0.5
        elif speed == "slow":
            reward -= 1
        
        if movement == "up":
            if pos[1] > 0 and self.grid[pos[1] - 1][pos[0]].type != WALL:
                pos[1] -= 1
            else:
                reward -= 10
        elif movement == "down":
            if pos[1] < self.height - 1 and self.grid[pos[1] + 1][pos[0]].type != WALL:
                pos[1] += 1
            else:
                reward -= 10
        elif movement == "left":
            if pos[0] > 0 and self.grid[pos[1]][pos[0] - 1].type != WALL:
                pos[0] -= 1
            else:
                reward -= 10
        elif movement == "right":
            if pos[0] < self.width - 1 and self.grid[pos[1]][pos[0] + 1].type != WALL:
                pos[0] += 1
            else:
                reward -= 10

        self.pos[agent.name] = pos

        reward_handle, flags, global_flags = self.handleObjectsOnPosition(agent)
        reward += reward_handle
        flags.append("road" if self.grid[pos[1]][pos[0]].type == ROAD else "pavement")

        signals = {"R": reward, "V": 0}
        return signals, flags, global_flags
