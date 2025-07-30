from pinocchio import *

import time
import copy as cp
import random as rd

import matplotlib.pyplot as plt # type: ignore

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

    def loadTaxi(self, reset_agent=True):

        self.window = 10000
        
        if reset_agent:
            self.steps = 400000
            self.timeout = 50
            self.loadFile("src/environments/taxi_10x10.txt")
            taxi = Pinocchio("Taxi")
            taxi.loadOptimalAgent(self.steps)
            self.agents.append(taxi)

        movements = ["up", "down", "left", "right"]
        speeds = ["stop", "slow", "fast"]
        actions = []
        for m in movements:
            for s in speeds:
                actions.append((m, s))

        self.doAction = self.doAction_2  # change action method for taxi

        self.objects["parking"] = self.makeObject()
        self.objects["parking"]["pos"] = [7, 3]
        self.objects["parking"]["symbol"] = "P"
        self.objects["parking"]["flags"] = ["parked"]
        self.objects["parking"]["reward"] = -5
        self.objects["parking"]["inv_add"] = ["passenger"]

        self.objects["street"] = self.makeObject()
        self.objects["street"]["pos"] = [3, 3]
        self.objects["street"]["symbol"] = "S"
        self.objects["street"]["flags"] = ["parked"]
        self.objects["street"]["inv_add"] = ["passenger"]

        self.objects["destination"] = self.makeObject()
        self.objects["destination"]["pos"] = [8, 6]
        self.objects["destination"]["symbol"] = "D"
        self.objects["destination"]["reward"] = 100
        self.objects["destination"]["inv_rem"] = ["passenger"]
        self.objects["destination"]["condition"] = ["passenger"]

        for agent in self.agents:
            agent.resetInventory()
            agent.setActions(actions)
            # agent.isRandom = True  # comment this
            self.setPos(agent, [1, 1])

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
            r1 = RegulativeNorm()
            r1.type = "F"
            r1.premise = ["knowledge"]
            adam.addNorm(r1)

            # add fact functions
            adam.addFact("eat", lambda state, flags: "eat" in flags)
            adam.addFact("longtime", lambda state, flags: state["iterations"] > 5)

            # constitutive norms
            c1 = ConstitutiveNorm()
            c1.premise = ["eat"]
            c1.conclusion = ["knowledge"]

            c2 = ConstitutiveNorm()
            c2.premise = ["longtime"]
            c2.conclusion = ["hungry"]

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
        while i < self.steps or not reset:
            i += 1
            reset = False
            if display and self.iterations == 0:
                print("=========VVVVV=========VVVVV=========")
            log = self.step()
            logs.append(log)
            if display:
                print(f"Run '{run_title}': Iteration {i + 1}/{self.steps} - Step {self.iterations + 1}/{self.timeout}")
                for agent in self.agents:
                    print(f"{agent.name}: {agent.getLastAction()}   Inv: {agent.getInventory()}")
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
            state_dict = self.getStateDict()
            all_states.append(state)
            all_states_dict.append(state_dict)
            
            action = agent.getAction(state)
            agent.setLastAction(action)
            all_actions.append(action)

            signals, flags, gflags = self.doAction(agent, action)
            all_signals.append(signals)

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
            all_signals[i]['V'] = agent.judge(next_state_dict, all_flags[i])  # judges the consequences
            agent.updateQFunctions(state, all_actions[i], all_signals[i], next_state)
        
        return all_signals[-1]
    
    def getStateDict(self):
        state = {}

        state["grid"] = [[cell.type for cell in row] for row in self.grid]
        state["pos"] = {agent.name: self.pos[agent.name] for agent in self.agents}
        state["objects"] = {name: obj for name, obj in self.objects.items()}
        state["inventory"] = {agent.name: agent.getInventory() for agent in self.agents}
        state["iterations"] = self.iterations

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

        # final state
        state = [grid_state, agent_pos_state, objects_state, agent_inventory]
        return tuple(state)

    def processObject(self, obj_name, obj, agent):
        reward = 0
        flags = []
        global_flags = []
        inventory = agent.getInventory()
        toRemove = []

        for item in obj["condition"]:
            if item not in inventory:
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
        elif speed in ["slow", "stop"]:
            reward -= 1
        
        if speed != "stop":
            if movement == "up":
                if pos[1] > 0 and self.grid[pos[1] - 1][pos[0]].type != WALL:
                    pos[1] -= 1
                    reward = 0
            elif movement == "down":
                if pos[1] < self.height - 1 and self.grid[pos[1] + 1][pos[0]].type != WALL:
                    pos[1] += 1
                    reward = 0
            elif movement == "left":
                if pos[0] > 0 and self.grid[pos[1]][pos[0] - 1].type != WALL:
                    pos[0] -= 1
                    reward = 0
            elif movement == "right":
                if pos[0] < self.width - 1 and self.grid[pos[1]][pos[0] + 1].type != WALL:
                    pos[0] += 1
                    reward = 0
            else:
                reward -= 10  # hit a wall

        self.pos[agent.name] = pos

        reward_handle, flags, global_flags = self.handleObjectsOnPosition(agent)
        reward += reward_handle

        signals = {"R": reward, "V": 0}
        return signals, flags, global_flags
