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

        self.historic = []

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
        pass

    def loadPacman(self, reset_agent=True):
        
        if reset_agent:
            self.steps = 10000
            self.timeout = 20
            self.loadFile("src/environments/basic_5x5.txt")
            pacman = Pinocchio("Pacman")
            pacman.loadOptimalAgent(self.steps)
            self.agents.append(pacman)

        actions = ["up", "down", "left", "right"]
        for agent in self.agents:
            agent.setActions(actions)
            # agent.isRandom = True  # comment this
            self.setPos(agent, [1, 1])  # default position

    def loadAdam(self, reset_agent=True):

        if reset_agent:
            self.steps = 30000
            self.timeout = 10
            self.loadFile("src/environments/apple_7x7.txt")
            adam = Pinocchio("Adam")
            adam.loadOptimalAgent(self.steps)

            # regulative norms
            n1 = RegulativeNorm()
            n1.type = "F"
            n1.premise = ["knowledge"]
            adam.addNorm(n1)

            # constitutive norms
            c1 = ConstitutiveNorm()
            c1.premise = ["eat"]
            c1.conclusion = ["knowledge"]

            sh = []  # stakeholders

            god = Stakeholder("God")
            god.addNorm(n1)
            god.addNorm(c1)
            god.afs[str(n1)] = AF()
            god.setArguments(str(n1))
            sh.append(god)

            for s in sh:
                adam.addStakeholder(s)
            self.agents.append(adam)

        self.objects["apple"] = {}
        self.objects["apple"]["pos"] = [3, 3]
        self.objects["apple"]["symbol"] = "A"
        self.objects["apple"]["flags"] = ["eat"]
        self.objects["apple"]["reward"] = 10

        actions = ["up", "down", "left", "right"]
        for agent in self.agents:
            agent.setActions(actions)
            # agent.isRandom = True  # comment this
            self.setPos(agent, [2, 2])

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

        count = 0
        reset = True
        i = 0
        while i < self.steps or not reset:
            i += 1
            reset = False
            if display and count == 0:
                print("=========VVVVV=========VVVVV=========")
            log = self.step()
            logs.append(log)
            if display:
                print(f"Step {i + 1}/{self.steps}")
                self.display()
            count += 1
            if count >= self.timeout:  # reset the agent every X steps
                self.loadPreset(self.loadedPreset, reset_agent=False)
                count = 0
                reset = True

        end_time = time.time()

        # movingAverage of the tracked Q-Functions
        evolution = {}
        qfunctions = ['R']
        window = 100
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
        for agent in self.agents:
            state = self.getState()
            action = agent.getAction(state)
            signals = self.doAction(agent, action)
            all_signals.append(signals)
            next_state = self.getState()
            agent.updateQFunctions(state, action, signals, next_state)
        
        return all_signals[-1]

    def getState(self):
        # state of the map
        grid_state = ""
        for row in self.grid:
            grid_state += "".join([str(cell.type) for cell in row])
        grid_state = hash(grid_state)

        # agent positions
        agent_pos_state = tuple([pos[0] + pos[1] * self.width for pos in self.pos.values()])

        # objects
        objects_state = tuple(sorted((name, obj["pos"][0] + obj["pos"][1] * self.width)
                                     for name, obj in self.objects.items()))

        # final state
        state = [grid_state, agent_pos_state, objects_state]
        return tuple(state)

    def doAction(self, agent, action):
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

        flags = []
        toRemove = []
        for obj_name, obj in self.objects.items():
            if obj["pos"] == pos:
                reward += obj["reward"]
                flags.extend(obj["flags"])
                toRemove.append(obj_name)
        for obj_name in toRemove:
            del self.objects[obj_name]

        signals = {"R": reward}
        return signals
