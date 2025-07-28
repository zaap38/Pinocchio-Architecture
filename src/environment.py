from pinocchio import *


WALL = 0
ROAD = 1
PLAIN = 2
PACMAN = 3


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


class Environment:

    def __init__(self):
        self.width = 0
        self.height = 0
        self.grid = []

        self.agents = []
        self.pos = {}  # position of each agent

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

    def loadPreset(self, presetName):
        if presetName == "taxi":
            self.loadTaxi()
        elif presetName == "pacman":
            self.loadPacman()

    def loadTaxi(self):
        pass

    def loadPacman(self):
        actions = ["up", "down", "left", "right"]
        self.loadFile("src/environments/basic_5x5.txt")
        self.addAgent(Pinocchio("Pacman"))
        for agent in self.agents:
            agent.setActions(actions)
            agent.isRandom = True  # comment this
            agent.pos = [1, 1]

    def display(self):
        cp_grid = cp.deepcopy(self.grid)
        for agent in self.agents:
            pos = self.pos[agent.name]
            cp_grid[pos[1]][pos[0]].setType(PACMAN)
        for row in cp_grid:
            print("".join(SYMBOLS[cell.type] for cell in row))

    def run(self, display=False):
        for i in range(100):
            self.step()
            if display:
                self.display()

    def step(self):
        for agent in self.agents:
            state = self.getState()
            action = agent.getAction(state)
            print(action)
            signals = self.doAction(agent, action)
            next_state = self.getState()
            agent.updateQFunctions(state, action, signals, next_state)

    def getState(self):
        state = []
        for row in self.grid:
            state.append([cell.type for cell in row])
        return state
    
    def doAction(self, agent, action):
        signals = {}
        pos = self.pos[agent.name]
        reward = -1
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
        signals[agent.name] = {"R": reward}
