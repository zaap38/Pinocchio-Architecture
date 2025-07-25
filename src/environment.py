from pinocchio import *


WALL = 0
ROAD = 1
PLAIN = 2


SYMBOLS = {
    WALL: "#",
    ROAD: " ",
    PLAIN: "-"
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

    def addAgent(self, agent):
        self.agents.append(agent)

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

    def loadTaxi(self):
        pass

    def display(self):
        for row in self.grid:
            print("".join(SYMBOLS[cell.type] for cell in row))