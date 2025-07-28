from pinocchio import *

import matplotlib.pyplot as plt


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

        self.steps = 1000

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

    def loadPreset(self, presetName):
        if presetName == "taxi":
            self.loadTaxi()
        elif presetName == "pacman":
            self.loadPacman()

    def loadTaxi(self):
        pass

    def loadPacman(self):
        self.steps = 1000
        actions = ["up", "down", "left", "right"]
        self.loadFile("src/environments/basic_5x5.txt")
        
        pacman = Pinocchio("Pacman")
        pacman.loadOptimalAgent(self.steps)
        self.agents.append(pacman)

        for agent in self.agents:
            agent.setActions(actions)
            # agent.isRandom = True  # comment this
            self.setPos(agent, [1, 1])  # default position

    def setPos(self, agent_name, pos):
        if type(agent_name) != str:
            agent_name = agent_name.name
        self.pos[agent_name] = pos

    def display(self):
        cp_grid = cp.deepcopy(self.grid)
        for agent in self.agents:
            pos = self.pos[agent.name]
            cp_grid[pos[1]][pos[0]].setType(PACMAN)
        for row in cp_grid:
            print("".join(SYMBOLS[cell.type] for cell in row))

    def run(self, display=False):

        logs = []

        for i in range(self.steps):
            log = self.step()
            logs.append(log)
            if display:
                self.display()
                
            # Compute average reward over the last 20 logs
            last_logs = logs[-20:] if len(logs) >= 20 else logs
            total_reward = 0
            for log in last_logs:
                if isinstance(log, dict) and "R" in log:
                    total_reward += log["R"]
            avg_reward = total_reward / len(last_logs) if last_logs else 0
            print(f"Average reward over last {len(last_logs)} steps: {avg_reward}")

        run_hist = {}
        run_hist["id"] = len(self.historic)
        run_hist["steps"] = self.steps
        run_hist["logs"] = logs
        self.historic.append(run_hist)

        self.print_historic(logs)

    def print_historic(self, logs):
        rewards = []
        for log in logs:
            if isinstance(log, dict) and "R" in log:
                rewards.append(log["R"])
            elif isinstance(log, list) and len(log) > 0 and isinstance(log[-1], dict) and "R" in log[-1]:
                rewards.append(log[-1]["R"])
            else:
                rewards.append(0)

        plt.plot(rewards)
        plt.xlabel("Step")
        plt.ylabel("Reward")
        plt.title("Reward Evolution")
        plt.show()

    def step(self):
        all_signals = []
        for agent in self.agents:
            state = self.getStateHash()
            action = agent.getAction(state)
            print(agent.getQValues("R", state))
            signals = self.doAction(agent, action)
            all_signals.append(signals)
            next_state = self.getStateHash()
            agent.updateQFunctions(state, action, signals, next_state)
        
        return all_signals[-1]

    def getState(self):
        state = []
        for row in self.grid:
            state.append([cell.type for cell in row])
        return state
    
    def getStateHash(self):
        state = self.getState()
        return tuple(tuple(row) for row in state)
    
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
        signals = {"R": reward}
        return signals
