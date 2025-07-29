from environment import *


if __name__ == "__main__":

    env = Environment()
    # env.loadFile("src/environments/basic_5x5.txt")
    env.loadPreset("pacman")
    env.display()

    env.run(display=False, run_title="Training")

    env.setSteps(200)
    env.run(display=True, run_title="Testing")

    env.printHistoric()