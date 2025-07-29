from environment import *


if __name__ == "__main__":

    env = Environment()
    # env.loadFile("src/environments/basic_5x5.txt")

    preset = "adam"

    env.loadPreset(preset, reset_agent=True)

    env.run(display=False, run_title="Training")

    env.setSteps(200)
    env.loadPreset(preset, reset_agent=False)
    env.run(display=True, run_title="Testing")

    env.printHistoric()