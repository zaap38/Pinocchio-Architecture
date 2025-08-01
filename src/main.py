from environment import *
import random as rd


if __name__ == "__main__":

    rd.seed(42)

    env = Environment()

    # pacman
    # adam
    # mini_taxi
    # taxi
    preset = "taxi"

    env.loadPreset(preset, reset_agent=True)

    env.debug = False
    env.debug_judgement = False
    env.run(display=False, run_title="Training")

    env.debug = False
    env.debug_judgement = False
    env.setOptimal(True)
    env.setSteps(100000)
    env.loadPreset(preset, reset_agent=False)
    env.run(display=True, run_title="Testing")

    env.printHistoric()