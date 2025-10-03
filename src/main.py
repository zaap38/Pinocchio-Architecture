from environment import *
import random as rd


if __name__ == "__main__":

    rd.seed(42)

    env = Environment()

    # pacman
    # adam
    # mini_taxi
    # taxi
    # avoidant_taxi
    preset = "avoidant_taxi"

    env.loadPreset(preset, reset_agent=True)

    env.debug = 0
    env.debug_judgement = 0
    env.run(display=0, run_title="Training")

    # env.debug = False
    # env.debug_judgement = False
    # env.setOptimal(True)
    # env.setSteps(2000)
    # env.loadPreset(preset, reset_agent=False)
    # env.run(display=False, run_title="Convergence")

    env.debug = 0
    env.debug_judgement = 0
    env.setOptimal(True)
    env.setLearning(False)
    env.setSteps(10)
    env.loadPreset(preset, reset_agent=False)
    env.run(display=True, run_title="Testing")

    # env.printHistoric()