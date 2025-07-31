from environment import *
import random as rd


if __name__ == "__main__":

    rd.seed(42)

    env = Environment()

    preset = "mini_taxi"#"taxi"

    env.loadPreset(preset, reset_agent=True)

    env.setDebug(False)
    env.run(display=False, run_title="Training")

    env.setSteps(200)
    env.loadPreset(preset, reset_agent=False)
    env.setDebug(False)
    env.run(display=True, run_title="Testing")

    env.printHistoric()