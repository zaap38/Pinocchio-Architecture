from environment import *


if __name__ == "__main__":
    
    print("START")

    env = Environment()
    env.loadFile("src/environments/basic_5x5.txt")
    env.display()

    print("END")