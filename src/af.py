


class AF:

    def __init__(self):
        self.arguments = []  # List of arguments
        self.attacks = {}  # Dictionary of attacks, key: attacker, value: list of attacked arguments

    def addArgument(self, argument):
        self.arguments.append(argument)

    def addAttack(self, attacker, attacked):
        if attacker not in self.attacks:
            self.attacks[attacker] = []
        self.attacks[attacker].append(attacked)

    def getAttacks(self):
        return self.attacks