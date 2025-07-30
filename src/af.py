

UNDEC = 0
IN = 1
OUT = 2


class AF:

    def __init__(self):
        self.arguments = []  # List of arguments
        self.attacks = []
        self.attacked_by = {}
        self.attacking = {}
        
    def addArgument(self, argument):
        if argument not in self.arguments:
            self.arguments.append(argument)

    def addAttack(self, attacker, attacked=None):
        if attacked is None:
            attacked = attacker[1]
            attacker = attacker[0]

        if (attacker, attacked) in self.attacks:
            raise ValueError(f"Attack from '{attacker}' to '{attacked}' already exists.")
        
        self.attacks.append((attacker, attacked))

        # update the attacked_by and attacking dicts
        if attacker not in self.attacked_by:
            self.attacked_by[attacker] = []
        self.attacked_by[attacker].append(attacked)

        if attacked not in self.attacking:
            self.attacking[attacked] = []
        self.attacking[attacked].append(attacker)

    def getAttacks(self):
        return self.attacks
    
    def getInAttack(self, arg):
        # arguments attacking arg
        if arg in self.attacking:
            return self.attacking[arg]
        return []
    
    def getOutAttack(self, arg):
        # arguments attacked by arg
        if arg in self.attacked_by:
            return self.attacked_by[arg]
        return []
    
    def computeExtension(self, extension):
        ext = []
        if extension == "grounded":
            # Compute grounded extension
            ext = self.groundedExtension()
        elif extension == "preferred":
            # Compute preferred extension
            ext = self.preferredExtension()
        return ext
    
    def getRootArguments(self, status):
        # Get arguments that are not attacked by any other argument which is IN or UNDEC
        root_args = []
        for arg in self.arguments:
            if status[arg] != UNDEC:
                continue
            attacked_by = self.getInAttack(arg)
            if all(status[attacker] == OUT for attacker in attacked_by):
                root_args.append(arg)
        return root_args

    def groundedExtension(self):
        # Placeholder for grounded extension computation
        status = {}
        for arg in self.arguments:
            status[arg] = UNDEC

        roots = self.getRootArguments(status)
        while len(roots) > 0:
            for arg in roots:
                status[arg] = IN
                for attacked in self.getOutAttack(arg):
                    status[attacked] = OUT
            roots = self.getRootArguments(status)

        return [arg for arg, stat in status.items() if stat == IN]
    
    def preferredExtension(self):
        # Placeholder for preferred extension computation
        return []
    
    def getStatus(self, value):
        if value == IN:
            return "IN"
        elif value == OUT:
            return "OUT"
        elif value == UNDEC:
            return "UNDEC"
        else:
            raise ValueError(f"Unknown status value: {value}")
    
    def printArgs(self, status=None):
        if status is None:
            print("Arguments:", self.arguments)
        else:
            print([(arg, self.getStatus(status[arg])) for arg in self.arguments])

    def print(self):
        print("Arguments:", self.arguments)
        print("Attacks:", self.attacks)