# functions for the more complex facts


def parking_close(state, flags):
    if "parked" in flags:
        return False
    dist = 2
    dist_to_parking = 10000
    if "parking" in state["objects"]:
        dist_to_parking = abs(state["pos"]["Taxi"][0] - state["objects"]["parking"]["pos"][0]) + \
                          abs(state["pos"]["Taxi"][1] - state["objects"]["parking"]["pos"][1])
    return dist_to_parking < dist