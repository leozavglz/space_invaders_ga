from enum import IntEnum

class Cell(IntEnum):
    EMPTY = 0
    DEFENDER = 1
    INVADER = 2
    MISSILE = 3
    BOMB = 4

class Action(IntEnum):
    STAY = 0
    LEFT = 1
    RIGHT = 2
    SHOOT = 3

ACTIONS = [Action.STAY, Action.LEFT, Action.RIGHT, Action.SHOOT]
