from dataclasses import dataclass

@dataclass
class Defender:
    r: int
    c: int

@dataclass
class Invader:
    r: int
    c: int
    dir: int  # +1 derecha, -1 izquierda

@dataclass
class Missile:
    r: int
    c: int
    active: bool = False

@dataclass
class Bomb:
    r: int
    c: int
    active: bool = False
