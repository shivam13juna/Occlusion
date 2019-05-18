from enum import Enum


# type of operation
class Toperation(Enum):
    classification = 1
    occlusion = 2
    metrics = 3
    oldmetrics = 4
    demo = 5
