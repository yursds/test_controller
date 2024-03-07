import pinocchio as pin
from pinocchio.robot_wrapper import RobotWrapper as robWrap
from example_robot_data import load

# get robot using example_robot_data
robot:robWrap = load('double_pendulum')


class MyPinWrap():
    def __init__(self):
        1