from __future__ import print_function
 
import pinocchio
 
model = pinocchio.buildSampleModelManipulator()
data = model.createData()
 
q = pinocchio.neutral(model)
v = pinocchio.utils.zero(model.nv)
a = pinocchio.utils.zero(model.nv)
 
tau = pinocchio.rnea(model,data,q,v,a)
print('tau = ', tau.T)

 
import pinocchio as pin
from pinocchio.robot_wrapper import RobotWrapper
from pinocchio.visualize import (GepettoVisualizer, MeshcatVisualizer)
from sys import argv
import os
from os.path import dirname, join, abspath
 
# If you want to visualize the robot in this example,
# you can choose which visualizer to employ
# by specifying an option from the command line:
# GepettoVisualizer: -g
# MeshcatVisualizer: -m
VISUALIZER = None
if len(argv)>1:
    opt = argv[1]
    if opt == '-g':
        VISUALIZER = GepettoVisualizer
    elif opt == '-m':
        VISUALIZER = MeshcatVisualizer
    else:
        raise ValueError("Unrecognized option: " + opt)
 