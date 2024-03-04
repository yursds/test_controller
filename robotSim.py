from __future__ import print_function
 
import pinocchio as pin
from pinocchio.robot_wrapper import RobotWrapper as robWrap
import numpy as np

from example_robot_data import load
from pinocchio.visualize import MeshcatVisualizer

# get panda robot usinf example_robot_data
robot:robWrap = load('double_pendulum')

# get joint position ranges
q_max = robot.model.upperPositionLimit.T
q_min = robot.model.lowerPositionLimit.T
# get max velocity
dq_max = robot.model.velocityLimit
dq_min = -dq_max

# Use robot configuration.
q = (q_min+q_max)/2

viz = MeshcatVisualizer(robot.model, robot.collision_model, robot.visual_model)

# Start a new MeshCat server and client.
viz.initViewer(open=True)
#viz.initViewer()

# Load the robot in the viewer.
try:
    viz.loadViewerModel("pinocchio")
except AttributeError as err:
    print("Error while loading the viewer model. It seems you should start gepetto-viewer")
    print(err)
    
viz.display(q)

while True:
    # some sinusoidal motion
    for i in np.sin(np.linspace(-np.pi,np.pi,200)):

        # update the joint position
        q[0] = i
        q[1] = i
        #q[2] = i

        # calculate the jacobian
        data = robot.model.createData()
        pin.framesForwardKinematics(robot.model,data,q)
        pin.computeJointJacobians(robot.model,data, q)
        J = pin.getFrameJacobian(robot.model, data, robot.model.getFrameId(robot.model.frames[-1].name), pin.LOCAL_WORLD_ALIGNED)
        # use only position jacobian
        J = J[:3,:]

        # end-effector pose
        Xee = data.oMf[robot.model.getFrameId(robot.model.frames[-1].name)]

        # calculate the polytope
        opt = {'calculate_faces':True}

        # visualise the robot
        viz.display(q)

        