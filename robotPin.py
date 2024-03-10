import torch
import pinocchio as pin
from pinocchio.robot_wrapper import RobotWrapper as robWrap

from example_robot_data import load

# get robot using example_robot_data
robot:robWrap = load('double_pendulum')

# get joint position ranges
q_max = robot.model.upperPositionLimit.T
q_min = robot.model.lowerPositionLimit.T
# get max velocity
dq_max = robot.model.velocityLimit
dq_min = -dq_max

# Use robot configuration.
q = (q_min+q_max)/2

while True==False:
    # some sinusoidal motion
    for i in torch.sin(torch.linspace(-torch.pi,torch.pi,200)):

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


