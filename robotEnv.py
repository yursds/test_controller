import gymnasium as gym
from gym.spaces import Box
import pinocchio as pin
from example_robot_data import load
import numpy as np

from pinocchio.robot_wrapper import RobotWrapper as robWrap

from pinocchio.visualize import MeshcatVisualizer

import time
import random
# External parameters
LOW_OBS = [0,0,0]
HIGH_OBS = [0,0,0]
LOW_ACT = [0,0,0]
HIGH_ACT = [0,0,0]
# get robot using example_robot_data

ROB_STR = 'double_pendulum'



class PinocchioEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, rob_str):
        super(PinocchioEnv, self).__init__()
        
        self.robot:robWrap = load(rob_str)
        # get joint position ranges
        q_max = self.robot.model.upperPositionLimit.T
        q_min = self.robot.model.lowerPositionLimit.T
        # get max velocity
        dq_max = self.robot.model.velocityLimit
        dq_min = -dq_max
        self.q = (q_min+q_max)/2
        self.dq = (dq_min+dq_max)/2
        self.length = 0
        self.viz = MeshcatVisualizer(self.robot.model, self.robot.collision_model, self.robot.visual_model)
        self.viz.initViewer(open=True)
        # action
        # kinematic controls
        try:
            self.viz.loadViewerModel("pinocchio")
        except AttributeError as err:
            print("Error while loading the viewer model. It seems you should start gepetto-viewer")
            print(err)
        
        self.viz.display(self.q)
        # Define action and observation space (gym.spaces)
        self.action_space = Box(low=np.array([q_min, dq_min]), high=np.array([q_max, dq_max]))
        self.observation_space = Box(low=np.array([q_min, dq_min]), high=np.array([q_max, dq_max]))
        self.state = self.q
        self.reward = 0
        # Add other attributes
        
    def step(self, action):
        
        done = False
        self.dq = np.array([0.1,0.1])
        self.q[0] = action[0]
        self.q[1] = action[1]
        self.length += 1
        # visualise the robot
        self.viz.display(self.q)
        time.sleep(0.1)
        if self.q[0] == np.pi/3:
            self.reward = 10
        if self.q[1] == np.pi/3:
            self.reward = 10
        if self.length >= 50:
            done = True
             
        return self.state, self.reward, done, {}
    
    def reset(self, seed=None, options=None):
        
        super().reset(seed=seed)
        self.length = 0
        self.state = self.q
        observation = np.array([self.q, self.dq])
        return observation, {}
    
    def render(self, mode='human'):
        pass
    def close(self):
        pass
    
    
    
    
env = PinocchioEnv(ROB_STR)
env.action_space.sample()

episodes = 10
for episode in range(episodes+1):
    state = env.reset()
    done = False
    score = 0
    while not done:
        action = np.array([random.choice([0,1]),random.choice([0,1])])
        n_state, reward, done, _ = env.step(action)
        score += reward
    print (score)