## This class wrap data to construct an enviroment similiar to EnvBase of Gymnasium Library.

import torch


class RealEnv():
    
    def __init__(self):
        
        self.observation_space:torch.Tensor = 0
        self.action_space:torch.Tensor = 0