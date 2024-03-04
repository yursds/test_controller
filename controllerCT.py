import torch

import torch
from tensordict     import TensorDict

class controllerCT():
    """ This class implements a Computed Torque. """

    def __init__(self, robot=None, Kp=None, Kv=None):
        """ Args:
                dim_u (int): dimension of input vector (single step).
                dim_e (int): dimention of error vector (single step).
                Q (torch.Tensor[optional]): Q-filter matrix (square matrix)
                L (torch.Tensor[optional]): learning gain matrix (square matrix) """

        # check
            
        self.M:torch.Tensor
        self.C:torch.Tensor
        self.G:torch.Tensor
        self.Kp:torch.Tensor
        self.Kv:torch.Tensor
        
    def getControl(self) -> torch.Tensor:
        return self.u
 
    def step(self) -> None:
        """
        Single step of CT.
        Update control.
        """
        tau_cmd = self.M * ddot_q_d + self.C + self.Kp * error + self.Kv * dot_error;  #// C->C*dq
        
        self.idx += 1 
                               
       
if __name__ == '__main__':
    
    dimU = 5
    dimE = dimU
    samples = 10
    episodes = 50
    
    # definition of reference
    ref = torch.tensor([[i]*dimE for i in range(samples)]).t()
    for h in range(ref.size()[0]):
        ref[h,:] = ref[h,:]*(h+1)
    
    # ILC controller instance
    conILC = controllerCT(dimU=dimU,dimE=dimE,dimSamples=samples)     
   
    # initialization of ILC memory
    conILC.newEp()                              # start new episode
    for k in range(samples):
    
        # controller do nothing and save error as reference
        conILC.firstEpLazy(ref[:, k:k+1])
    
    mem = conILC.getMemory()                      # get memory

    # start ILC iteration   
    for _ in range(episodes):
        
        conILC.newEp()                          # start new episode
        conILC.stepILC()                        # update control
        
        for k in range(samples):
            
            new_u = conILC.getControl()         # get ILC control
            conILC.updateMemInput(new_u)        # save total u for next episode
            
            out = new_u/2                       # simulate robot (simple function)
            
            new_e = ref[:, k:k+1]-out           # get new error
            conILC.updateMemError(new_e)        # save new_error
            
            mem = conILC.getMemory()
            
    from matplotlib import pyplot as plt

    last_err = []
    for td in mem:
        err = td["error"]
        last_err.append(err[:, -1])
    
    plt.ion()

    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(last_err)
    plt.title("error")
    plt.xlabel("iteration")
    plt.grid()
    
    print('Complete')
    plt.ioff()
    plt.show()
    
    print("finish")