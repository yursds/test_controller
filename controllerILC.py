import torch
from tensordict     import TensorDict

# ============================================================== #
# TO DO
# define a Q and L to satisfy error convergence to zero.
# ============================================================== #

class ctrlILC():
    """ 
    This class implements a model free ILC action ONLY for square MIMO.
    """
        
    
    def __init__(self, dimMIMO:int, dimSamples:int, Q:float=1.0, Le:float=0.5, Ledot:float=0.5, dtype=torch.float64) -> None:
        """ 
        Args:
            dimMIMO (int): dimension of input vector (single step).
            dimSamples (int[optional]): number of samples in a single episode.
            Q (torch.Tensor[optional]): Q-filter matrix (square matrix).
            L (torch.Tensor[optional]): learning gain matrix (square matrix).
        """
        
        self.dtype      = dtype
        self.Q          = Q
        self.Le         = Le
        self.Lde        = Ledot
        self.dimMIMO    = dimMIMO
        
        self.uEp        = torch.zeros(dimMIMO,dimSamples).type(self.dtype)  # control inputs for current episode
        self.uk         = torch.zeros(dimMIMO,1).type(self.dtype)              # current control input
        self.ek         = torch.zeros(dimMIMO,1).type(self.dtype)              # current error
        self.idx        = 0                                                 # idx step
        self.mem        = []                                                # list of episode's memory
        
        # data memory template (of error and input) stacked in column
        self.__tmplMem = TensorDict(
            {
            'error': torch.Tensor(), 
            'input': torch.Tensor()
            },
            batch_size=[]
        )
        
    def __updateMem__(self, data:torch.Tensor, dict:str) -> None:
        """
        Store new data in a tensordict in a list. Stacked in column.
        """
        if not isinstance(data, torch.Tensor):
            raise TypeError("data must be a torch.Tensor")
        if data.size()[1] != 1:
            raise ("data must be a column vector")
        
        # use newest tensordict
        tmp_mem:torch.Tensor = self.mem[-1][dict]
        tmp_mem = torch.cat([tmp_mem.clone(), data.type(self.dtype)],dim=1)      # stack in column
        
        # update mem
        self.mem[-1][dict] = tmp_mem
           
    def updateMemError(self, e_new:torch.Tensor) -> None:
        """
        Store new error in a tensordict in a list.
        """
        dict:str='error'
        self.__updateMem__(e_new, dict)
    
    def updateMemInput(self, u_new:torch.Tensor) -> None:
        """
        Store new input in a tensordict in a list.
        """
        
        dict:str='input'
        self.__updateMem__(u_new, dict)
   
    def newEp(self) -> None:
        """ 
        Create new tensordict to store new data of a new episode.
        Reset step index. 
        """
        self.mem.append(self.__tmplMem.clone())
        self.idx = 0
           
    def updateLe(self, newLe:torch.Tensor) -> None:
        """ Update Q tensor """
        self.Le = newLe.type(self.dtype)
    
    def updateLedot(self, newLedot:torch.Tensor) -> None:
        """ Update L tensor """
        self.Lde = newLedot.type(self.dtype)

    def stepILC(self) -> None:
        """
        Update control to use in this episode.
        Start new episode.
        """
        
        if len(self.mem) == 0:
            raise("ILC first episode is not initialized")
        
        Q   = self.Q
        Le  = self.Le
        Lde = self.Lde
        
        u_old:torch.Tensor  = self.mem[-1]["input"]
        e_old:torch.Tensor  = self.mem[-1]["error"]
        tmp0 = torch.tensor([[0.0]]).expand(self.dimMIMO,-1).type(self.dtype)
        de_old = torch.cat([tmp0,torch.diff(e_old, dim=1)],dim=1)
        #aaa = torch.einsum('dij,dj->di',L,e_old)
        #a = torch.mm(L[0,:,:].squeeze(),e_old[0:1,:].t())
        #print(aaa)
        #print(a.t())
        
        u_new = Q*u_old+Le*e_old+Lde*de_old
        
        self.uEp = u_new
        self.newEp() 
                          
    def resetAll(self) -> None:
        """
        Reset flag, index and memory
        """
        self.idx = 0
        self.mem = []

    def getMemory(self) -> list:
        """ Return list of dict of error and input stacked in column. """
        return self.mem

    def getControl(self) -> torch.Tensor:
        """ Return input of current step. """
        k = self.idx
        self.uk = self.uEp[:,k:k+1]
        self.idx += 1
         
        return self.uk
                 
    def firstEpLazy(self, ref:torch.Tensor) -> None:
        """ 
        Init step 0 of ILC: no input and save reference as error. 
        Reference is stacked in rows.
        """
        self.updateMemError(ref)
        self.updateMemInput(self.uk)
         

if __name__ == '__main__':
    
    dim = 5
    samples = 10
    episodes = 50
    
    # definition of reference
    ref = torch.tensor([[i]*dim for i in range(samples)]).t()
    for h in range(ref.size()[0]):
        ref[h,:] = ref[h,:]*(h+1)
    
    # ILC controller instance
    conILC = ctrlILC(dimMIMO=dim,dimSamples=samples)     
   
    # initialization of ILC memory
    conILC.newEp()                              # start new episode
    for k in range(samples):
    
        # controller do nothing and save error as reference
        conILC.firstEpLazy(ref[:, k:k+1])
    
    mem = conILC.getMemory()                      # get memory

    # start ILC iteration   
    for _ in range(episodes):
        
        conILC.stepILC()                        # update control
    
        for k in range(samples):
            
            new_u = conILC.getControl()         # get ILC control
            conILC.updateMemInput(new_u)        # save input for next episode (consider all inputs)
            
            out = new_u/2-torch.sin(new_u)                       # simulate robot (simple function)
            
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
    plt.plot(last_err)
    plt.title("error")
    plt.xlabel("episode (step k=5)")
    plt.grid()
    
    print('Complete')
    plt.ioff()
    plt.show()
    
    print("finish")