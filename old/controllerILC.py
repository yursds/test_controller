import torch
from tensordict     import TensorDict



SCALE_Q = 1
SCALE_L = 0.5
    
# ============================================================== #
# TO DO
# define a Q and L to satisfy error convergence to zero.
# ============================================================== #

class ctrlILC():
    """ 
    This class implements a model free ILC action ONLY for square MIMO.
    """
        
    
    def __init__(self, dimU:int, dimE:int, dimSamples:int, Q:torch.Tensor=None, L:torch.Tensor=None) -> None:
        """ 
        Args:
            dimU (int): dimension of input vector (single step).
            dimE (int): dimention of error vector (single step).
            dimSamples (int[optional]): number of samples in a single episode.
            Q (torch.Tensor[optional]): Q-filter matrix (square matrix).
            L (torch.Tensor[optional]): learning gain matrix (square matrix).
        """
        
        # Check        
        if dimU != dimE:
            raise TypeError("ILC class implemented only for square MIMO.")      
        
        self.uEp    = torch.zeros(dimU,dimSamples).type(torch.float64) # control inputs for current episode
        self.uk     = torch.zeros(dimU,1).type(torch.float64)          # current control input
        self.ek     = torch.zeros(dimE,1).type(torch.float64)          # current error
        self.idx    = 0                                                # idx step
        self.mem    = []                                                # list of episode's memory
        
        # Init optionals variables # ADD Q depth check or expand it
        if Q is not None:
            if not isinstance(Q, torch.Tensor):
                raise TypeError("Q must be a torch.Tensor")
            rows, cols = Q.size()
            if rows != cols:
                raise ValueError("Q must be a square matrix")
            self.Q:torch.Tensor = Q.type(torch.float64)
        else:
            #self.Q = (torch.tril(torch.ones(dimSamples,dimSamples))*SCALE_Q).expand(dimU, -1, -1)
            self.Q = (torch.eye(dimSamples)*1).expand(dimU, -1, -1).type(torch.float64)
        if L is not None:
            if not isinstance(L, torch.Tensor):
                    raise TypeError("L must be a torch.Tensor")
            rows, cols = L.size()
            if rows != cols:
                raise ValueError("L must be a square matrix")
            self.L:torch.Tensor = L.type(torch.float64)
        else:
            #self.L = (torch.tril(torch.ones(dimSamples,dimSamples))*SCALE_L).expand(dimE, -1, -1).type(torch.float64)
            self.L = (torch.eye(dimSamples)*0.51).expand(dimU, -1, -1).type(torch.float64)
        
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
        tmp_mem = torch.cat([tmp_mem.clone(),data.type(torch.float64)],dim=1)      # stack in column
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
           
    def updateQ(self, newQ:torch.Tensor) -> None:
        """ Update Q tensor """
        self.Q = newQ(torch.float64)
    
    def updateL(self, newL:torch.Tensor) -> None:
        """ Update L tensor """
        self.L = newL.type(torch.float64)

    def stepILC(self) -> None:
        """
        Start a new episode of ILC.
        Update control for to use in this episode.
        """
        
        if len(self.mem) == 0:
            raise("ILC first episode is not initialized")
        
        self.newEp()
        
        Q   = self.Q
        L   = self.L
        
        u_old:torch.Tensor  = self.mem[-2]["input"]
        e_old:torch.Tensor  = self.mem[-2]["error"]
        
        #aaa = torch.einsum('dij,dj->di',L,e_old)
        #a = torch.mm(L[0,:,:].squeeze(),e_old[0:1,:].t())
        #print(aaa)
        #print(a.t())
        u_new = torch.einsum('dij,dj->di',Q,(u_old+torch.einsum('dij,dj->di',L,e_old)))
        
        self.uEp = u_new
                          
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
        Init step 0 of ILC: no input and save  reference as error. 
        Reference is stacked in rows.
        """
        self.updateMemError(ref)
        self.updateMemInput(self.uk)
         

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
    conILC = ctrlILC(dimU=dimU,dimE=dimE,dimSamples=samples)     
   
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
        last_err.append(err[:, -5])
    
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