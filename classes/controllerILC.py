import torch
from tensordict     import TensorDict


class ILCctrl(object):
    """ 
    This class implements a model free ILC action ONLY for square MIMO.\n
    NOTE the control law used is:
        u_{j+1}(k) = u_{j}(k) + Le * e_{j}(k+1) + Ledot * de_{j}(k+1) + Leddot * dde_{j}(k+1)
    """
        
    
    def __init__(self, dimMIMO:int, dimSamples:int, 
                 Le:float=0.01, Ledot:float=0.0, Leddot:float=0.0, dtype=torch.float64) -> None:
        """
        This class implements a model free ILC action ONLY for square MIMO.\\
        NOTE the control law used is:\\
        u_{j+1}(k) = u_{j}(k) + Le * e_{j}(k+1) + Ledot * de_{j}(k+1) + Leddot * dde_{j}(k+1)
        
        Args:
            dimMIMO (int): dimension of input vector (single step).
            dimSamples (int): number of samples in a single episode.
            Le (float, optional): learning gain scalar for error. Defaults to 0.01.
            Ledot (float, optional): learning gain scalar for dot error. Defaults to 0.0.
            Leddot (float, optional): learning gain scalar for ddot error. Defaults to 0.0.
            dtype (_type_, optional): data type. Defaults to torch.float64.
        """
        
        self.dtype      = dtype
        self.Le         = Le
        self.Lde        = Ledot
        self.Ldde       = Leddot
        self.dimMIMO    = dimMIMO
        
        self.uEp        = torch.zeros(dimMIMO,dimSamples).type(self.dtype)  # control inputs for current episode
        self.uk         = torch.zeros(dimMIMO,1).type(self.dtype)           # current control input
        self.ek         = torch.zeros(dimMIMO,1).type(self.dtype)           # current error
        self.idx        = 0                                                 # idx step
        self.mem        = []                                                # list of episode's memory
        
        # data memory template (of error and input) stacked as column
        self.__tmplMem = TensorDict(
            {
            'error': torch.Tensor(),
            'dot_error': torch.Tensor(),
            'ddot_error': torch.Tensor(),
            'input': torch.Tensor()
            },
            batch_size=[]
        )
        
    def __updateMem__(self, data:torch.Tensor, dict:str) -> None:
        """
        Store new data in a tensordict in a list. Data stacked as column.

        Args:
            data (torch.Tensor): data as column vector
            dict (str): string to specify a field in the tensordict.
        """
        if not isinstance(data, torch.Tensor):
            raise TypeError("data must be a torch.Tensor")
        if data.size()[1] != 1:
            raise ("data must be a column vector")
        
        # use newest tensordict
        tmp_mem:torch.Tensor = self.mem[-1][dict]
        tmp_mem = torch.cat([tmp_mem.clone(), data.type(self.dtype)],dim=1)      # stack as column
        
        # update mem
        self.mem[-1][dict] = tmp_mem
           
    def updateMemError(self, e_new:torch.Tensor, de_new:torch.Tensor=None, dde_new:torch.Tensor=None) -> None:
        """
        Store new error, dot error, ddot error in a tensordict in a list, used in control law.
        
        Args:
            e_new (torch.Tensor): error as column vector.
            de_new (torch.Tensor, optional): dot error as column vector. If None is set to zeros.
            dde_new (torch.Tensor, optional): ddot error as column vector. If None is set to zeros.
        """
        
        if de_new is None:
            de_new = torch.zeros((self.dimMIMO,1),dtype=self.dtype)
            
        if dde_new is None:
            dde_new = torch.zeros((self.dimMIMO,1),dtype=self.dtype)
            
        dict0 = 'error'
        self.__updateMem__(e_new, dict0)
    
        dict1 = 'dot_error'
        self.__updateMem__(de_new, dict1)
    
        dict2 = 'ddot_error'
        self.__updateMem__(dde_new, dict2)
        
    def updateMemInput(self, u_new:torch.Tensor) -> None:
        """
        Store new input in a tensordict in a list, used in control law.
        
        Args:
            u_new (torch.Tensor): input as column vector.
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
           
    def updateLearningGain(self, newLe:float=None, newLde:float=None, newLdde:float=None) -> None:
        """
        Update learning gain for error, dot_error, ddot_error.

        Args:
            newLe (float, optional): learning gain for error. If None not change.
            newLde (float, optional): learning gain for dot error. If None not change.
            newLdde (float, optional): learning gain for ddot error. If None not change.
        """
        
        if newLe is not None:
            self.Le = torch.tensor(newLe).type(self.dtype)
        if newLde is not None:
            self.Lde = torch.tensor(newLde).type(self.dtype)
        if newLdde is not None:
            self.Ldde = torch.tensor(newLdde).type(self.dtype)
    
    def stepILC(self) -> None:
        """
        Update control to use in this episode.
        Start new episode.
        NOTE the control law used is:\\
        u_{j+1} = u_{j} + Le * e_{j} + Ledot * de_{j} + Leddot * dde_{j}
        """
        
        if len(self.mem) == 0:
            raise("ILC first episode is not initialized")
        
        Le      = self.Le
        Lde     = self.Lde
        Ldde    = self.Ldde
        
        u_old:torch.Tensor      = self.mem[-1]["input"]
        e_old:torch.Tensor      = self.mem[-1]["error"]
        
        u_new = u_old + Le*e_old
        
        if Lde != 0.0:    
            de_old:torch.Tensor     = self.mem[-1]["dot_error"]
            u_new = u_new + Lde*de_old
        
        if Ldde != 0.0:    
            dde_old:torch.Tensor    = self.mem[-1]["ddot_error"]
            u_new = u_new + Ldde*dde_old
        
        self.uEp = u_new
        self.newEp() 
                          
    def resetAll(self) -> None:
        """
        Reset flag, index and memory
        """
        self.idx = 0
        self.mem = []

    def getMemory(self) -> list:
        """ Return list of dict of error and input stacked as column. """
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
    l0 = 0.5
    
    # definition of reference
    ref = torch.tensor([[i]*dim for i in range(samples)]).t()
    for h in range(ref.size()[0]):
        ref[h,:] = ref[h,:]*(h+1)
    
    # ILC controller instance
    conILC = ILCctrl(dimMIMO=dim, dimSamples=samples, Le=l0)     
   
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
        last_err.append(err[:, 5])
    
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