import torch
from classes.controllerILC  import ILCctrl

class ILCMBctrl(ILCctrl):
    """ 
    This class implements a model based ILC action ONLY for square MIMO.\n
    NOTE the control law used is:
        u_{j+1}(k) = u_{j}(k) + Le * e_{j}(k+1) + Ledot * de_{j}(k+1) + Leddot * dde_{j}(k+1)
    """
        
    
    def __init__(self, dimMIMO:int, dimSamples:int, 
                 Le:float=0.01, Ledot:float=0.0, Leddot:float=0.0, dtype=torch.float64) -> None:
        """
        This class implements a model based ILC action ONLY for square MIMO.\\
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
                
        super().__init__(dimMIMO, dimSamples, Le, Ledot, Leddot, dtype)
        
        # data memory template (of error and input) stacked in column
        self.__tmplMem.update(
            {
                'massMat': torch.Tensor(),
                'invMassMat': torch.Tensor(),
                'nleVec': torch.Tensor()
            }
        )
    
    def updateMemDecoupMat(self, M:torch.Tensor=None, iM:torch.Tensor=None, nle:torch.Tensor=None) -> None:
        """
        Store new error in a tensordict in a list.
        """
        
        if nle is None:
            nle = torch.zeros((self.dimMIMO,1),dtype=self.dtype)
        if M is None:
            M = torch.eye(2,dtype=self.dtype)
        if iM is None:
            iM = torch.eye(2,dtype=self.dtype)
            
        dictM = 'massMat'
        self.__updateMem__(M, dictM)
        
        dictIM = 'invMassMat'
        self.__updateMem__(iM, dictIM)
        
        dictNLE = 'nleVec'    
        self.__updateMem__(nle, dictNLE)
          
    
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
    conILC = ILCMBctrl(dimMIMO=dim, dimSamples=samples, Le=l0)     
   
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