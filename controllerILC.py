import torch
from tensordict     import TensorDict

SCALE_Q = 1
SCALE_L = 1

class controllerILC():
    """ This class implements a model free ILC action ONLY for square MIMO. """
    
    def __init__(self, dim_u:int, dim_e:int, Q=None, L=None):
        """ Args:
                dim_u (int): dimension of input vector (single step).
                dim_e (int): dimention of error vector (single step).
                Q (torch.Tensor[optional]): Q-filter matrix (square matrix)
                L (torch.Tensor[optional]): learning gain matrix (square matrix) """

        # check        
        if dim_u != dim_e:
            raise TypeError("ILC class implemented only for square MIMO.")
        
        # init Q and L
        if Q is not None:
            if not isinstance(Q, torch.Tensor):
                raise TypeError("Q must be a torch.Tensor")
            rows, cols = Q.size()
            if rows != cols:
                raise ValueError("Q must be a square matrix")
            self.Q:torch.Tensor = Q
        else:
            self.Q:torch.Tensor = torch.tril(torch.ones(dim_u, dim_u))*SCALE_Q
        if L is not None:
            if not isinstance(L, torch.Tensor):
                    raise TypeError("L must be a torch.Tensor")
            rows, cols = L.size()
            if rows != cols:
                raise ValueError("L must be a square matrix")
            self.L:torch.Tensor = L
        else:
            self.L:torch.Tensor = torch.tril(torch.ones(dim_e, dim_e))*SCALE_L
            
        self.u:torch.Tensor     = torch.zeros([dim_u,1])   # last control input
        self.e:torch.Tensor     = torch.zeros([dim_e,1])   # last error
        self.flagNewEp:bool     = True             # flag for new episode
        self.ep:int             = 0
        self.idx:int            = 0
        self.mem:list           = []
        
        # data memory template (of error and input) 
        self.__tmplMem:TensorDict = TensorDict(
            {
            'error': torch.Tensor(), 
            'input' : torch.Tensor()
            },
            batch_size=[]
        )
        
    def __updateMem__(self, data:torch.Tensor, dict:str) -> None:
        """
        Store new data in a tensordict in a list.
        For new episode append new tensordict.
        """
        if self.flagNewEp:
            self.flagNewEp = False
           
        # use newest tensordict
        tmp_mem:torch.Tensor = self.mem[-1][dict]
        tmp_mem = torch.cat([tmp_mem.clone(),data],dim=1)      # stack respect row
        
        self.mem[-1][dict] = tmp_mem
           
    def updateMemError(self, e_new:torch.Tensor) -> None:
        """
        Store new error in a tensordict in a list.
        For new episode append new tensordict.
        """
        dict:str='error'
        self.__updateMem__(e_new, dict)
    
    def updateMemInput(self, u_new:torch.Tensor) -> None:
        """
        Store new input in a tensordict in a list.
        For new episode append new tensordict.
        """
        dict:str='input'
        self.__updateMem__(u_new, dict)
   
    def newEp(self) -> None:
        """ 
        Reset flag for new episode.
        Reset step index. 
        """
        self.flagNewEp = True
        self.mem.append(self.__tmplMem.clone())
        self.idx = 0
        self.ep += 1
        
    def updateQ(self, newQ:torch.Tensor) -> None:
        self.Q = newQ
    
    def updateL(self, newL:torch.Tensor) -> None:
        self.L = newL

    def getMemory(self) -> torch.Tensor:
        return self.mem

    def getControl(self) -> torch.Tensor:
        return self.u
 
    def step(self) -> None:
        """
        Single step of ILC.
        Update control.
        """
        k   = self.idx
        Q   = self.Q
        L   = self.L
        
        if self.ep == 0:
            print("banana")
            u_old:torch.Tensor  = self.mem[-2]["input"][:,k:k+1]
            e_old:torch.Tensor  = self.mem[-2]["error"][:,k:k+1]
        else:
            u_old:torch.Tensor  = self.mem[-2]["input"][:,k:k+1]
            e_old:torch.Tensor  = self.mem[-2]["error"][:,k:k+1]
            
        u_new:torch.Tensor = torch.mm(Q,(u_old+torch.mm(L,e_old)))
        self.u = u_new
        
        self.idx += 1 
                          
    def resetAll(self) -> None:
        """
        Reset flag, index and memory
        """
        self.flagNewEp = True
        self.idx = 0
        self.mem = []

    def firstEpLazy(self, ref:torch.Tensor) -> None:
        """ no input and save error as reference"""
        self.updateMemError(ref)
        self.updateMemInput(torch.zeros(self.u.size()))
        
                
        
if __name__ == '__main__':
    
    # definition of reference
    ref = torch.tensor([[i,i,i] for i in range(10)]).t()
    
    conILC = controllerILC(3,3)     # ILC controller instance
    conILC.newEp()                  # start new episode

    # initialization of ILC memory
    # controller do nothing and save error as reference
    for k in range(ref.shape[1]):
        conILC.firstEpLazy(ref[:, k:k+1])
        
    a = conILC.getMemory()                      # get memory
    
    # start ILC iteration
    for k in range(50):
        
        conILC.newEp()                          # start new episode
        
        for k in range(ref.shape[1]):
            
            conILC.step()
            new_u = conILC.getControl()
            conILC.updateMemInput(new_u)        # save new_control
            out = new_u/2                       # simulate robot (simple function)
            new_e = ref[:, k:k+1]-out           # get new error
            conILC.updateMemError(new_e)        # save new_error
            
        b = conILC.getMemory()
    
    
    from matplotlib import pyplot as plt

    last_err = []

    # Ciclo su ciascun TensorDict in b
    for tensor_dict in b:
        
        # Estrai il tensore associato a "error"
        errore_tensor = tensor_dict["error"]
        
        # Estrai l'ultima colonna del tensore
        ultima_colonna = errore_tensor[:, -1]
        
        # Aggiungi l'ultima colonna alla lista delle ultime colonne
        last_err.append(ultima_colonna)
    
    plt.ion()


    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(last_err)
    plt.title("error")
    plt.xlabel("iteration")

    print('Complete')
    plt.ioff()
    plt.show()
    
    conILC.resetAll()
    b = conILC.getMemory()
    print("my memory lost", b)
    print("finish")