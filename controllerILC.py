import torch
from tensordict     import TensorDict, TensorDictBase

SCALE_Q = 0.1
SCALE_L = 0.1

class controllerILC():
    """ This class implements a model free ILC action. """
    
    def __init__(self, dim_u:torch.int32, dim_e:torch.int32, Q=None, L=None):
        """ Args:
                dim_u (int16): dimension of input vector (single step).
                dim_e (int16): dimention of error vector (single step).
                dt (float16): sampling time (single step).
                Q (torch.Tensor[optional]): Q-filter matrix (square matrix)
                L (torch.Tensor[optional]): learning gain matrix (square matrix) """
        
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
            
        self.u:torch.Tensor     = torch.Tensor()   # last control input
        self.e:torch.Tensor     = torch.Tensor()   # last error
        self.flagNewEp:bool     = True             # flag for new episode
        self.ep:int             = 0
        self.idx:int            = 0
        self.mem:list           = []
        
        # data memory template (of error and input) 
        self.__tmplMem:TensorDict = TensorDict(
            {
            'error': torch.Tensor(self.e), 
            'input': torch.Tensor(self.u)
            },
            batch_size=[]
        )
        
        self.K_P = 0.1 * torch.eye(dim_u, dim_u)    #
        self.K_V = 0.01 * torch.eye(dim_u, dim_u)   #
        self.K_A = 0.001 * torch.eye(dim_u, dim_u)  #
    
    def updateMemory(self, e_new:torch.Tensor, u_new:torch.Tensor) -> None:
        """
        Store new error and input a list of tensordict.
        For new episode, create new tensordict to store new data.
        """
        if self.flagNewEp:
            self.flagNewEp = False
            self.mem.append(self.__tmplMem)
            tmp_memE = e_new
            tmp_memU = u_new
        else:    
            # use newest tensordict
            tmp_memE:torch.Tensor = self.mem[-1]['error']        
            tmp_memU:torch.Tensor = self.mem[-1]['input']
            tmp_memE = torch.cat(tmp_memE,e_new,dim=1)      # stack respect row
            tmp_memU = torch.cat(tmp_memU,u_new,dim=1)
            
        self.mem[-1]['error'] = tmp_memE
        self.mem[-1]['input'] = tmp_memU
                
    def resetEp(self) -> None:
        """ 
        Reset flag for new episode.
        Reset step index. 
        """
        self.flagNewEp = True
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
        k:int               = self.idx
        
        if self.ep == 0:
            u_old:torch.Tensor  = self.mem[-2]["input"][:,k]
            e_old:torch.Tensor  = self.mem[-2]["error"][:,k]
        else:
            u_old:torch.Tensor  = self.mem[-2]["input"][:,k]
            e_old:torch.Tensor  = self.mem[-2]["error"][:,k]
            
        Q:torch.Tensor      = self.Q
        L:torch.Tensor      = self.L
        
        u_new:torch.Tensor = Q*(u_old+L*e_old)
        self.u = u_new
        
        self.idx += 1 
                          
    def resetAll(self) -> None:
        """
        Reset flag, index and memory
        """
        self.flagNewEp = True
        self.idx = 0
        self.mem = []
        
        
if __name__ == '__main__':
    
    a = controllerILC(2,2)
    ee = torch.tensor([2,1]).view(-1,1)
    #uu = torch.tensor([3,5]).view(-1,1)
    
    for _ in range(10):
        a.step()
        uN = a.getControl()
        a.updateMemory(ee,uN)
        
    a.resetEp()
    for _ in range(5):
        a.step()
        uN = a.getControl()
        a.updateMemory(ee,uN)
        
    b = a.getMemory()
    print("my memory", b)
    a.resetAll()
    b = a.getMemory()
    print("my memory lost", b)
    
    print("finish")