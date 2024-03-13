#import numpy as np
import torch
import pinocchio as pin
import subprocess
from pinocchio.robot_wrapper    import RobotWrapper as robWrap
from example_robot_data         import load
from pinocchio.visualize        import MeshcatVisualizer


ROB_STR = 'double_pendulum'
bob:robWrap = load(ROB_STR)


class robPin():

    def __init__(self, robot:robWrap, dtype:torch.dtype=torch.float32, visual:bool=False, dt:float=0.01) -> None:

        self.robModel   = robot.model               # urdf info
        self.robData    = robot.data                # useful functions
        
        self.dtype      = dtype
        self.q0         = torch.from_numpy(pin.neutral(self.robModel)).expand(1,-1).T.type(self.dtype)
        self.dq0        = torch.zeros(self.q0.shape).type(self.dtype)
        self.ddq0       = torch.zeros(self.q0.shape).type(self.dtype)
        self.action     = torch.zeros(self.q0.shape).type(self.dtype)
        
        self.dim_q:int  = self.robModel.nq
        self.dim_dq:int = self.robModel.nv
        
        self.q          = self.q0
        self.dq         = self.dq0
        self.ddq        = self.ddq0
        
        self.dt         = dt
        
        if visual:
            
            self.robColl    = robot.collision_model
            self.robVis     = robot.visual_model
            self.viz    = MeshcatVisualizer(self.robModel, self.robColl, self.robVis)
            self.viz.initViewer(open=True)            
            self.viz.loadViewerModel("pinocchio")
            meshcat_url = self.viz.viewer.url()
            subprocess.run(['open', meshcat_url], check=True)
            self.viz.display(self.q.flatten().numpy())
                
    def getMass(self, q:torch.Tensor) -> torch.Tensor:
        """ Return Mass Matrix """    
        M = torch.from_numpy(pin.crba(self.robModel,self.robData,q.flatten().numpy())).type(self.dtype)
        return M

    def getCoriolis(self, q:torch.Tensor, dq:torch.Tensor) -> torch.Tensor:
        """ Return Coriolis Matrix """    
        C = torch.from_numpy(pin.computeCoriolisMatrix(self.robModel,self.robData,q.flatten().numpy(),dq.flatten().numpy())).type(self.dtype)
        return C

    def getInvMass(self, q:torch.Tensor) -> torch.Tensor:
        """ Return Inverse of Mass Matrix """    
        iM = torch.from_numpy(pin.computeMinverse(self.robModel,self.robData,q.flatten().numpy())).type(self.dtype)
        return iM

    def getGravity(self, q:torch.Tensor) -> torch.Tensor:
        """ Return Generalized Gravity Matrix """    
        G = torch.from_numpy(pin.computeGeneralizedGravity(self.robModel,self.robData,q.flatten().numpy())).type(self.dtype).view(-1,1)
        return G
        
    def getInvDyn(self, state:torch.Tensor, action:torch.Tensor=None) -> torch.Tensor:
        """
        Returns:
            torch.Tensor: get dot state as vector column [dq,ddq]
        """
        if action is None:
            action = torch.zeros(self.dq.shape).type(self.dtype)
        else:
            action = action.type(self.dtype)
            
        q  = state[:self.dim_q]
        dq  = state[-self.dim_dq:]
        
        iM  = self.getInvMass(q)
        C   = self.getCoriolis(q,dq)
        G   = self.getGravity(q)

        ddq = torch.matmul(iM, torch.matmul(-C,dq) - G + action)
        
        return torch.concatenate([dq, ddq],dim=0)
       
    def updateState(self, dt:float = None, action:torch.Tensor=None) -> None:
        """
        Update state [q,dq]^T variables wrt robot dynamics.

        Args:
            action (torch.Tensor, optional): input to system. If None it is set to zero.
        """
        
        if dt==None:
            dt = self.dt
            
        x       = self.getState()
        x_new   = self.rk4Step(x, action, dt)
        #x_new   = self.eulerStep(x, action, dt)
        
        self.q  = x_new[:self.dim_q]
        self.dq = x_new[-self.dim_dq:]
        
    def getNewState(self, dt:float = None, action:torch.Tensor=None) -> list[torch.Tensor, torch.Tensor]:
        """
        Update state [q,dq]^T variables wrt robot dynamics.

        Args:
            action (torch.Tensor, optional): input to system. If None it is set to zero.
        """
        if dt==None:
            dt = self.dt
        self.action = action
        self.updateState(dt, action)
        
        return [self.q, self.dq]
    
    def getState(self) -> torch.Tensor:
        """
        Returns:
            torch.Tensor: get state as vector column [q,dq]
        """
        return torch.cat([self.q,self.dq],dim=0)
    
    def getDotState(self) -> torch.Tensor:
        """
        Returns:
            torch.Tensor: get state as vector column [dq,ddq]
        """
        x = self.getState()
        u = self.action
        x_dot = self.getInvDyn(x, u)
        
        return x_dot
        
    def setState(self,q:torch.Tensor=None, dq:torch.Tensor=None, ddq:torch.Tensor=None) -> None:
        """
        Update only not None input of function.

        Args:
            q (torch.Tensor, optional): joint position variable. Defaults to None.
            dq (torch.Tensor, optional): joint velocity variable. Defaults to None.
            ddq (torch.Tensor, optional): joint acceleration variable. Defaults to None.
        """
        if q != None:
            self.q = q.type(self.dtype)
        if dq != None:
            self.dq = dq.type(self.dtype)
        if ddq != None:
            self.ddq = ddq.type(self.dtype)
         
    def rk4Step(self, x:torch.Tensor, u:torch.Tensor, dt:float = None) -> torch.Tensor:
        """Runge Kutta

        Args:
            x (torch.Tensor): state column vector [q,dq]
            u (torch.Tensor): action column vector
            dt (float): sample time [s]

        Returns:
            _type_: _description_
        """
        
        if dt==None:
            dt = self.dt
        k1 = self.getInvDyn(x, u)
        k2 = self.getInvDyn(x + k1*dt/2, u) 
        k3 = self.getInvDyn(x + k2*dt/2, u)
        k4 = self.getInvDyn(x + k3*dt, u) 
        
        x_new = x + (k1 + (k2 + k3)*2 + k4)* dt / 6
        
        return x_new 

    def eulerStep(self, x:torch.Tensor, u:torch.Tensor, dt:float = None) -> torch.Tensor:
        """Runge Kutta

        Args:
            x (torch.Tensor): state column vector [q,dq]
            u (torch.Tensor): action column vector
            dt (float): sample time [s]

        Returns:
            _type_: _description_
        """
        
        if dt==None:
            dt = self.dt
        x_dot = self.getInvDyn(x, u)
        
        x_new = x + (x_dot)* dt
        
        return x_new 

    def render(self, dt:float = None) -> None:
        
        if dt==None:
            dt = self.dt
        self.viz.display(self.q.flatten().numpy())
        self.viz.sleep(dt)
    