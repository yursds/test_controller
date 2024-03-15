import torch
import pinocchio as pin
import subprocess
from pinocchio.robot_wrapper    import RobotWrapper as robWrap
from pinocchio.visualize        import MeshcatVisualizer
from example_robot_data         import load


class robPin():

    def __init__(self, robot:robWrap, dtype:torch.dtype=torch.float64, visual:bool=False, dt:float=0.01) -> None:

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
            self.viz        = MeshcatVisualizer(self.robModel, self.robColl, self.robVis)
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
    
    def getDamping(self):
        
        d_vec = self.robModel.damping
        D = torch.diag(torch.from_numpy(d_vec)).type(self.dtype)
        
        return D
    
    def getInvDyn(self, state:torch.Tensor, action:torch.Tensor=None) -> torch.Tensor:
        """
        Returns:
            torch.Tensor: get dot state as vector column [dq,ddq]
        """
        if action is None:
            action = torch.zeros(self.dq.shape()).type(self.dtype)
        else:
            action = action.type(self.dtype)
            
        q  = state[:self.dim_q]
        dq  = state[-self.dim_dq:]
        
        iM  = self.getInvMass(q)
        C   = self.getCoriolis(q,dq)
        G   = self.getGravity(q)
        D   = self.getDamping()
        
        ddq = torch.matmul(iM, torch.matmul(-C-D*0,dq) - G + action)
        
        return torch.cat([dq, ddq],dim=0)
    
    def getDyn(self, q:torch.Tensor,dq:torch.Tensor,ddq:torch.Tensor) -> torch.Tensor:
        """
        Returns:
            torch.Tensor: get dot state as vector column [dq,ddq]
        """
        q = q.type(self.dtype)
        dq = dq.type(self.dtype)
        ddq = ddq.type(self.dtype)
        M   = self.getMass(q)
        C   = self.getCoriolis(q,dq)
        G   = self.getGravity(q)
        D   = self.getDamping()
        
        tau = torch.matmul(M,ddq) + torch.matmul(C+D*0,dq) + G
        
        return tau
    
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
        
        self.q      = x_new[:self.dim_q]
        self.dq     = x_new[-self.dim_dq:]
        self.ddq    = self.getInvDyn(x, action)[-self.dim_dq:]
                 
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
            
        # dt0 = dt/10
        # dt_ = dt0
        # while dt_< dt:
        #     k1 = self.getInvDyn(x, u)
        #     k2 = self.getInvDyn(x + k1*dt0/2, u) 
        #     k3 = self.getInvDyn(x + k2*dt0/2, u)
        #     k4 = self.getInvDyn(x + k3*dt0, u) 
            
        #     x = x + (k1 + (k2 + k3)*2 + k4)* dt0 / 6
        #     dt_ += dt0
        
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

    def pinIntegrateStep(self, x:torch.Tensor, u:torch.Tensor, dt:float = None) -> torch.Tensor:
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
        
        #x_new = x + (x_dot)* dt
        x_new = pin.integrate(self.robModel,x.flatten().numpy(),x_dot.flatten().numpy())
        return x_new 

    def render(self, dt:float = None) -> None:
        
        if dt==None:
            dt = self.dt
        self.viz.display(self.q.flatten().numpy())
        self.viz.sleep(dt)



def angle_normalize(x:torch.Tensor):
    """ angle in range [-pi; pi]"""
    
    sx = torch.sin(x)
    cx = torch.cos(x)
    x = torch.atan2(sx,cx)
    return x

ROB_STR = 'double_pendulum'


if __name__ == '__main__':
    
    # parameters
    vis_flag = True
    dt = 0.01
    samples = 2000
    
    # load robot
    robot:robWrap = load(ROB_STR)
    bob = robPin(robot, visual=vis_flag, dt = dt)
    
    # logging for plot
    e_list      = []
    u_list      = []
    q_list      = []
    dq_list     = []
    ddq_list    = []
    
    # init
    q_new  = bob.q0 + torch.tensor([[torch.pi+torch.pi/3,0]]).T
    dq_new = bob.dq0
    bob.setState(q=q_new, dq=dq_new)
    
    if vis_flag:
        bob.render()
    
    # reference and init error
    ref = torch.tensor([[torch.pi/3,-torch.pi/3]]).T.expand(-1,samples)
    new_e=torch.zeros(2,1).type(torch.float64)
    
    for k in range(samples):
        
        # set to zero for free response
        u_new = torch.zeros(bob.dim_dq,1)
        
        # computed torque
        """ u_new = bob.getDyn(q_new, dq_new, torch.zeros(2,1).type(torch.float64))+ \
            torch.matmul(torch.diag(torch.tensor([0.1, 0.1])).type(torch.float64),new_e) + \
            torch.matmul(torch.diag(torch.tensor([0.05, 0.05])).type(torch.float64),-dq_new)
         """
        # update state
        q_new, dq_new = bob.getNewState(action=u_new)
        ddq_new = bob.ddq
        q_new = angle_normalize(q_new)
        bob.setState(q=q_new, dq=dq_new)
        
        # update output
        out = q_new
        new_e = angle_normalize(ref[:, k:k+1]-out)

        # update logging
        q_list.append(q_new.flatten())
        dq_list.append(dq_new.flatten())
        ddq_list.append(ddq_new.flatten())
        e_list.append(new_e.flatten())
        u_list.append(u_new.flatten())
        
        # render
        if vis_flag:
            bob.render()
        
    from matplotlib import pyplot as plt
    
    plt.ion()

    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(e_list)
    plt.title("error")
    plt.xlabel("steps")
    plt.grid()
    
    plt.subplot(1, 2, 2)
    plt.plot(u_list)
    plt.title("u")
    plt.xlabel("steps")
    plt.grid()
    
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 3, 1)
    plt.plot(q_list)
    plt.title("q")
    plt.xlabel("steps")
    plt.grid()
    
    plt.subplot(1, 3, 2)
    plt.plot(dq_list)
    plt.title("dq")
    plt.xlabel("steps")
    plt.grid()
    
    plt.subplot(1, 3, 3)
    plt.plot(ddq_list)
    plt.title("ddq")
    plt.xlabel("steps")
    plt.grid()
    
    plt.ioff()
    plt.show()