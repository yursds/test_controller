import torch
import pinocchio as pin
from pinocchio.robot_wrapper    import RobotWrapper as robWrap
from example_robot_data         import load
from pinWrapTorch               import robPin

class robDoublePendulum(robPin):
    
    def __init__(self, dtype:torch.dtype=torch.float64, visual:bool=False, dt:float=0.01) -> None:
        """
        Class that wrap pinocchio's "double_pendulum"  with torch library. Only main methods are implemented.
        
        Args:
            robot (robWrap): class to import robot using URDF in pinocchio
            dtype (torch.dtype, optional): type of variables in class. Defaults to torch.float64.
            visual (bool, optional): to visualize robot with meshcat. Defaults to False.
            dt (float, optional): time sample [s] for integration and visualization. Defaults to 0.01 [s].
        
        NOTE: For particular method and some examples see: 
            https://docs.ros.org/en/melodic/api/pinocchio/html/namespacepinocchio.html
            https://gepettoweb.laas.fr/doc/stack-of-tasks/pinocchio/topic/doc-v2/doxygen-html/index.html

        """
        
        rob_str = 'double_pendulum'
        robot:robWrap = load(rob_str)
        super().__init__(robot=robot, dtype=dtype, visual=visual, dt=dt)
        
        # spec of franka emika panda
        dqM         = 10        #[rad/s]
        ddqM        = 2.5       #[rad/s^2]
        uM          = 87        #[Nm]
        duM         = 1000      #[Nm/s]
        
        # init past action
        self.uold   = torch.zeros(self.dim_q,1).type(self.dtype)
        
        # limits (symmetric)
        self.dqM    = torch.tensor([dqM]).type(self.dtype)
        self.ddqM   = torch.tensor([ddqM]).type(self.dtype)
        self.uM     = torch.tensor([uM]).type(self.dtype)
        self.duM    = torch.tensor([duM*self.dt]).type(self.dtype) # scale dot u is not accessible
    
    def saturatedq(self,dq)->torch.Tensor:
        for idx, value in enumerate(dq):
           if torch.abs(value) > self.dqM:
               if value < 0:
                   dq[idx] = -self.dqM
               else:
                   dq[idx] = self.dqM
        return dq
    
    def saturateddq(self,ddq)->torch.Tensor:
        
        for idx, value in enumerate(ddq):
           if torch.abs(value) > self.ddqM:
               if value < 0:
                   ddq[idx] = -self.ddqM
               else:
                   ddq[idx] = self.ddqM
        return ddq
    
    def saturateu(self, u:torch.Tensor)->torch.Tensor:
        
        delta_u = u-self.uold
        for idx, value in enumerate(delta_u):
           if torch.abs(value) > self.duM:
               if value < 0:
                   delta_u[idx] = -self.duM
               else:
                   delta_u[idx] = self.duM
        u = self.uold + delta_u
        
        for idx, value in enumerate(u):
            if torch.abs(value) > self.uM:
                if value < 0:
                    u[idx] = -self.uM
                else:
                    u[idx] = self.uM
        self.uold = u
        return u
        
    def getInvDyn(self, q:torch.Tensor,dq:torch.Tensor,ddq:torch.Tensor,dampFlag:bool=True) -> torch.Tensor:
        """ 
        Get inverse dynamic (tau) of the system given current state and action.
        Compute all necessary matrices from state.
        
        Args:
            q (torch.Tensor): joint position variables.
            dq (torch.Tensor): joint velocity variables.
            ddq (torch.Tensor): joint acceleration variables.
            dampFlag (bool, optional): boolean to use damping in dynamics. Defaults to False.

        Returns:
            torch.Tensor: tau as vector column
        """
        
        tau = super().getInvDyn(q,dq,ddq,dampFlag)
        tau = self.saturateu(tau)
        
        return tau
        
    def getForwDyn2(self, state:torch.Tensor, action:torch.Tensor=None, dampFlag:bool=True) -> torch.Tensor:
        """ 
        Get forward dynamic (dot state) of the system given current state and action.
        Compute all necessary matrices from state.
        
        Args:
            state (torch.Tensor): [q,dq] column vector
            action (torch.Tensor, optional): action to system. Defaults to None, action is set to zero.
            dampFlag (bool, optional): boolean to use damping in dynamics. Defaults to False.

        Returns:
            torch.Tensor: dot state as vector column [dq,ddq]^T
        """
            
        state[-self.dim_q:]  = self.saturatedq(state[-self.dim_q:])
        
        x_dot = super().getForwDyn(state, action, dampFlag)
        
        x_dot[-self.dim_q:]  = self.saturateddq(x_dot[-self.dim_q:])
        
        return x_dot
   
    def getNewState(self, dt:float = None, action:torch.Tensor=None) -> list[torch.Tensor, torch.Tensor]:
            """
            Update state [q,dq]^T variables wrt robot dynamics.

            Args:
                action (torch.Tensor, optional): input to system. If None it is set to zero.
            Returns:
                list[torch.Tensor, torch.Tensor]: [self.q, self.dq]
            """
            
            action = self.saturateu(action)
            q, dq = super().getNewState(dt,action)
            self.q = self.angle_normalize(q)
            self.dq = self.saturatedq(dq)
            self.ddq = self.saturateddq(self.ddq)
            
            return [self.q, self.dq]

    def angle_normalize(self, x:torch.Tensor)->torch.Tensor:
        """ angle in range [-pi; pi]"""
        
        sx = torch.sin(x)
        cx = torch.cos(x)
        x = torch.atan2(sx,cx)
        return x




if __name__ == '__main__':
    
    # parameters
    vis_flag    = False
    dt          = 0.01
    time_sim    = 5         # time [s]
    dtype       = torch.float64
    samples     = torch.floor(torch.tensor(time_sim/dt)).type(torch.int64)

    # load robot
    bob = robDoublePendulum(visual=vis_flag, dt = dt, dtype=dtype)
        
    # init
    q_0     = bob.q0 + torch.tensor([[torch.pi+torch.pi/3,0]], dtype=dtype).T
    dq_0    = torch.zeros((2,1), dtype=dtype)
    ddq_0   = torch.zeros((2,1), dtype=dtype)
    bob.setState(q=q_0, dq=dq_0, ddq=ddq_0)
    
    # init
    new_e   = torch.zeros((2,1), dtype=dtype)
    q_new   = q_0
    dq_new  = dq_0
    ddq_new = ddq_0
    kp      = 0.3
    kv      = 0.05
    
    # reference
    ref     = torch.tensor([[torch.pi/3,-torch.pi/3]], dtype=dtype).T.expand(-1,samples)
    
    # logging for plot
    e_list      = []
    u_list      = []
    q_list      = []
    dq_list     = []
    ddq_list    = []
    
    for k in range(samples):
        
        # set to zero for free response
        # u_new = torch.zeros((bob.dim_dq,1), dtype=dtype)
        
        # computed torque
        u_new = bob.getInvDyn(q_new, dq_new, torch.zeros(2,1))+ \
            torch.matmul(torch.diag(torch.tensor([kp, kp])).type(dtype),new_e) + \
            torch.matmul(torch.diag(torch.tensor([kv, kv])).type(dtype),-dq_new)
            
        # update state
        q_new, dq_new = bob.getNewState(action=u_new)
        ddq_new = bob.ddq
        q_new = bob.angle_normalize(q_new)
        bob.setState(q=q_new, dq=dq_new)
        
        # update output
        out = q_new
        new_e = bob.angle_normalize(ref[:, k:k+1]-out)

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