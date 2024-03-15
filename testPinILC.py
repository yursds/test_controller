from controllerILC import ctrlILC
import torch
from pinocchio.robot_wrapper    import RobotWrapper as robWrap
from example_robot_data         import load
from pinWrapTorch import robPin


ROB_STR = 'double_pendulum'

def angle_normalize(x:torch.Tensor):
    """ angle in range [-pi; pi]"""
    
    sx = torch.sin(x)
    cx = torch.cos(x)
    x = torch.atan2(sx,cx)
    return x


if __name__ == '__main__':
    
    robot:robWrap = load(ROB_STR)
    vis_flag = True
    dt = 0.01
    bob = robPin(robot, visual=vis_flag, dt = dt)
    
    samples = 1000
    
    q_mem = []
    dq_mem = []
    print(bob.q0)
    q_new  = bob.q0 + torch.tensor([[torch.pi+torch.pi/10*0,0]]).T.type(torch.float32)
    
    #q_new = bob.q0
    dq_new = bob.dq0.type(torch.float32) #+torch.tensor([[0,0]]).T.type(torch.float32)
    ddq_new = bob.ddq0.type(torch.float32)
    #print(bob.q0.flatten())
    
    bob.setState(q=q_new, dq=dq_new)
    if vis_flag:
        bob.render()
    
    dimU = q_new.shape[0]
    dimE = dimU
    episodes = 10
    
    # definition of reference
    ref = torch.tensor([[torch.pi/3,-torch.pi/3]]).T.expand(-1,samples)
    
    # ILC controller instance
    conILC = ctrlILC(dimU=dimU,dimE=dimE,dimSamples=samples)
   
    new_e = torch.zeros(2,1).type(torch.float32)
    q_new = angle_normalize(q_new).type(torch.float32)
  
    #mem = conILC.getMemory()                      # get memory
    q_list = []
    # start ILC iteration   
    conILC.newEp()
    
    for ep in range(episodes):
        if ep != 0:
            conILC.stepILC()
        
        new_e  =torch.zeros(dimE,1).type(torch.float32)
        q_new  = bob.q0 + torch.tensor([[torch.pi+torch.pi/10*0,0]]).T.type(torch.float32)
        dq_new = bob.dq0.type(torch.float32)
        ddq_new = bob.ddq0.type(torch.float32)
        
        bob.setState(q=q_new, dq=dq_new, ddq=ddq_new)
        q_new = angle_normalize(q_new).type(torch.float32)
        
        for k in range(samples):
            
            if ep != 0:
                new_u = conILC.getControl()*1         # get ILC control
            else:
                new_u = torch.zeros(dimU,1).type(torch.float32)
                
            u_CT = bob.getDyn(q_new, dq_new, torch.zeros(2,1)).type(torch.float32)+ \
            torch.matmul(torch.diag(torch.tensor([0.1, 0.1])).type(torch.float32),new_e.type(torch.float32)).type(torch.float32) + \
            torch.matmul(torch.diag(torch.tensor([0.05, 0.05])).type(torch.float32),-dq_new.type(torch.float32))
        
            u = new_u + u_CT
            conILC.updateMemInput(u-bob.getGravity(q_new))     # save input for next episode (consider all inputs)
            
            # update state
            q_new, dq_new = bob.getNewState(action=u)
            ddq_new = bob.ddq
            q_new = angle_normalize(q_new)
            bob.setState(q=q_new, dq=dq_new)
        
            # update output
            out = q_new             # simulate robot (simple function)
            new_e = angle_normalize(ref[:, k:k+1]-out)              # get new error
            
            conILC.updateMemError(new_e)        # save new_error
            
           
            q_mem.append(q_new.flatten())
            
            if ep == episodes-1:
                bob.render()
        mem = conILC.getMemory()    
            
        q_list.append(q_mem)
        
    from matplotlib import pyplot as plt

    last_err = []
    for td in mem:
        err = td["error"]
        last_err.append(err[:, -100].T)

    
    e_list0:torch.Tensor = mem[0]["error"]
    u_list0 = mem[0]["input"]
    q_list0 = q_list[0]
    
    e_list1:torch.Tensor = mem[1]["error"]
    u_list1 = mem[1]["input"]
    q_list1 = q_list[1]
    
    e_listL = mem[-1]["error"]
    u_listL = mem[-1]["input"]
    q_listL = q_list[-1]
    
    plt.ion()

    plt.figure(figsize=(10, 5))
    plt.plot(last_err)
    plt.title("error iteration")
    plt.xlabel("steps")
    plt.grid()
    
    plt.figure(figsize=(10, 5))
    plt.subplot(3, 3, 1)
    plt.plot(e_list0.T)
    plt.title("error")
    plt.xlabel("steps")
    plt.grid()
    
    plt.subplot(3, 3, 2)
    plt.plot(q_list0)
    plt.title("q")
    plt.xlabel("steps")
    plt.grid()
    
    plt.subplot(3, 3, 3)
    plt.plot(u_list0.T)
    plt.title("u")
    plt.xlabel("steps")
    plt.grid()
    
    plt.subplot(3, 3, 4)
    plt.plot(e_list1.T)
    plt.title("error")
    plt.xlabel("steps")
    plt.grid()
    
    plt.subplot(3, 3, 5)
    plt.plot(q_list1)
    plt.title("q")
    plt.xlabel("steps")
    plt.grid()
    
    plt.subplot(3, 3, 6)
    plt.plot(u_list1.T)
    plt.title("u")
    plt.xlabel("steps")
    plt.grid()
    
    plt.subplot(3, 3, 7)
    plt.plot(e_listL.T)
    plt.title("error")
    plt.xlabel("steps")
    plt.grid()
    
    plt.subplot(3, 3, 8)
    plt.plot(q_listL)
    plt.title("q")
    plt.xlabel("steps")
    plt.grid()
    
    plt.subplot(3, 3, 9)
    plt.plot(u_listL.T)
    plt.title("u")
    plt.xlabel("steps")
    plt.grid()
    
    print('Complete')
    plt.ioff()
    plt.show()
    
    print("finish")
    