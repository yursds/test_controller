import torch
from pinocchio.robot_wrapper    import RobotWrapper as robWrap
from example_robot_data         import load
from pinWrapTorch1 import robPin


ROB_STR = 'double_pendulum'

def angle_normalize(x:torch.Tensor):
    """ angle in range [-pi; pi]"""
    return ((x + torch.pi) % (2 * torch.pi)) - torch.pi


if __name__ == '__main__':
    
    
    #a = angle_normalize(torch.pi)
    
    
    robot:robWrap = load(ROB_STR)
    bob = robPin(robot, visual=True, dt = 0.01)
    
    samples = 1000
    
    q_mem = []
    dq_mem = []
    q_new  = bob.q0 + torch.tensor([[torch.pi+torch.pi/3,0]]).T
    dq_new = bob.dq0
    bob.setState(q=q_new, dq=dq_new)
    
    bob.render()
    a =  bob.getGravity(q_new)
    print(a.flatten())
    
    for k in range(samples):
        
        u_new = torch.zeros(2,1)
        #u_new = torch.tensor([[0.6489, 0.2849]]).T
        #q_new = angle_normalize(q_new)
        q_mem.append(q_new.flatten())
        dq_mem.append(dq_new.flatten())
        
        q_new, dq_new = bob.getNewState(action=u_new)
        
        #print(bob.getState().flatten())
        #print(bob.getDotState().flatten())
        #print(bob.action.flatten())
        #print(dq_new.flatten())
        
        bob.render()
        

    from matplotlib import pyplot as plt
    
    
    plt.ion()
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(q_mem)
    plt.title("q")
    plt.xlabel("steps")
    plt.grid()
    
    plt.subplot(1, 2, 2)
    plt.plot(dq_mem)
    plt.title("dq")
    plt.xlabel("steps")
    plt.grid()
    
    plt.ioff()
    plt.show()
    
    