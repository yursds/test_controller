#--------------------------------------------------------------------------------------------------#
# TEST controllerILC
#
# NOTE the control of ILC used is:
#   u_{j+1}(k) = u_{j}(k) + Le * e_{j}(k+1) + Ledot * de_{j}(k+1) + Leddot * dde_{j}(k+1)
# 
# control in ILC is DECOUPLED.
# control used in ILC is u_ILC_decoupled = torch.matmul(iM, u_).
# control used to robot from ILC is u_ILC_coupled =  torch.matmul(M, u_).
# error (and dot error) used in ILC is "decoupled" to obtain original decoupling in u_ILC_coupled.
# coupling and decoupling is done with different matrix due to different episodes.
#--------------------------------------------------------------------------------------------------#


import torch
from classes.controllerILCMB    import ILCMBctrl
from classes.pinDoublePendulum  import robDoublePendulum
from matplotlib                 import pyplot as plt

if __name__ == '__main__':
    
    # parameters
    vis_flag    = False
    dtype       = torch.float64
    episodes    = 5
    time_sim    = 10            # time [s]
    dt          = 0.01          # sample time [s]
    samples     = torch.floor(torch.tensor(time_sim/dt)).type(torch.int64)
    
    # reference
    ref     = torch.tensor([[torch.pi/3,-torch.pi/3]], dtype=dtype).T.expand(-1,samples)
    
    # load robot
    bob     = robDoublePendulum(visual=vis_flag, dt = dt, dtype=dtype)
        
    # init
    e_0     = torch.zeros((2,1), dtype=dtype)
    q_0     = bob.q0 + torch.tensor([[torch.pi,0]], dtype=dtype).T
    dq_0    = torch.zeros((2,1), dtype=dtype)
    ddq_0   = torch.zeros((2,1), dtype=dtype)
    bob.setState(q=q_0, dq=dq_0, ddq=ddq_0)
        
    # ILC controller instance
    Le0         = 0.02
    Lde0        = 0.02
    conILC      = ILCMBctrl(dimMIMO=bob.dim_q, dimSamples=samples, Le=Le0, Ledot=Lde0, dtype=dtype)

    # feedback parameters
    kp = 0.3/10
    kv = 0.08/10
    
    # logging for plot
    e_list      = []
    de_list     = []
    q_list      = []
    dq_list     = []
    ddq_list    = []
    u_list      = []
    uMB_list    = []
    uFF_list    = []
    uFB_list    = []
    
    
    # init first episode of ILC
    conILC.newEp()
    
    for ep in range(episodes):
        
        # init same conditions for ILC
        bob.setState(q=q_0, dq=dq_0, ddq=ddq_0)
        e_new  = e_0
        q_new  = q_0
        dq_new = dq_0
        ddq_new = ddq_0
        
        # init partial logging
        e_tmp      = []
        de_tmp     = []
        q_tmp      = []
        dq_tmp     = []
        ddq_tmp    = []
        u_tmp      = []
        uMB_tmp    = []
        uFF_tmp    = []
        uFB_tmp    = []
        
        # compute new feedforward control of ILC
        if ep != 0:
            conILC.stepILC()
        
        # start simulation for a single espisode
        for k in range(samples):
            
            # get useful matrix of robot
            iM = bob.getInvMass(q_new)
            M = bob.getMass(q_new)
            C = bob.getCoriolis(q_new,dq_new)
            G = bob.getGravity(q_new)
            
            if ep != 0:
                uFF = conILC.getControl()         # get ILC control
                uFF = torch.matmul(M,uFF).type(dtype)
            else:
                uFF = torch.zeros(2,1).type(dtype)
            
            # compensate and simplify dynamics
            uMB = torch.matmul(C,dq_new) + G
            # PD controller
            uFB = torch.matmul(torch.diag(torch.tensor([kp, kp])).type(dtype),e_new) + \
                    torch.matmul(torch.diag(torch.tensor([kv, kv])).type(dtype),-dq_new)
            
            # total control
            d = torch.tensor([[0,0.01*torch.sin(torch.tensor(0.1*k))]],dtype=dtype).T
            u_new = uMB+uFB+uFF+d
            u_new = bob.saturateu(u_new)
            
            # what learn ILC
            u_delta = torch.matmul(iM, uFF+uFB).type(dtype) 
            conILC.updateMemInput(u_delta)
            
            # get new info from robot bob
            q_new, dq_new = bob.getNewState(action=u_new)
            ddq_new = bob.ddq
            
            # compute output
            out = q_new
            
            # compute error
            e_new = bob.angle_normalize(ref[:, k:k+1]-out)
            de_new = -dq_new
            
            # update ILC memory
            conILC.updateMemError(e_new=torch.matmul(iM,e_new),de_new=torch.matmul(iM,de_new))        # save new_error
            
            # render
            if vis_flag:
                bob.render()
            
            # update partial logging
            e_tmp.append(e_new.flatten())
            de_tmp.append(de_new.flatten())
            q_tmp.append(q_new.flatten())
            dq_tmp.append(dq_new.flatten())
            ddq_tmp.append(ddq_new.flatten())
            
            u_tmp.append(u_new.flatten())
            uMB_tmp.append(uMB.flatten())
            uFF_tmp.append(uFF.flatten())
            uFB_tmp.append(uFB.flatten())
        
        # scale PD gain
        kp = kp
        kv = kv
        
        # update complete logging
        e_list.append(e_tmp)
        de_list.append(de_tmp)
        q_list.append(q_tmp)
        dq_list.append(dq_tmp)
        ddq_list.append(ddq_tmp)
        u_list.append(u_tmp)
        uMB_list.append(uMB_tmp)
        uFF_list.append(uFF_tmp)
        uFB_list.append(uFB_tmp) 
            

    plt.ion()  
    
    step_idx=100
    err_ep = []
    for jj in range(episodes):
        data = e_list[jj][step_idx]
        err_ep.append(data)    
    plt.figure(figsize=(10, 5))
    plt.plot(err_ep)
    plt.title(f"error in iteration step{step_idx}")
    plt.xlabel("steps")
    plt.grid()
    
    plt.figure(figsize=(10, 5))
    plt.subplot(2,1,1)
    plt.plot(e_list[0])
    plt.title("error first")
    plt.xlabel("steps")
    plt.grid()
    plt.subplot(2,1,2)
    plt.plot(e_list[-1])
    plt.title("last error")
    plt.xlabel("steps")
    plt.grid()
    
    plt.figure(figsize=(10, 5))
    plt.subplot(2,1,1)
    plt.plot(de_list[0])
    plt.title("first dot error")
    plt.xlabel("steps")
    plt.grid()
    plt.subplot(2,1,2)
    plt.plot(de_list[-1])
    plt.title("last dot error")
    plt.xlabel("steps")
    plt.grid()

    # for i in [0,-1]:
    #     plt.figure(figsize=(10, 5))
        
    #     plt.subplot(4, 1, 1)
    #     plt.plot(u_list[i])
    #     plt.title("u")
    #     plt.xlabel("steps")
    #     plt.grid()
        
    #     plt.subplot(4, 1, 2)
    #     plt.plot(uMB_list[i])
    #     plt.title("uMB")
    #     plt.xlabel("steps")
    #     plt.grid()
        
    #     plt.subplot(4, 1, 3)
    #     plt.plot(uFF_list[i])
    #     plt.title("uFF")
    #     plt.xlabel("steps")
    #     plt.grid()
        
    #     plt.subplot(4, 1, 4)
    #     plt.plot(uFB_list[i])
    #     plt.title("uFB")
    #     plt.xlabel("steps")
    #     plt.grid()
        
    #     plt.suptitle(f"Episode {i}")
    
        
    # plt.figure(figsize=(10, 5))
    # plt.subplot(1, 3, 1)
    # plt.plot(q_list[-1])
    # plt.title("q")
    # plt.xlabel("steps")
    # plt.grid()
    # plt.subplot(1, 3, 2)
    # plt.plot(dq_list[-1])
    # plt.title("dq")
    # plt.xlabel("steps")
    # plt.grid()
    # plt.subplot(1, 3, 3)
    # plt.plot(ddq_list[-1])
    # plt.title("ddq")
    # plt.xlabel("steps")
    # plt.grid()
    
    plt.ioff()
    plt.show()