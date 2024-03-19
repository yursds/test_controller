import torch
from controllerILC import ctrlILC
from pinDoublePendulum import robDoublePendulum

if __name__ == '__main__':
    
    # parameters
    vis_flag    = False
    dt          = 0.01
    episodes    = 10
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
    e_0   = torch.zeros((2,1), dtype=dtype)
    
    # render
    if vis_flag:
        bob.render()
    
    # reference
    ref     = torch.tensor([[torch.pi/3,-torch.pi/3]], dtype=dtype).T.expand(-1,samples)
    ref     = bob.angle_normalize(ref)
    
    # definition of reference
    #ref = q_0 + torch.tensor([[i*0.005]*2 for i in range(samples)]).T.type(dtype)

    # ILC controller instance
    conILC = ctrlILC(dimMIMO=2, dimSamples=samples, Le=0.01, Ledot = 0.00, dtype=dtype)
    
    kp = 0.3
    kv = 0.05
    
    # logging for plot
    e_list      = []
    q_list      = []
    dq_list     = []
    ddq_list    = []
    u_list      = []
    uMB_list    = []
    uFF_list    = []
    uFB_list    = []
    
    conILC.newEp()
    for ep in range(episodes):
        
        bob.setState(q=q_0, dq=dq_0, ddq=ddq_0)
        e_new  = e_0
        q_new  = q_0
        dq_new = dq_0
        ddq_new = ddq_0
        kp = kp
        kv = kv
        
        e_tmp      = []
        q_tmp      = []
        dq_tmp     = []
        ddq_tmp    = []
        u_tmp      = []
        uMB_tmp    = []
        uFF_tmp    = []
        uFB_tmp    = []
        
        if ep != 0:
            conILC.stepILC()
        
        for k in range(samples):
            
            if ep != 0:
                #u_ff = torch.matmul(bob.getMass(q_new), conILC.getControl())         # get ILC control
                uFF = conILC.getControl()         # get ILC control
            else:
                uFF = torch.zeros(2,1)
            
            # set to zero for free response
            # u_new = torch.zeros((bob.dim_dq,1), dtype=dtype)
            #uMB = bob.getInvDyn(q_new, dq_new, torch.zeros(2,1))*0
            uMB = bob.getGravity(q_new)
            
            uFB = torch.matmul(torch.diag(torch.tensor([kp, kp])).type(dtype),e_new) + \
                    torch.matmul(torch.diag(torch.tensor([kv, kv])).type(dtype),-dq_new)
            
            u_new = uMB+uFB+uFF            # update state
            u_new = bob.saturateu(u_new)
            u_delta = u_new-uMB
            conILC.updateMemInput(u_delta)     # save input for next episode (consider all inputs)
            
            q_new, dq_new = bob.getNewState(action=u_new)
            ddq_new = bob.ddq
            q_new = bob.angle_normalize(q_new)
            bob.setState(q=q_new, dq=dq_new)
            
            # update output
            out = q_new
            e_new = bob.angle_normalize(ref[:, k:k+1]-out)
            conILC.updateMemError(e_new)        # save new_error
            
            # render
            if vis_flag:
                bob.render()
            
            # update logging
            e_tmp.append(e_new.flatten())
            q_tmp.append(q_new.flatten())
            dq_tmp.append(dq_new.flatten())
            ddq_tmp.append(ddq_new.flatten())
            
            u_tmp.append(u_new.flatten())
            uMB_tmp.append(uMB.flatten())
            uFF_tmp.append(uFF.flatten())
            uFB_tmp.append(uFB.flatten())
        
        mem = conILC.getMemory()
        
        e_list.append(e_tmp)
        q_list.append(q_tmp)
        dq_list.append(dq_tmp)
        ddq_list.append(ddq_tmp)
        u_list.append(u_tmp)
        uMB_list.append(uMB_tmp)
        uFF_list.append(uFF_tmp)
        uFB_list.append(uFB_tmp) 
            
        from matplotlib import pyplot as plt
    
    plt.ion()

    for i in range(episodes):
        plt.figure(figsize=(10, 5))
        
        plt.subplot(4, 1, 1)
        plt.plot(u_list[i])
        plt.title("u")
        plt.xlabel("steps")
        plt.grid()
        
        plt.subplot(4, 1, 2)
        plt.plot(uMB_list[i])
        plt.title("uMB")
        plt.xlabel("steps")
        plt.grid()
        
        plt.subplot(4, 1, 3)
        plt.plot(uFF_list[i])
        plt.title("uFF")
        plt.xlabel("steps")
        plt.grid()
        
        plt.subplot(4, 1, 4)
        plt.plot(uFB_list[i])
        plt.title("uFB")
        plt.xlabel("steps")
        plt.grid()
        
        plt.suptitle(f"Episode {i+1}")
    
    
    plt.figure(figsize=(10, 5))
    plt.plot(e_list[-1])
    plt.title("error")
    plt.xlabel("steps")
    plt.grid()
    
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 3, 1)
    plt.plot(q_list[-1])
    plt.title("q")
    plt.xlabel("steps")
    plt.grid()
    
    plt.subplot(1, 3, 2)
    plt.plot(dq_list[-1])
    plt.title("dq")
    plt.xlabel("steps")
    plt.grid()
    
    plt.subplot(1, 3, 3)
    plt.plot(ddq_list[-1])
    plt.title("ddq")
    plt.xlabel("steps")
    plt.grid()
    
    plt.ioff()
    plt.show()