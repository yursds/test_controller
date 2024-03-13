from controllerILC import ctrlILC
import torch
import pinocchio as pin
from pinocchio.robot_wrapper    import RobotWrapper as robWrap
from pinocchio.visualize        import MeshcatVisualizer
from example_robot_data         import load
import subprocess
from pinWrapTorch import robPin


ROB_STR = 'double_pendulum'

def angle_normalize(x:torch.Tensor):
    """ angle in range [0; 2*pi]"""
    return ((x + torch.pi) % (2 * torch.pi)) - torch.pi


if __name__ == '__main__':
    
    robot:robWrap = load(ROB_STR)
    bob = robPin(robot)
        
    # pinocchio objects
    robModel       = robot.model               # urdf info
    robData        = robot.data                # useful functions
    robColl        = robot.collision_model
    robVis         = robot.visual_model
    
    #NUMPY
    # -----------------------------------------------------------------
    q0:torch.Tensor  = pin.neutral(robModel)+[torch.pi+torch.pi/3, torch.pi/3*0]
    dq0:torch.Tensor = torch.zeros(q0.shape).numpy()
    pin.computeAllTerms(robModel,robData,q0,dq0)
    
    q      = q0
    dq     = dq0
    u      = torch.zeros(q0.shape).numpy()
    # -----------------------------------------------------------------
    
    viz = MeshcatVisualizer(robModel, robColl, robVis)
    viz.initViewer(open=True)            
    viz.loadViewerModel("pinocchio")
    viz.display(q)
    meshcat_url = viz.viewer.url()
    subprocess.run(['open', meshcat_url], check=True)
    dt = 0.01
    
    
    dimU = q0.shape[0]
    dimE = dimU
    samples = 1000
    episodes = 2
    # definition of reference
    #ref = torch.linspace(0,torch.pi,samples).expand(dimE,-1)
    ref = torch.tensor([[torch.pi*0,torch.pi/3]]).T.expand(-1,samples)
    #for h in range(ref.size()[0]):
    #    ref[h,:] = ref[h,:]*(h+1)
    
    # ILC controller instance
    conILC = ctrlILC(dimU=dimU,dimE=dimE,dimSamples=samples)
   
    # initialization of ILC memory
    conILC.newEp()                              # start new episode
    q_list= []
    q_mem = []
    
    new_e=torch.zeros(dimE,1)
    q_new   = q0
    dq_new  = dq0
    
    for k in range(samples):
        
        #FEEDBACK
        u_new = torch.matmul(torch.eye(dimU,dtype=torch.float64),new_e.type(torch.float64))*0.5
        # controller do nothing and save error as reference
        conILC.updateMemInput(u_new)        # save input for next episode (consider all inputs)
        
        pin.computeAllTerms(robModel,robData,q_new,dq_new)
            
        iMmat:torch.Tensor  = torch.from_numpy(pin.computeMinverse(robModel,robData,q_new)).type(torch.float64)
        #Mmat:torch.Tensor   = torch.from_numpy(pin.crba(robModel,robData,q))
        Cmat:torch.Tensor   = torch.from_numpy(pin.computeCoriolisMatrix(robModel,robData,q_new,dq_new)).type(torch.float64)
        Gmat:torch.Tensor   = torch.from_numpy(pin.computeGeneralizedGravity(robModel,robData,q_new).reshape(-1,1)).type(torch.float64)
        ddq = torch.matmul(iMmat,torch.matmul(-Cmat,torch.from_numpy(dq_new.reshape(-1,1)).type(torch.float64))-Gmat-u_new*0)
        
        kinEn:float = pin.computeKineticEnergy(robModel,robData,q_new, dq_new)
        potEn:float = pin.computePotentialEnergy(robModel,robData,q_new)
        
        totEn = kinEn + potEn
                
        # NOTE implementare runge_kutta
        dq_new  = dq_new + ddq.numpy().squeeze()*dt
        q_new   = angle_normalize(q_new + 0.5*dq_new*dt)
        
        out = torch.from_numpy(q_new.reshape(-1,1))
        
        new_e = angle_normalize(ref[:, k:k+1]-out)
        
        #print(new_e*180/torch.pi)
        #new_e = (ref[:, k:k+1]-out)                         # get new error
        conILC.updateMemError(new_e)                        # save new_error
        #viz.display(q_new)
        #viz.sleep(dt)
        #conILC.firstEpLazy((ref[:, k:k+1]-torch.from_numpy(q0.reshape(-1,1))).type(torch.float32))
        q_mem.append(totEn)
        
    q_list.append(q_mem)
    mem = conILC.getMemory()                      # get memory

    from matplotlib import pyplot as plt
    
    e_list0:torch.Tensor = mem[0]["error"]
    u_list0 = mem[0]["input"]
    q_list0 = q_list[0]
    
    plt.ion()

    plt.figure(figsize=(10, 5))
    plt.subplot(1, 3, 1)
    plt.plot(e_list0.T)
    plt.title("error")
    plt.xlabel("steps")
    plt.grid()
    
    plt.subplot(1, 3, 2)
    plt.plot(q_list0)
    plt.plot(ref)
    plt.title("q")
    plt.xlabel("steps")
    plt.grid()
    
    plt.subplot(1, 3, 3)
    plt.plot(u_list0.T)
    plt.title("u")
    plt.xlabel("steps")
    plt.grid()
    plt.ioff()
    plt.show()
""" 
    # start ILC iteration   
    for _ in range(episodes):
        
        conILC.stepILC()                        # update control
        q_new   = q0
        dq_new  = dq0
        pin.computeAllTerms(robModel,robData,q,dq)
        q_mem = []
        new_e=torch.zeros(dimE,1)
        
        for k in range(samples):
            
            new_u = conILC.getControl()         # get ILC control
            
            pin.computeAllTerms(robModel,robData,q_new,dq_new)
            
            iMmat:torch.Tensor  = torch.from_numpy(pin.computeMinverse(robModel,robData,q_new)).type(torch.float64)
            #Mmat:torch.Tensor   = torch.from_numpy(pin.crba(robModel,robData,q))
            Cmat:torch.Tensor   = torch.from_numpy(pin.computeCoriolisMatrix(robModel,robData,q_new,dq_new)).type(torch.float64)
            Gmat:torch.Tensor   = torch.from_numpy(pin.computeGeneralizedGravity(robModel,robData,q_new).reshape(-1,1)).type(torch.float64)
            
            new_u = new_u + torch.matmul(torch.eye(dimU,dtype=torch.float64),new_e.type(torch.float64))*0.1 + torch.tensor([[0.6,0.2]]).T.type(torch.float64)*0
            conILC.updateMemInput(new_u)        # save input for next episode (consider all inputs)
            
            
            ddq = torch.matmul(iMmat,torch.matmul(-Cmat,torch.from_numpy(dq_new.reshape(-1,1)).type(torch.float64))-Gmat-new_u)
            # NOTE implementare runge_kutta
            dq_new  = dq + ddq.numpy().squeeze()*dt
            q_new   = angle_normalize(q_new + dq_new*dt)
            
            out = torch.from_numpy(q_new.reshape(-1,1))             # simulate robot (simple function)
        
            new_e = (ref[:, k:k+1]-out)              # get new error
            conILC.updateMemError(angle_normalize(new_e.type(torch.float32)))        # save new_error
            
            mem = conILC.getMemory()
            q_mem.append(q_new)
            viz.display(q_new)
            
        q_list.append(q_mem)
        
    from matplotlib import pyplot as plt

    last_err = []
    for td in mem:
        err = td["error"]
        last_err.append(err[:, -1])

    
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
    
    print("finish") """