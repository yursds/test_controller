
from matplotlib     import pyplot as plt

def plotILC(episodes,e_list,de_list):
           
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

    plt.ioff()
    plt.show()