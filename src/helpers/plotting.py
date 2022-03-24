import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
from matplotlib import ticker  


#The main plotting function
def plot_simulation(predictions_orig,targets_orig,sb_preds,var_all,model_params,num):
    t_len=sb_preds.shape[0]
    targets_orig=targets_orig[:t_len,:]
    predictions_orig=predictions_orig[:t_len,:]
    #print(sb_preds.shape)
    sns.set_style("darkgrid", {"grid.color": ".6", "grid.linestyle": ":"})
    var=['Lc','Nc','Lr','Nr']
    color=["#26235b","#bc473a","#812878","#f69824"]
    fig= plt.figure()
    fig.set_size_inches(13, 10)
    #time= [x for x in range(0, len(predictions_orig))]
    time=[x for x in range(0, len(sb_preds))]
    #color=iter(cm.rainbow(np.linspace(0,1,4)))
    for i in range(4):
        ax = fig.add_subplot(2,2, i + 1)
        #c=next(color)
        plt.plot(time[:],predictions_orig[:,i],c=color[i])
        plt.plot(time[:],targets_orig[:,i],c='black')
        plt.plot(time[:],sb_preds[:,i],c=color[i],linestyle='dashed')
        plt.fill_between(time[:], targets_orig[:,i]-var_all[:len(time),i], targets_orig[:,i]+var_all[:len(time),i],facecolor = "gray")
        #plt.plot(time[:], targets_orig[:,i]-var_all[:len(time),i])
        plt.title(var[i])
        plt.xlabel('Timestep')    

        plt.legend(['Neural Network','Simulations'])
    
    
    fig.suptitle("Lo: %.4f; rm:%.6f ; Nu: %.1f : Stepsize 5, ode constrained, training un-constrained, low lr"%((model_params[0]),(model_params[1]),(model_params[2])), fontsize="x-large")

    return fig #Figure needs to be plotted in tensorboard
#    plt.show()
#     fig.savefig("/gpfs/home/sharmas/warm-rain-emulator/Notebooks/foo.png", dpi=300) 


def calc_errors(all_arr,n):

    var_1=np.std(all_arr[:,1,n,:],axis=-1)
    var_2=np.std(all_arr[:,2,n,:],axis=-1)
    var_3=np.std(all_arr[:,4,n,:],axis=-1)
    var_4=np.std(all_arr[:,5,n,:],axis=-1)
    var_all=np.concatenate((var_1.reshape(1,-1),var_2.reshape(1,-1),var_3.reshape(1,-1),var_4.reshape(1,-1)),axis=0)
    return var_all