import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
from matplotlib import ticker
from scipy.stats import spearmanr
from sklearn.metrics import mean_absolute_error


def plot_train(train_loss,val_loss,start=0,last=-1):
    sns.set_theme(style="darkgrid")
    x=np.arange(1,len(val_loss)+1)
    fig, ax1 = plt.subplots()
    fig.set_size_inches(13, 10)
    plt.plot(x[start:last],train_loss[start:last])
    plt.plot(x[start:last],val_loss[start:last])
    plt.legend(["Training Loss","Validation Loss"])

def plot_predictions(targets,predictions,x,y,var,main_title,tick=4,fig_size=[13,10]):
    sns.set_theme(style="darkgrid")
    n=targets.shape[-1]
    fig= plt.figure()
    fig.set_size_inches(fig_size)
    color=iter(cm.rainbow(np.linspace(0,1,n)))
    for i in range(n):
        ax = fig.add_subplot(x,y, i + 1)
        c=next(color)
       
        sns.regplot(x=targets[:,i],y=predictions[:,i],color=c,label=var[i])
        corr, _ = spearmanr(predictions[:,i], targets[:,i])
        plt.xlabel('Targets')    
        plt.ylabel('Predictions_nn')
        plt.legend()
        plt.title(f" R: {corr}")
        ax.xaxis.set_major_locator(ticker.MaxNLocator(tick))   
    fig.suptitle(main_title, fontsize="x-large")

def plot_predictions_mm(targets,predictions,x,y,var,main_title,tick=4,fig_size=[13,10]):
    sns.set_theme(style="darkgrid")
    n=targets.shape[-1]
    fig= plt.figure()
    fig.set_size_inches(fig_size)
    color=iter(cm.rainbow(np.linspace(0,1,n)))
    for i in range(n):
        ax = fig.add_subplot(x,y, i + 1)
        c=next(color)
        sns.regplot(x=targets[:,i],y=predictions[:,i],color=c,label=var[i])
        
        mae= mean_absolute_error(predictions[:,i], targets[:,i])
        #corr, _ = spearmanr(predictions[:,i], targets[:,i])
        plt.xlabel('Targets')    
        plt.ylabel('Predictions_nn')
        plt.legend()
        plt.title(f" MAE: {mae}")
        ax.xaxis.set_major_locator(ticker.MaxNLocator(tick))   
    fig.suptitle(main_title, fontsize="x-large")


def plot_residuals(residuals,plot_title,last,start=0):
    sns.set_theme(style="darkgrid")
    t=[]
    series_=[]
    fig, axs = plt.subplots(1, 1)
    fig.set_size_inches(13, 10)

    for i in range (start,last):
        if residuals[i,0]<residuals[i+1,0]:
            t.append(residuals[i,0])
            series_.append(residuals[i,1])
        
        else:
            t.append(residuals[i,0])
            series_.append(residuals[i,1])
            plt.plot(t,series_)


            t=[]
            series_=[]
            
    if i==last-1:
        plt.plot(t,series_)
            

    plt.title(plot_title)
    plt.xlabel("Time")