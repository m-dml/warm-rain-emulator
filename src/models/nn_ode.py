import numpy as np
import torch


class simulation_forecast:
     def __init__(self, model,start_point,d,end_point,means,sds,device="cuda:0"):
        self.start_point=start_point
        self.end_point=end_point
        self.inputs=(d[start_point:end_point,1:input_num+1]).to(device)
        self.model_params=d[self.start_point,5:8].to('cpu').numpy()*sds[4:7].reshape(1, -1) + means[4:7].reshape(1, -1)
        self.targets_orig=d[self.start_point+1:self.end_point+1,1:5]
        self.means=means
        self.sds=sds
        self.model=model
        self.device=device
        self.predictions_orig=[]
        self.new_input=None
        self.preds=None
     
     def test(self):
        with torch.no_grad():
            model.eval()
            self.new_input=self.inputs[0,:]
          
            self.preds=self.model(self.new_input.reshape(1,-1))
            self.moment_calc()
            
            for i in range (self.start_point+1,self.end_point):
                model.eval()
                self.new_input=self.new_input.astype(np.float32)
                self.new_input=(torch.from_numpy(self.new_input)).to(self.device)
                self.preds=self.model(self.new_input)
                self.new_input=self.new_input.reshape(input_num)
                self.moment_calc()
               
             
                
     def moment_calc(self):
        #self.new_input=(torch.from_numpy(self.new_input.reshape(input_num))[:]).to(device)
        self.update_calc()
       
        self.input_calc()
        
        predictions_orig_=self.new_input+(self.preds*20.0)
        
        self.check_predictions(predictions_orig_)
        
        #print(f"Predictions:{predictions_orig_}")
        
        
        
                
     def update_calc(self):
        self.preds=(torch.cat((self.preds[:,0:2],self.preds[:,2].reshape(-1,1),self.preds[:,3].reshape(-1,1)),1)).to('cpu').numpy()#Normalized
        
       
        self.preds= remove_norm(self.preds,input_num,self.means,self.sds)#Un-normalize
        self.preds=self.preds.reshape(-1,)
       
        self.preds=((min(self.preds[0],0)),(min(self.preds[1],0)),(max(self.preds[2],0)),(self.preds[3]))
        #self.preds=(self.preds[0],0),(self.preds[1]),(-self.preds[0]),(self.preds[2]))
  
        self.preds=np.asarray(self.preds).reshape(1,-1)
     
        
     def input_calc(self):
        
        self.new_input=(self.new_input[0:4]).to('cpu').numpy()
        #print(f"Input After:{self.new_input}")
        self.new_input= remove_norm(self.new_input,self.means,self.sds,0,4)
        
     def check_predictions(self,predictions_orig_):
        if predictions_orig_[:,3]<0:
            predictions_orig_=np.concatenate(((predictions_orig_[:,0:3].reshape(1,-1)),np.asarray([0]).reshape(1,-1)),1)
        if predictions_orig_[:,1]<0:
            predictions_orig_=np.concatenate(((predictions_orig_[:,0].reshape(1,-1)),np.asarray(self.predictions_orig[-1][:,1]).reshape(1,-1),predictions_orig_[:,2:].reshape(1,-1)),1)
        if predictions_orig_[:,0]<0:
            predictions_orig_=np.concatenate((np.asarray([0]).reshape(1,-1),predictions_orig_[:,1:].reshape(1,-1)),1)
            if predictions_orig_[:,2]>self.model_params[:,0]:
                predictions_orig_=np.concatenate(((predictions_orig_[:,0:2].reshape(1,-1)),self.model_params[:,0].reshape(1,-1),predictions_orig_[:,-1].reshape(1,-1)),1)
        self.predictions_orig.append(predictions_orig_)
        self.create_input(predictions_orig_)
        
     def create_input(self,predictions_orig_):
  
        tau=predictions_orig_[0,2]/self.model_params[:,0]#self.model_params[:,0]
        xc=predictions_orig_[0,0]/predictions_orig_[0,1]
        
        
        new_input_=np.concatenate((predictions_orig_[:,0:],self.model_params.reshape(1,-1),tau.reshape(1,-1),xc.reshape(1,-1)),axis=1)
        #new_input_=np.concatenate((predictions_orig_[:,0:],self.model_params.reshape(1,-1)),axis=1)
        #print(f"New Input(not normalized):{new_input_}")
        self.new_input=(new_input_ - means[:input_num].reshape(1, -1)) / sds[:input_num].reshape(1, -1)
         
      
       


    
    
def nn_simulation(d,means,sds,n):
    new_forecast=simulation_forecast(int(sim_info[n][0]),int(sim_info[n][1]),d,means,sds)
    new_forecast.test()
    predictions_orig=np.asarray(new_forecast.predictions_orig).reshape(-1,4)
    targets_orig=remove_norm(new_forecast.targets_orig.to('cpu').numpy(),means,sds,0,4)
    
    return predictions_orig,targets_orig,new_forecast.model_params


        
       
def remove_norm(arr,means,sds,start=0,last=None):
    norm_arr= (arr * sds[start:last].reshape(1, -1)) + means[start:last].reshape(1, -1)
    return norm_arr