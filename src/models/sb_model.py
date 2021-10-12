
import numpy as np


class SB_forecast:
    
    kcc = 9.44e9      # Long kernel in m3 kg-2 s-1 
    kcr = 5.78        # Long kernel in m3 kg-1 s-1 
    krr = 4.33

    xstar = 2.6e-10   # xstar in kg
    a_phi = 600.
    p_phi = 0.68
    rhow = 1e3
    
    def __init__(self, start_point,end_point,sds,means):
        self.start_point=start_point
        self.end_point=end_point
        
        self.model_params=d[self.start_point,5:8].to('cpu').numpy()*sds[4:7].reshape(1, -1) + means[4:7].reshape(1, -1)
        self.targets_orig=d[self.start_point+1:self.end_point+1,1:5]
        self.lo=d[self.start_point,5].to('cpu').numpy()*sds[4].reshape(1, -1) + means[4].reshape(1, -1)
        self.rm=d[self.start_point,6].to('cpu').numpy()*sds[5].reshape(1, -1) + means[5].reshape(1, -1)
        self.nu=d[self.start_point,7].to('cpu').numpy()*sds[6].reshape(1, -1) + means[6].reshape(1, -1)
        
        
        self.lc=((d[start_point,1]).to('cpu').numpy()*sds[0].reshape(1, -1)) + means[0].reshape(1, -1)
        self.nc=((d[start_point,2]).to('cpu').numpy()*sds[1].reshape(1, -1)) + means[1].reshape(1, -1)
        self.lr=((d[start_point,3]).to('cpu').numpy()*sds[2].reshape(1, -1)) + means[2].reshape(1, -1)
        self.nr=((d[start_point,4]).to('cpu').numpy()*sds[3].reshape(1, -1)) + means[3].reshape(1, -1)
        
        self.auto=None
        self.acc=None
        self.scc=None
        self.scr=None
        self.xc0=None
        

        self.predictions=[]
        
    
    
    def SB_calc(self):
        for i in range (self.start_point,self.end_point):
            self.autoconSB()
            self.accretionSB()
            self.selfcloudSB()
            self.selfrainSB()
            self.solve_ode()
            self.predictions.append([self.lc,self.nc,self.lr,self.nr])

        
    @staticmethod
    def drop_mass(r):
      
        return 4./3. *np.pi * SB_forecast.rhow * (r)**3

    def autoconSB(self):
    
        nu=self.nu
        xc=self.lc/self.nc
      
        auto = SB_forecast.kcc/(20*SB_forecast.xstar) * (nu+2.0)*(nu+4.0)/(nu+1.0)**2 * self.lc**2 * xc**2
        tau  = self.lr/(self.lc+self.lr+1e-15)
        taup = np.power(tau,SB_forecast.p_phi)
        phi  = SB_forecast.a_phi * taup * (1.0 - taup)**3
        self.auto = auto * (1.0+phi)
        
    

    def accretionSB(self):
  
        tau = self.lr/(self.lc+self.lr)
        phi = (tau / (tau + 5e-4))**4
        self.acc = SB_forecast.kcr * self.lc * self.lr * phi
        

    def selfcloudSB(self):
     
        self.scc= SB_forecast.kcc * (self.nu+2.0)/(self.nu+1.0) * self.lc**2
     

    def selfrainSB(self):
 
        self.scr = SB_forecast.krr * self.lr * self.nr
      

    def solve_ode(self):
        dt=5
        xm=SB_forecast.drop_mass(self.rm)
        autoN = 1.0/SB_forecast.xstar*self.auto
        accrN = self.acc/xm
        for i in range (4):
            self.lc = self.lc - self.auto*dt - self.acc*dt
            self.lr = self.lr + self.auto*dt + self.acc*dt
            self.nc = self.nc - accrN*dt - self.scc*dt
            self.nr = self.nr + autoN*dt - self.scr*dt

             
                
def sb_simulation(d,sds,means,sim_info,n):
   
    new_forecast=SB_forecast(int(sim_info[n][0]),int(sim_info[n][1]),sds,means)
    new_forecast.SB_calc()
    predictions_orig=np.asarray(new_forecast.predictions).reshape(-1,4)
    targets_orig=remove_norm(new_forecast.targets_orig.to('cpu').numpy(),sds,means,0,4)
    
    return predictions_orig,targets_orig,new_forecast.model_params


def remove_norm(arr,sds,means,start=0,last=None):
    norm_arr= (arr * sds[start:last].reshape(1, -1)) + means[start:last].reshape(1, -1)
    return norm_arr
    