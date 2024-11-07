import numpy as np
import matplotlib as mpl
import pylab
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib import gridspec

rcParams["font.size"] = 18
rcParams["font.family"] = "sans-serif"
rcParams["font.sans-serif"] = ["Computer Modern Sans"]
rcParams["text.usetex"] = True
rcParams["text.latex.preamble"] = r"\usepackage{cmbright}"
from matplotlib.ticker import FormatStrFormatter
from matplotlib.gridspec import GridSpec
from scipy.signal import savgol_filter
from scipy.ndimage import uniform_filter1d
from scipy.interpolate import make_interp_spline
from scipy import interpolate
from scipy.ndimage.filters import gaussian_filter1d

################################################################################
def smooth(y, box_pts):
    box = np.ones(box_pts)/box_pts
    y_smooth = np.convolve(y, box, mode='valid')
    return (y_smooth)

def ErrorTESS(maga):
    emt=-1.0;     
    if(maga<7.5):       emt=float(0.22*maga-5.850); 
    elif(maga<12.5):    emt=float(0.27*maga-6.225);  
    else:               emt=float(0.31*maga-6.725);    
    #emt=emt+np.random.randn(0.0,0.1)
    if(emt<-5.0): emt=-5.0 
    emt=np.power(10.0,emt)
    if(emt<0.00001 or emt>0.5 or maga<0.0):
        print("Error emt:  ", emt, maga); 
    return(emt); 
################################################################################
nc=int(32+1) 
direc="./"
Num=int(3)
################################################################################

 
for ll in range(1): 
    f1=open(direc+"C_{0:d}.dat".format(Num),"r")
    nf= sum(1 for line in f1)  
    par=np.zeros((nf,nc))
    par=np.loadtxt(direc+"C_{0:d}.dat".format(Num)) 
    for i in range(nf):
        count,numl, nums, lat, lon       =par[i,0],par[i,1], par[i,2], par[i,3], par[i,4]
        Dist,mass,Teffs,Rstar,limbs      =par[i,5],par[i,6], par[i,7], par[i,8], par[i,9]
        Lumis,Loggs,Maps,Mabs,magb,fb,Ext=par[i,10],par[i,11],par[i,12],par[i,13],par[i,14],par[i,15],par[i,16]
        MBH, RBH, Lumil, Loggl, Teffl    =par[i,17],par[i,18], par[i,19],par[i,20],par[i,21]
        Mapl,Mabl,inc,teta,ecen          =par[i,22],par[i,23], par[i,24],par[i,25],par[i,26]
        period,lsemi,tp,ratio,cdpp,limbl =par[i,27],par[i,28], par[i,29],par[i,30],par[i,31],par[i,32]        
        a1=1.000/(1.0+ratio)
        a2=ratio/(1.0+ratio) 
        ########################################################################
        l1=-1;l2=-1
        o1=-1;o2=-1 
        count=int(count)
        nm=-1;  
        try:
            f2=open(direc+'M_{0:d}.dat'.format(count))
            nm=int(len(f2.readlines()))  
        except: 
            print("file does not exist", count)        
        if(nm>1 and count>=1000):  
            mod=np.zeros((nm,12));  
            ast=np.zeros((nm, 2)); 
            lef=np.zeros((nm, 3)); 
            mod=np.loadtxt(direc+'M_{0:d}.dat'.format(count)) 
            #tim,Astar,Astar2,finl,x1/l.a,y1/l.a,z1/l.a,l.RE/Rsun,s.ros,u/s.ros,disp/l.a, self    
            ##mod[:,2]=uniform_filter1d(mod[:,2],size=25)####<<<<Smoothing<<<<<<########
            umin=np.min(mod[:,9])
            
            for j in range(nm): 
                selfl=int(mod[j,11]) 
                if(selfl==1 and l1<0 ): l1=int(j)
                if(selfl==1 and l1>0 ): l2=int(j)
                if(selfl==2 and o1<0 ): o1=int(j)
                if(selfl==2 and o1>0 ): o2=int(j)
                if(selfl==1): 
                    ast[j,0]=fb*((mod[j,2]-mod[j,3])*a1+a2)+1.0-fb;#IRS_method 
                    ast[j,1]=fb*((mod[j,1]-mod[j,3])*a1+a2)+1.0-fb;#Valerio_model
                    lef[j,0]=fb*((1.0-mod[j,3])*a1+a2)+1.0-fb;#IRS_method, Occultation
                    lef[j,1]=fb*((mod[j,2]-0.0)*a1+a2)+1.0-fb;#IRS_method, Self_lensing
                    lef[j,2]=fb*((mod[j,1]-0.0)*a1+a2)+1.0-fb;#Valerio_method, Self_lensing
                elif(selfl==2): 
                    ast[j,0]=fb*((mod[j,2]-mod[j,3])*a2+a1)+1.0-fb;#IRS_method 
                    ast[j,1]=fb*((mod[j,1]-mod[j,3])*a2+a1)+1.0-fb;#Valerio_model   
                    lef[j,0]=fb*((1.0-mod[j,3])*a2+a1)+1.0-fb;#IRS_method, Occultation
                    lef[j,1]=fb*((mod[j,2]-0.0)*a2+a1)+1.0-fb;#IRS_method, Self_lensing
                    lef[j,2]=fb*((mod[j,1]-0.0)*a2+a1)+1.0-fb;#Valerio_method, Self_lensing
                else:
                    ast[j,0]=1.0;  ast[j,1]=1.0;  lef[j,0]=1.0;  lef[j,1]=1.0;  lef[j,2]=1.0 
            print("limit:  ", l1, l2, o1, o2, mod[:,9],  np.mean(mod[:,9])  )
            '''
            l1=l1+845
            l2=l2-845
            o1=o1+845
            o2=o2-845
            '''
            ####################################################################
            '''
            if(l1>=0 and l2>l1):
                plt.cla()
                plt.clf()
                fig=plt.figure(figsize=(8,6))
                ax1=fig.add_subplot(111)
                plt.plot(mod[l1:l2,0],lef[l1:l2,0],'r-.',label=r"$\rm{Finite}-\rm{Lens}$",lw=1.7)   
                plt.plot(mod[l1:l2,0],lef[l1:l2,2],'g--',label=r"$\rm{Self}-\rm{Lensing}(Valerio)$",lw=1.4)
                plt.plot(mod[l1:l2,0],ast[l1:l2,1],'k:', label=r"$\rm{Overall}-\rm{Flux}(Valerio)$",lw=1.4)
                plt.title(
                r"$M_{1}(M_{\odot})=$"+'{0:.1f}'.format(MBH)+
                r"$,~M_{2}(M_{\odot})=$"+'{0:.1f}'.format(mass)+
                r"$,~T(\rm{days})=$"+'{0:.1f}'.format(period)+
                r"$,~\log_{10}[\mathcal{F}]=$"+'{0:.2f}'.format(np.log10(ratio))+
                r"$,~i(\rm{deg})=$"+'{0:.1f}'.format(inc)+"\n"+
                r"$\epsilon=$"+'{0:.2f}'.format(ecen)+
                r"$,~\rho_{\star}=$"+'{0:.1f}'.format(ros1)+
                r"$,~u_{0}/\rho_{\star}=$"+'{0:.1f}'.format(imp1)+
                r"$,~R_{\rm{l}}/R_{\rm{E}}=$"+'{0:.2f}'.format(fin1),fontsize=15.0,color='k')
                #plt.xlim(np.min(mod[l1:l2,0]),np.max(mod[l1:l2,0]) )
                plt.xticks(fontsize=18, rotation=0)
                plt.yticks(fontsize=18, rotation=0)
                plt.xlabel(r"$\rm{time}(\rm{days})$", fontsize=18)
                plt.ylabel(r"$\rm{Normalized}~\rm{Flux}$",fontsize=19)
                plt.legend(prop={"size":15.},loc='best')
                fig=plt.gcf()
                fig.tight_layout()
                fig.savefig(direc+"SelfA{0:d}.jpg".format(count), dpi=200)
            ####################################################################    
            if(o1>=0 and o2>o1):       
                plt.cla()
                plt.clf()
                fig=plt.figure(figsize=(8,6))
                ax1=fig.add_subplot(111)
                plt.plot(mod[o1:o2,0],lef[o1:o2,0],'r-.',label=r"$\rm{Finite}-\rm{Lens}$",lw=1.7)   
                plt.plot(mod[o1:o2,0],lef[o1:o2,2],'g--',label=r"$\rm{Self}-\rm{Lensing}$",lw=1.4)
                plt.plot(mod[o1:o2,0],ast[o1:o2,1],'k:', label=r"$\rm{Overall}-\rm{Flux}$",lw=1.4)
                plt.title(
                r"$M_{1}(M_{\odot})=$"+'{0:.1f}'.format(MBH)+
                r"$,~M_{2}(M_{\odot})=$"+'{0:.1f}'.format(mass)+
                r"$,~T(\rm{days})=$"+'{0:.2f}'.format(period)+
                r"$,~\log_{10}[\mathcal{F}]=$"+'{0:.2f}'.format(np.log10(ratio))+
                r"$,~i(\rm{deg})=$"+'{0:.1f}'.format(inc)+"\n"+
                r"$\epsilon=$"+'{0:.2f}'.format(ecen)+
                r"$,~\rho_{\star}=$"+'{0:.1f}'.format(ros2)+
                r"$,~u_{0}/\rho_{\star}=$"+'{0:.1f}'.format(imp2)+
                r"$,~R_{\rm{l}}/R_{\rm{E}}=$"+'{0:.2f}'.format(fin2),fontsize=15.0,color='k')
                plt.xticks(fontsize=18, rotation=0)
                plt.yticks(fontsize=18, rotation=0)
                plt.xlabel(r"$\rm{time}(\rm{days})$", fontsize=18)
                plt.ylabel(r"$\rm{Normalized}~\rm{Flux}$",fontsize=19)
                plt.legend(prop={"size":15.},loc='best')
                fig=plt.gcf()
                fig.tight_layout()
                fig.savefig(direc+"SelfB{0:d}.jpg".format(count), dpi=200)
            '''
            ####################################################################
            if(o1>=0 and l1>=0):
                ros1= np.mean(mod[l1:l2,8])      
                imp1= np.min(mod[l1:l2,9])    
                fin1= RBH/np.mean(mod[l1:l2,7])    
                ros2= np.mean(mod[o1:o2,8])      
                imp2= np.min(mod[o1:o2,9])   
                fin2= Rstar/np.mean(mod[o1:o2,7]) 
                print("times: ", mod[l1,0], mod[l2,0], mod[o1,0],  mod[o2,0], l1, l2, o1, o2)
                nml=int(np.argmax(lef[l1:l2,2])+l1)
                nmo=int(np.argmax(lef[o1:o2,2])+o1)
                print (l1, l2, nml, o1, o2, nmo)
                #input("Enter a number ")
                sft=0.0
                if(mod[l2,0]<mod[o1,0]):
                    ros=np.array([ros1, ros2]);  imp= np.array([imp1,imp2]); fin=np.array([fin1, fin2])
                    lim=np.array([limbs,limbl])
                    sft=-mod[o1,0]+ mod[l2,0]; stat=1
                    tt=[int(l1+(l2-l1)*0.1),int(nml),int(l1+(l2-l1)*0.85),int(l2),
                        int(o1+(o2-o1)*0.15),int(nmo),int(o1+(o2-o1)*0.9)] 
                    
                        
                elif(mod[o2,0]<mod[l1,0]):        
                    ros=np.array([ros2,ros1]); imp=np.array([imp2,imp1]); fin=np.array([fin2,fin1]);
                    lim=np.array([limbl,limbs])            
                    sft=-mod[o2,0]+mod[l1,0]; stat=2
                    tt=[int(o1+(o2-o1)*0.1),int(nmo),int(o1+(o2-o1)*0.85),int(l1),
                        int(l1+(l2-l1)*0.15),int(nml),int(l1+(l2-l1)*0.9)]
                else:  
                    print(l1, l2, o1, o2, mod[l1,0], mod[l2,0], mod[o1,0], mod[o2,0]  )                   
                    #input("Error ")
                labs=[str(round(mod[tt[0],0],2)),str(round(mod[tt[1],0],2)),str(round(mod[tt[2],0],2)), str('...'),
                      str(round(mod[tt[4],0],2)),str(round(mod[tt[5],0],2)),str(round(mod[tt[6],0],2))]
                plt.clf()
                plt.cla()
                fig=plt.figure(figsize=(8,6))
                ax=fig.add_subplot(111)
                plt.plot(mod[l1:l2,0],lef[l1:l2,0],color='darkred',ls='-',lw=1.9)
                plt.plot(mod[l1:l2,0],lef[l1:l2,2],color='darkgreen',ls='--',lw=1.9)
                plt.plot(mod[l1:l2,0],ast[l1:l2,1],'k',ls='-.', lw=1.7)
                plt.plot(mod[o1:o2,0]+sft,lef[o1:o2,0],'r-',label=r"$\rm{Finite}-\rm{Lens}$",lw=1.9)   
                plt.plot(mod[o1:o2,0]+sft,lef[o1:o2,2],'g--',label=r"$\rm{Self}-\rm{Lensing}$",lw=1.9)
                plt.plot(mod[o1:o2,0]+sft,ast[o1:o2,1],'k-.', label=r"$\rm{Overall}-\rm{Flux}$",lw=1.5) 
                plt.title(
                r"$M_{1}(M_{\odot})=$"+'{0:.1f}'.format(MBH)+
                r"$;~M_{2}(M_{\odot})=$"+'{0:.1f}'.format(mass)+
                r"$;~T(\rm{days})=$"+'{0:.1f}'.format(period)+
                r"$;~\mathcal{F}=$"+'{0:.2f}'.format(ratio)+
                r"$;~i(\rm{deg})=$"+'{0:.2f}'.format(inc)+"\n"+
                r"$\epsilon=$"+'{0:.2f}'.format(ecen)+
                r"$;~\Gamma=$"+'{0:.1f},~{1:.1f}'.format(lim[0],lim[1])+
                r"$;~\rho_{\star}=$"+'{0:.1f},~{1:.1f}'.format(ros[0],ros[1])+
                r"$;~\rho_{\rm{l}}=$"+'{0:.2f},~{1:.2f}'.format(fin[0],fin[1])+
                r"$;~u_{0}/\rho_{\star}=$"+'{0:.1f},~{1:.1f}'.format(imp[0],imp[1]),fontsize=16.0,color='k')
                if(stat==1): 
                    plt.xlim(mod[l1,0], mod[o2,0]+sft)
                    ticc=np.array([mod[tt[0],0],mod[tt[1],0],mod[tt[2],0],mod[tt[3],0],mod[tt[4],0]+sft,mod[tt[5],0]+sft,mod[tt[6],0]+sft])
                if(stat==2):  
                    plt.xlim(mod[o1,0]+sft, mod[l2,0])
                    ticc=np.array([mod[tt[0],0]+sft,mod[tt[1],0]+sft,mod[tt[2],0]+sft,mod[tt[3],0],mod[tt[4],0],mod[tt[5],0], mod[tt[6],0]])
                ax.set_xticks(ticc,labels=labs)
                rcParams['xtick.major.pad']='-3.0'
                
                y_vals = ax.get_yticks()
                if(abs(np.max(y_vals)-np.min(y_vals))<0.0001): 
                    plt.ylim(1.0-0.0005, 1.0+0.0005)
                #ax1.set_yticklabels(['{:.2f}'.format(x * 100) for x in y_vals])
                
                plt.ylabel(r"$\rm{Normalized}~\rm{Flux}$",fontsize=20)
                plt.xlabel(r"$\rm{time}(\rm{days})$",fontsize=20)
                plt.xticks(fontsize=17, rotation=30)
                plt.yticks(fontsize=17, rotation=0)
                legend=ax.legend(prop={"size":16.5},loc='best',frameon=True, fancybox = True,shadow=True)
                legend.get_frame().set_facecolor('whitesmoke')
                fig=plt.gcf()
                fig.tight_layout()  
                fig.savefig(direc+"LightCurve{0:d}.jpg".format(count),dpi=200)
            ####################################################################
                print ("Light curve is plotted:  ", count )
                print ("HHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHH")






















