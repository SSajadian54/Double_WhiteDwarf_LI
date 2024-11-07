import numpy as np 
from numpy import conj
import matplotlib.pyplot as plt
import matplotlib
import pylab as py 
from matplotlib import rcParams
from matplotlib.ticker import FormatStrFormatter
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter, AutoMinorLocator)
rcParams["font.size"] = 11.5
rcParams["font.family"] = "sans-serif"
rcParams["font.sans-serif"] = ["Computer Modern Sans"]
rcParams["text.usetex"] = True
rcParams["text.latex.preamble"] = r"\usepackage{cmbright}"
rcParams['text.usetex'] = True
matplotlib.rc('text', usetex=True)
rcParams["text.latex.preamble"].join([r"\usepackage{dashbox}",r"\setmainfont{xcolor}",])
cmap=plt.get_cmap('viridis')
import scipy.special as ss
import warnings
warnings.filterwarnings("ignore")
cmap=plt.get_cmap('viridis')


addres="./BOmaps4/"
u0=3.5
################################################################################

def tickfun(x, start, dd0):
    return((start+x*dd0) )


def fmts(x):
    s = f"{x:.1f}"
    return (str(r"$\rho_{\star}=$")+rf"{s}" if plt.rcParams["text.usetex"] else f"{s} %")


def fmtl(x):
    s = f"{x:.1f}"
    return (str(r"$\rho_{\rm{l}}/\rho_{\rm{out}}=$")+rf"{s}" if plt.rcParams["text.usetex"] else f"{s} %")

def fmtl2(x):
    s = f"{x:.1f}"
    return (str(r"$\rho_{\rm{l}}/\rho_{\rm{in}}=$")+rf"{s}" if plt.rcParams["text.usetex"] else f"{s} %")


def fmtt(x):
    s = f"{x:.1f}"
    return (str(r"$\rho_{\rm{l}}\rho_{\star}=$")+rf"{s}" if plt.rcParams["text.usetex"] else f"{s} %")


################################################################################

G= 6.67430*pow(10.0,-11.0)
AU=1.495978707*pow(10.0,11)
Msun=1.989*pow(10.0,30.0)
Rsun =6.9634*pow(10.0,8.0)
KPC= 3.0857*pow(10.0,19.0)## meter 
velocity=299792458.0##m/s
nm=int(70)
tt=int(9)
v=np.zeros((tt))
nam=[r"$A$", r"$A_{\rm{IRS}}$", r"$f_{\mathcal{O}}$", r"$A-\mathcal{O}-1$"]

################################################################################
'''
rho=np.arange(0.1,10.0,0.001)
plt.cla()
plt.clf()
fig=plt.figure(figsize=(8,6))
ax1=fig.add_subplot(111)
plt.plot(rho, np.abs(rho+np.sqrt(rho**2.0+4.0))*0.5,"-",  color='k', lw=1.9, label=r"$\rho_{\rm{out}}$")
plt.plot(rho, np.abs(rho-np.sqrt(rho**2.0+4.0))*0.5,"--",  color='b', lw=1.9, label=r"$\rho_{\rm{in}}$")
plt.xlabel(r"$\rho_{\star}$",    fontsize=18)
plt.ylabel(r"$\rm{Images}~\rm{Ring}~\rm{radius}$", fontsize=18)
plt.xticks(fontsize=19, rotation=0)
plt.yticks(fontsize=19, rotation=0)
ax1.legend(prop={"size":17})
ax1.grid("True")
ax1.grid(linestyle='dashed')
fig=plt.gcf()
plt.subplots_adjust(hspace=.0)
fig.savefig("./RinRout.jpg",dpi=200)
input("Enter a number ")
'''
################################################################################
Dl=1.0*KPC
for ii in range(50):
    i=ii
    peri=  float(1.0+i)*1.0##days
    period=float(peri*24.0*3600.0) #[s]
    mapp=np.zeros((nm, nm,8))
    par=np.zeros((nm*nm, 14))
    par=np.loadtxt(addres+"Bmap{0:d}.dat".format(int(peri)))
    cou=0;
    for k in range(nm):## Mstar
        for j in range(nm):## Mlens
            mapp[k,j,0]=par[cou,11]
            mapp[k,j,1]=par[cou,12]
            mapp[k,j,2]=par[cou,13]/par[cou,11]
            mapp[k,j,3]=par[cou,11]-par[cou,13]-1.0
            mapp[k,j,4]=par[cou,4]#rhoS
            mapp[k,j,5]=par[cou,7]/par[cou,9]#RhoL/Rout
            mapp[k,j,6]=par[cou,4]*par[cou,7]#RhoL*Rhos
            mapp[k,j,7]=par[cou,7]/par[cou,8]#RhoL/Rin
            cou+=1
    #in,jn,p.Mstar,p.Rstar,p.Rhos,p.Mlens,p.Rlens,p.Rhol,p.Rin,p.Rout,p.RE/(Rsun*0.01),p.Astar0,p.Astar1,p.occult  
    ############################################################################
    for l in range(4): 
        plt.cla()
        plt.clf()
        fig=plt.figure(figsize=(7,6))
        ax=plt.gca()  
        plt.imshow(mapp[:,:,l],cmap=cmap,interpolation='nearest',aspect='equal', origin='lower')
        plt.clim()
        minn=np.min(mapp[:,:,l])
        maxx=np.max(mapp[:,:,l])
        step=float((maxx-minn)/(tt-1.0));
        
        for m in range(tt):
            v[m]=round(float(minn+m*step),1)
        
        cbar=plt.colorbar(orientation='vertical',shrink=0.9,pad=0.01, ticks=v)
        cbar.ax.tick_params(labelsize=17)
        cbar.set_label(str(nam[l]), rotation=270, fontsize=21,labelpad=20.0)
        if(l==0 or l==1): 
            contours=plt.contour(mapp[:,:,4],levels=[0.2,0.3, 0.8,1.5,3.0,4.0],colors='k', linewidths =0.7,linestyles ='dashed')
            plt.clabel(contours, inline=5,fmt=fmts, fontsize=19)
            
            
        #if(l==2): 
        #    contours=plt.contour(mapp[:,:,5],levels=[0.3, 1.0,2.0, 3.0],colors='k', linewidths =0.6,linestyles ='dashed')
        #    plt.clabel(contours, inline=5,fmt=fmtl, fontsize=19) 
            
        #   contours=plt.contour(mapp[:,:,7],levels=[1, 2, 3],colors='r', linewidths =0.6,linestyles ='dashdot')
        #    plt.clabel(contours, inline=10,fmt=fmtl2, fontsize=19) 
            
        if(l==3): 
            contours=plt.contour(mapp[:,:,6],levels=[0.1,0.3,1.0,3.0,6.0],colors='k', linewidths =0.7,linestyles ='dashed')
            plt.clabel(contours, inline=5,fmt=fmtt, fontsize=19)      
            
        plt.title(r"$T(\rm{days})=$"+str(round(peri,1))+r"$,~u_{0}/\rho_{\star}=$"+str(u0),fontsize=22, pad=-0.05)    
        plt.xticks(fontsize=21, rotation=0)
        plt.yticks(fontsize=21, rotation=0)
        plt.xlim(0.0,nm)
        plt.ylim(0.0,nm)
        ticc=np.array([ int(nm*0.1), int(nm*0.3), int(nm*0.5), int(nm*0.7), int(nm*0.9) ])
        ax.set_xticks(ticc,labels=[round(j,1) for j in tickfun(ticc,float(0.17),float(1.4-0.17)/nm)])
        ax.set_yticks(ticc,labels=[round(j,1) for j in tickfun(ticc,float(0.17),float(1.4-0.17)/nm)])
        ax.set_aspect('equal', adjustable='box')
        plt.xlabel(r"$M_{\rm{l}}(M_{\odot})$",fontsize=21,labelpad=0.05)
        plt.ylabel(r"$M_{\star}(M_{\odot})$",fontsize=21,labelpad=0.05)
        fig=plt.gcf()
        fig.tight_layout(pad=0.1)
        fig.savefig(addres+"mapm{0:d}_{1:d}.jpg".format(i,l),dpi=200)     
    























