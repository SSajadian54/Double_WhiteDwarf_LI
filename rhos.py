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


################################################################################

def tickfun(x, start, dd0):
    return((start+x*dd0) )

################################################################################

G= 6.67430*pow(10.0,-11.0)
AU=1.495978707*pow(10.0,11)
Msun=1.989*pow(10.0,30.0)
Rsun =6.9634*pow(10.0,8.0)
KPC= 3.0857*pow(10.0,19.0)## meter 
velocity=299792458.0##m/s
nm=int(100)
tt=int(9)
v=np.zeros((tt))
nam=[r"$R_{\rm E}(0.01~R_{\odot})$", r"$\rho_{\star}$", r"$\rho_{\rm{l}}$"]
################################################################################
Dl=1.0*KPC
for i in range(50): 
    peri=float(1.0+i*1.0)##days
    period=float(peri*24.0*3600.0) #[s]
    mapp=np.zeros((nm, nm,3))
    for j in range(nm): 
        for k in range(nm): 
            Ms=float(0.17+(1.4-0.17)*j/nm)#Msun
            Ml=float(0.17+(1.4-0.17)*k/nm)#Msun
            Rl=float(0.01125*np.sqrt(np.power(Ml/1.454,-2.0/3.0)-np.power(Ml/1.454,2.0/3.0)))#[Rsun]
            Rs=float(0.01125*np.sqrt(np.power(Ms/1.454,-2.0/3.0)-np.power(Ms/1.454,2.0/3.0)))#[Rsun]
            semi=np.power(period*period*G*(Ml+Ms)*Msun/(4.0*np.pi*np.pi),1.0/3.0) #[m]
            RE=np.sqrt(4.0*G*Msun*Ml*semi)/velocity/Rsun
            rhos= float(Rs/RE)*(Dl/(Dl+semi))
            rhol= float(Rl/RE)
            print(RE, rhol, rhos)
            mapp[j,k,0]=RE/0.01
            mapp[j,k,1]=rhos
            mapp[j,k,2]=rhol
    ############################################################################
    for l in range(3): 
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
            v[m]=round(float(minn+m*step),2)
        cbar=plt.colorbar(orientation='vertical',shrink=0.95,pad=0.03, ticks=v)
        cbar.ax.tick_params(labelsize=17)
        cbar.set_label(str(nam[l]), rotation=270, fontsize=21,labelpad=20.0)
        contours=plt.contour(mapp[:,:,l],6,colors='k', linewidths =0.5,linestyles ='dashed')
        plt.clabel(contours, inline=5, fontsize=18)
        plt.title(r"$T(\rm{days})=$"+str(round(peri,1)) ,fontsize=22)    
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
        fig.tight_layout(pad=0.01)
        fig.savefig("./map{0:d}_{1:d}.jpg".format(l,i),dpi=200)     
    























