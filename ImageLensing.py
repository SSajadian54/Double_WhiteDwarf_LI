import numpy as np 
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
import VBBinaryLensingLibrary as vb
vbb =vb.VBBinaryLensing()
vbb.Tol=1.0e-5;
vbb.SetLDprofile(vbb.LDlinear);
vbb.LoadESPLTable("./ESPL.tbl"); 
from numba import jit, njit, prange
direc="./"

################################################################################
G= 6.67430*pow(10.0,-11.0)
AU=1.495978707*pow(10.0,11)
Msun=1.989*pow(10.0,30.0)
Rsun =6.9634*pow(10.0,8.0)
Kpc= 3.0857*pow(10.0,19.0)## meter 
velocity=299792458.0##m/s
thre=float(0.001); 
epsi=float(0.0005)
################################################################################
Nb=int(1000) 
Nm=int(49001/1.3)
nx=int(4091)
ny=int(4091)
tt=int(7)
sini=  np.zeros((nx,ny))
pro=   np.zeros((nx,ny))
phi=   np.zeros((Nm));  
ksi=   np.zeros((Nm));  
x1=    np.zeros((Nm));  
x0=    np.zeros((Nm));
y1=    np.zeros((Nm));  
y0=    np.zeros((Nm));
z1=    np.zeros((Nm));  
AstarV=np.zeros((Nm));
AstarI=np.zeros((Nm));
Fintl= np.zeros((Nm));
Flux=  np.zeros((Nm));
dis=   np.zeros((Nm));
disp=  np.zeros((Nm));
tim=   np.zeros((Nm));
v=     np.zeros((tt));
################################################################################

def tickfun(x, start, dd0, rho):
    return((start+x*dd0)/rho)
    
#=====================================

def Thirdlaw(MSum, period):
    return(np.power(G*MSum*period*period*24.0*24.0*3600.0*3600.0/(4.0*np.pi*np.pi),1.0/3.0))
#=====================================

@jit
def Kepler(phi, ecen):
    phi=phi*180.0/np.pi
    for kk in range(len(phi)): 
        while(phi[kk]>360.0):
            phi[kk]=phi[kk]-360.0  
        while(phi[kk]<0.0):
            phi[kk]=phi[kk]+360.0       
        if(phi[kk]>180):  phi[kk]=float(phi[kk]-360)
        if(phi[kk]<-181.0 or phi[kk]>181.0):  
            print("Phi:  ",  phi[kk], ecen[kk])
            input("Enter a number ")
    phi=phi*np.pi/180.0##radian 
    ksi=phi; 
    for iw in range(Nb):
        term=2.0/(iw+1.0)*ss.jv(int(iw+1),(iw+1.0)*ecen)*np.sin((iw+1)*phi)
        if(iw==0):   term0=np.abs(term)
        ksi+=term
        if(np.mean(np.abs(term))<np.mean(abs(thre)*term0) and iw>5):  
            break 
    return(ksi) 
#=====================================

#@jit
def Fluxs(rho, xs0, ys0, sini, LimB):
    for i in range(nx):  
        for j in range(ny):
            sini[j,i]=0.0;   
            xi=float(i-nx/2.0)*dx#[RE]
            yi=float(j-ny/2.0)*dy#[RE]
            dis=np.sqrt((xi-xs0)**2.0+(yi-ys0)**2.0)
            if(dis<rho or dis==rho):               
                mu=np.sqrt(1.0-dis**2.0/rho**2.0) 
                sini[j,i]=1.0*abs(1.0-LimB*abs(1.0-mu))
            else: 
                sini[j,i]=0.0;               
    return(sini); 
#=====================================

#@jit
def LensEq(xi, yi, xlens, ylens):#[RE, RE, RE, RE] 
    xm=float(xi-xlens) 
    ym=float(yi-ylens) 
    d2=xm**2.0+ ym**2.0 +1.0e-50
    xs=float(xi-xm/d2)
    ys=float(yi-ym/d2)
    return(xs, ys)
#=====================================

def FluxSelf(As, Fl, Flx, n1, n2,H, Self):
    plt.cla()
    plt.clf()
    fig=plt.figure(figsize=(8,6))
    plt.plot(tim[n1:n2]*period, As[n1:n2] ,'g--',lw=1.5,label=r"$\rm{Self}-\rm{Lensing}$")
    plt.plot(tim[n1:n2]*period, Fl[n1:n2] ,'r-.',lw=1.5,label=r"$\rm{Finite}-\rm{Lens}~\rm{effect}$")
    plt.plot(tim[n1:n2]*period,Flx[n1:n2],'k:', lw=1.5,label=r"$\rm{Overall}~\rm{flux}$")
    plt.xlabel(r"$\rm{time}(\rm{days})$", fontsize=18)
    plt.ylabel(r"$\rm{Normalized}~\rm{Flux}$", fontsize=18)
    plt.xlim([tim[n1]*period , tim[n2]*period ])
    plt.xticks(fontsize=18, rotation=0)
    plt.yticks(fontsize=18, rotation=0)
    plt.legend()
    plt.legend(prop={"size":16.0}, loc='best')
    fig=plt.gcf()
    fig.tight_layout()
    plt.savefig(direc+"LightSelf{0:d}_{1:d}.jpg".format(h,int(Self)),dpi=200)

#=====================================

def Fluxtot(Flx,H):
    plt.cla()
    plt.clf()
    fig=plt.figure(figsize=(8,6))
    plt.plot(tim*period,Flx,'k:',lw=2.0,label=r"$\rm{Overall}~\rm{flux}$")
    plt.xlabel(r"$\rm{time}(\rm{days})$", fontsize=18)
    plt.ylabel(r"$\rm{Normalized}~\rm{Flux}$", fontsize=18)
    plt.xlim([tim[0]*period,tim[Nm-1]*period])
    plt.xticks(fontsize=18, rotation=0)
    plt.yticks(fontsize=18, rotation=0)
    plt.legend()
    plt.legend(prop={"size":16.0}, loc='best')
    fig=plt.gcf()
    fig.tight_layout()
    plt.savefig(direc+"Lighttot{0:d}.jpg".format(H),dpi=200) 
    
#=====================================

def MapI(Xsc, Ysc, Rho, Re, Rlens, Pro, Num, H, RlRE, uRo, RlRin, As, Fin, timt, fluxt):
    plt.cla()   
    plt.clf()
    fig=plt.figure(figsize=(8,6))
    ax=fig.add_subplot(111)
    circle1=plt.Circle((Xsc/dx+nx/2, Ysc/dy+ny/2), float(Rho/dx),fill = False,color='w',lw=2.4, linestyle='-')
    circle2=plt.Circle((nx/2,ny/2),float(Rlens/Re/dx),fill = False,color='k', lw=1.9, linestyle='-')
    circle3=plt.Circle((nx/2,ny/2),float(Re/Re/dx),fill = False, color='r', lw=1.9, linestyle='--')
    plt.imshow(Pro,cmap=cmap,interpolation='nearest',aspect='equal', origin='lower')
    ax.add_artist(circle1)
    ax.add_artist(circle2)
    ax.add_artist(circle3)
    plt.clim()
    minn=np.min(Pro)
    maxx=np.max(Pro)
    step=float((maxx-minn)/(tt-1.0));
    for m in range(tt):
        v[m]=round(float(minn+m*step),1)
    cbar=plt.colorbar(orientation='vertical',shrink=0.95,pad=0.03,ticks=v)
    cbar.ax.tick_params(labelsize=18)
    plt.clim(v[0]+0.005*step,v[tt-1]-0.005*step)
    plt.xticks(fontsize=19, rotation=0)
    plt.yticks(fontsize=19, rotation=0)
    plt.xlim(300.0+nx*0.0,nx*1.0-300)
    plt.ylim(300.0+ny*0.0,ny*1.0-300)
    ticc=np.array([ int(nx*0.15), int(nx*0.35), int(nx*0.55), int(nx*0.75), int(nx*0.9) ])
    ax.set_xticks(ticc,labels=[round(j,1) for j in tickfun(ticc,float(-xsiz*0.5),dx,Rho) ])
    ax.set_yticks(ticc,labels=[round(j,1) for j in tickfun(ticc,float(-ysiz*0.5),dy,Rho) ])
    ax.set_aspect('equal', adjustable='box')
    plt.title(
    r"$\rm{Time}(\rm{days})=$"+str(round(timt,2))+
    r"$,~u/\rho_{\star}=$"+str(round(uRo,1))+
    r"$,~\rho_{\rm{l}}=$"+str(round(RlRE,1))+
    r"$,~A=$"+str(round(As,2))+r"$,~\mathcal{O}=$"+str(round(Fin,2)),color='k',fontsize=17)
    plt.xlabel(r"$x-\rm{axis}[R_{\star,~\rm{p}}]$",fontsize=23,labelpad=0.05)
    plt.ylabel(r"$y-\rm{axis}[R_{\star,~\rm{p}}]$",fontsize=23,labelpad=0.05)
    fig=plt.gcf()
    fig.tight_layout(pad=0.03)
    fig.savefig(direc+"map{0:d}_{1:d}.jpg".format(H,Num), dpi=200)     
    print ("One map is plotted ")

#===============================================================================

def LensEq2(xm, ym, Dl, Ds, Re, Mlens, Rlens):
    b=np.sqrt(xm**2.0+ym**2.0)+1.0e-50#[m]
    angle=float(4.0*G*Mlens/(velocity*velocity*b))#radian  
    tant= float(b/Dl)#tan(theta)
    tana= float(np.tan(angle))#tan(alpha)
    tanb= tant-float(tant + (tana-tant)/(1.0+ tant*tana))*np.abs(Ds-Dl)/Ds
    beta= float(tanb*Dl)#[m]
    xs=   float(beta*xm/b)/Re#[RE]
    ys=   float(beta*ym/b)/Re#[RE]
    d2=   b*b/(Re*Re)
    xs0=  float(xm/Re-xm/Re/d2)
    ys0=  float(ym/Re-ym/Re/d2)
    if(float(angle*180.0/np.pi)>10.0 or abs(xs-xs0)>10.0 or abs(ys-ys0)>10.0): 
        print("deflection angle: ", angle*180.0/np.pi)
        print("new ccordinate : ",  xs,    ys)
        print("old coordinate:  ",  xs0,  ys0)
    if(b<Rlens or b==Rlens):flag=0;
    else:                   flag=1;
    return(xs, ys, flag)
#=====================================
@jit   
def InRayShoot(Dl, Ds, Re, Mlens, Rlens, pro, sini):
    fintl=0.0; 
    for i in range(nx):
        for j in range(ny):
            pro[j,i]=0.0
            xi=float(i-nx/2.0)*dx*Re
            yi=float(j-ny/2.0)*dy*Re
            xsi,ysi,Flag=LensEq2(xi, yi, Dl, Ds, Re, Mlens, Rlens)
            px=int(round((xsi+xsiz*0.5)/dx,2))           
            py=int(round((ysi+ysiz*0.5)/dy,2))
            if(px>=0 and px<nx and py>=0 and py<ny):
                pro[j,i]+=sini[py,px]
                if(Flag<1): fintl+=sini[py,px]
    return(pro,fintl)         
################################################################################
for h in range(1):  
    fil0=open(direc+"param.dat","w")
    fil0.close()
    fil1=open(direc+"light{0:d}.dat".format(h),"w")
    fil1.close();
    
    param=np.array([295,676 ,231 ,-11.71890 ,178.86630, 0.088183,0.561000 , 7657.00 , 0.013200,  0.532,  0.0005 , 7.9450 , 17.5835  , 12.7993 ,17.5835 , 1.0000 , 0.0572  ,0.221000 , 0.021510,  0.000754 , 7.11600  ,6521.00000 , 17.00831 , 12.22416 , 0.15490 , 252.04887,  0.731799,  16.6216380,  3.28134314 , 8.834728 , 0.58874968  ,31042.73670 ,  0.54800])#33
    #count,l.num, s.num,  s.lat, s.lon, //5
    #Dist1, s.mass, s.Teff, s.Rstar, s.limb, s.Lumi, s.Logg, s.Map, s.Mab, s.magb, s.blend, s.Ai,//17
    #l.MBH, l.RBH, l.Lumi, l.Logg, l.Teff,l.Map,l.Mab,//24
    #l.inc*RA, l.tet*RA, l.ecen, l.period, log10(l.a/(s.Rstar*Rsun)), l.tp, l.ratio, s.cdp, limb
    Mstar= param[6]*Msun#[kg]   161
    Rstar= param[8]*Rsun#[m]
    MBH=   param[17]*Msun#[kg]
    RBH=   param[18]*Rsun;#[m]
    period=param[27]## days
    inc=   param[24] #degree
    teta=  param[25]#degree
    ecen=  param[26]
    Dist=  param[5]
    limbs= param[9]
    limbl= param[32]
    tp=    param[29]/period;
    alfa=  param[30];
    A1= 1.0/(1.0+alfa)
    A2=alfa/(1.0+alfa)
    
    for i in range(Nm):  
        tim[i]=float(0.0+1.0*i/(Nm-1.0)/1.0) #[-0.5,0.5]
        phi[i]=(tim[i]-tp)*2.0*np.pi 
    inc= float(inc*np.pi/180.0)
    teta=float(teta*np.pi/180.0)
    a=Thirdlaw(float(MBH+Mstar),period)#[m]
    print("Semi_major axis[AU]", a/AU)
    fil0=open(direc+"param.dat","a+")
    par=np.array([h,Mstar/Msun,Rstar/Rsun,MBH/Msun,RBH/Rsun,period, inc, teta, ecen, Dist, limbl, limbs, tp, alfa, a/AU])
    np.savetxt(fil0,par.reshape((-1,15)),fmt ="%d %.4f %.4f %.4f %.4f  %.4f  %.4f  %.4f  %.4f %.4f %.4f %.4f %.4f %.4f %.4f")
    fil0.close();
    ############################################################################
    if(ecen<0.01): ksi=phi
    else:          ksi=Kepler(phi, ecen)
    x0=a*(np.cos(ksi)-ecen)#[m]
    y0=a*np.sin(ksi)*np.sqrt(1.0-ecen**2.0)#[m]
    y1=                y0*np.cos(teta)+x0*np.sin(teta)#[m]
    x1=  np.cos(inc)*(-y0*np.sin(teta)+x0*np.cos(teta))#[m]
    z1= -np.sin(inc)*(-y0*np.sin(teta)+x0*np.cos(teta))#[m] 
    dis= np.sqrt(x1**2.0 + y1**2.0 + z1**2.0)+1.0e-50#[m]
    disp=np.sqrt(y1**2.0 + z1**2.0)+1.0e-50;#[m]
    Re=np.sqrt(4.0*G*(MBH+Mstar)*0.5*a)/velocity+1.0e-50;#meter 
    rho=float((RBH+Rstar)*0.5/Re);
    xsiz=float(13.10*rho)#[RE]
    ysiz=float(13.10*rho)#[RE]
    dx=float(xsiz/(nx-1.0))#[RE]
    dy=float(ysiz/(ny-1.0))#[RE]
    Fbase=np.pi*rho*rho*(1.0-limbs/3.0)/(dx*dy);
    s1=0; s2=0; o1=0; o2=0
    Self=0;  
    #=====================================
    nsi=0
    for k in range(Nm): 
        AstarV[k]=1.0;
        AstarI[k]=1.0;    
        Fintl[k]=0.0;  
        Self=0; 
        if(x1[k]<0.0): 
            Rlens=float(RBH)#meter
            Mlens=float(MBH)#kg
            Dl=float(Dist*Kpc);
            Ds=float(Dl+np.abs(x1[k]) );
            Dls=Ds-Dl;
            proj=float(Dl/Ds);
            RE=np.sqrt(4.0*G*Mlens)*np.sqrt(Dls*proj)/velocity+1.0e-50;
            ros=np.abs(Rstar*proj/RE) 
            SourceA=float(np.pi*ros*ros*(1.0-limbs/3.0)/(dx*dy));
            u=float(disp[k]/RE);
            xsc=float(y1[k]/RE);  
            ysc=float(z1[k]/RE);
            AstarV[k]=1.0; AstarI[k]=1.0; Fintl[k]=0.0; 
            Rin= np.abs(u-np.sqrt(u*u+4.0))*0.5#[RE]
            if(u<float(20.0*ros)):
                if(s1==0):  s1=k
                if(s1>0):   s2=k 
                Self=1;
                if(ros>100.0):
                    if(u<ros):  AstarV[k]=float(1.0+2.0/ros/ros);      
                    else:       AstarV[k]=float(u*u+2.0)/sqrt(u*u*(u*u+4.0));
                else:  
                    vbb.a1=limbs; 
                    AstarV[k]=vbb.ESPLMag2(u, ros);   
                if(u<float(5.3*ros)):
                    sini= Fluxs(ros, xsc, ysc, sini,limbs)
                    Fbase=float(np.pi*ros*ros*(1.0-limbs/3.0)/(dx*dy));
                    pro,Fintl[k]=InRayShoot(Dl, Ds, RE, Mlens, Rlens, pro, sini)
                    AstarI[k]=float(np.sum(pro)/Fbase)
                    Fintl[k]=float(Fintl[k]/Fbase)
                    nsi+=1
                    MapI(xsc,ysc,ros,RE,Rlens,pro, nsi, h, Rlens/RE, u/ros, Rlens/RE/Rin,AstarV[k], Fintl[k],tim[k]*period, float(AstarV[k]-Fintl[k])*A1+A2);
                    #MapI(xsc,ysc,ros,RE,Rlens,np.log10(np.abs(pro-sini)+0.01),nsi,h+1,Rlens/RE,u/ros,Rlens/RE/Rin,AstarV[k],Fintl[k],tim[k]*period)
            Flux[k]=float(AstarV[k]-Fintl[k])*A1+A2  
            print("Self_1: ",  k, nsi, Self, round(u,1),round(ros,1),round(u/ros,1), round(RE/Rsun,4) )
            print("AsV,AsI,Fintl,Flux:",round(AstarV[k],4), round(AstarI[k],4), round(Fintl[k],4), round(Flux[k],4))
            print("************************************************")            
        ########################################################################
        if(x1[k]>=0.0): 
            Rlens=float(Rstar)
            Mlens=float(Mstar)
            Dls=np.abs(x1[k])
            Dl=float(Dist*Kpc-Dls)
            Ds=float(Dist*Kpc)
            proj=float(Dl/Ds)
            RE=np.sqrt(4.0*G*Mlens)*np.sqrt(Dls*proj)/velocity+1.0e-50;
            ros=np.abs(RBH*proj/RE) 
            SourceA=float(np.pi*ros*ros*(1.0-limbl/3.0)/(dx*dy));
            u=float(disp[k]/RE);
            xsc=float(y1[k]/RE);  
            ysc=float(z1[k]/RE);
            Rin=np.abs(u-np.sqrt(u*u+4.0))*0.5
            AstarV[k]=1.0; AstarI[k]=1.0; Fintl[k]=0.0; 
            if(u<float(20.0*ros)):
                if(o1==0):  o1=k
                if(o1>0):   o2=k 
                Self=2;     
                if(ros>100.0):
                    if(u<ros):  AstarV[k]=float(1.0+2.0/ros/ros);      
                    else:       AstarV[k]=float(u*u+2.0)/sqrt(u*u*(u*u+4.0));
                else:  
                    vbb.a1=limbl; 
                    AstarV[k]=vbb.ESPLMag2(u, ros);   
                if(u<float(5.3*ros)):
                    sini= Fluxs(ros, xsc, ysc, sini, limbl)
                    Fbase=float(np.pi*ros*ros*(1.0-limbl/3.0)/(dx*dy));
                    pro,Fintl[k]=InRayShoot(Dl, Ds, RE, Mlens, Rlens, pro, sini)
                    AstarI[k]=float(np.sum(pro)/Fbase)
                    Fintl[k]=float(Fintl[k]/Fbase)
                    nsi+=1
                    MapI(xsc,ysc, ros, RE,Rlens,pro,nsi,h,Rlens/RE,u/ros,Rlens/RE/Rin,AstarV[k],Fintl[k],tim[k]*period,float(AstarV[k]-Fintl[k])*A2+A1);
                    #MapI(xsc,ysc, ros, RE,Rlens,np.log10(np.abs(pro-sini)+0.01),nsi,h+1,Rlens/RE,u/ros,Rlens/RE/Rin,AstarV[k],Fintl[k],tim[k]*period);
            Flux[k]=float(AstarV[k]-Fintl[k])*A2+A1              
            print("Self_2: ", k, nsi, Self, round(u,1), round(ros,1),round(u/ros,1), round(RE/Rsun,4))
            print("AsV,AsI,Fintl,Flux:",round(AstarV[k],4), round(AstarI[k],4), round(Fintl[k],4), round(Flux[k],4))
            print("************************************************")
        ########################################################################        
        fil1=open(direc+"light{0:d}.dat".format(h),"a+")
        ssa=np.array([k,nsi,Self,tim[k],AstarV[k],Fintl[k],AstarI[k],Flux[k],Fbase,u,ros,Rlens/RE ])
        np.savetxt(fil1,ssa.reshape((-1,12)),fmt ="%d  %d  %d  %.5f   %.8f   %.8f   %.8f  %.8f  %.5f  %.5f  %.4f  %.4f")
        fil1.close()
    print("limits of self_1:", s1, s2, tim[s1],   tim[s2] ) 
    print("limits of self_2:", o1, o2, tim[o1],   tim[o2] )
    FluxSelf(AstarV*A1+A2, (1.0-Fintl)*A1+A2, Flux, s1, s2, h,1)
    FluxSelf(AstarV*A2+A1, (1.0-Fintl)*A2+A1, Flux, o1, o2, h,2)
    Fluxtot(Flux,h)



