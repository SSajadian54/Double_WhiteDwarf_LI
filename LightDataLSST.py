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
from matplotlib import colors
cm=colors.ListedColormap(['purple', 'blue', 'darkgreen','yellowgreen', 'orange', 'red'])


import warnings
warnings.filterwarnings('ignore')
###############################################################################

G=6.67384*np.power(10.,-11.0);#// in [m^3/s^2*kg].
velocity=299792458.0;##//velosity of light  m/s
Msun=1.98892*np.power(10.,30.0); ##//in [kg].
Rsun=6.9634*np.power(10.0,8.0); ##///solar radius [meter]
AU=1.495978707*pow(10.0,11.0);
year   =float(365.2425)
Tobs   =float(10.0*year)
cadence=float(3.0) 
sea    =float(7.0*year/12.0);    
texp   =float(30.0)
Delta2 =float(0.005)
gama=  np.array([0.037,0.038,0.039,0.039,0.040,0.040])
seeing=np.array([0.77,0.73,0.70,0.67,0.65,0.63])
msky=  np.array([22.9,22.3,21.2,20.5,19.6,18.6])
Cm=    np.array([22.92,24.29,24.33,24.20,24.07,23.69])
Dci=   np.array([0.67,0.21,0.11,0.08,0.05,0.04])
km=    np.array([0.451,0.163,0.087,0.065,0.043,0.138])
wav=   np.array([0.3671, 0.4827, 0.6223, 0.7546, 0.8691, 0.9712])
sigma= np.array([0.022,0.02,0.017,0.017,0.027,0.027])
detect=np.array([23.4,24.6,24.3,23.6,22.9,21.7])
satu=  np.array([15.2,16.3,16.0,15.3,14.6,13.4])
M50=   np.array([23.68,24.89,24.43,24.00,24.45,22.60])
FWHM=  np.array([1.22087,1.10136,0.993103,0.967076,0.951766,0.936578])
AlAv=  np.array([1.55214, 1.17507, 0.870652, 0.665363, 0.509012, 0.423462])

###############################################################################

def tickfun(x, start, dd0):
    return( (start+x*dd0) )

###############################################################################

def smooth(y, box_pts):
    box = np.ones(box_pts)/box_pts
    y_smooth = np.convolve(y, box, mode='valid')
    return (y_smooth)

###############################################################################

def ErrorLSST(magA, fi, airmass):  
    seei=float(seeing[fi]+np.random.normal(0.0,0.2,1))
    if(seei<=0.4): seei=0.4; 
    mfive=float(Cm[fi]+0.0+0.5*(msky[fi]-21.0)+2.5*np.log10(0.7/seei)+1.25*np.log10(texp/30.0)-km[fi]*(airmass-1.0));
    x=np.power(10.0,0.4*(magA-mfive));
    Delta1=np.sqrt(abs((0.04-gama[fi])*x+gama[fi]*x*x)); 
    if(Delta1<0.0001):   Delta1=0.0001; 
    if(mfive<0.0 or Delta1<0.00001):
        print("Error:  ",  mfive, Delta1 , magA, msky[fi],  seei, airmass, fi)
        input("Enter a number")
    return(np.sqrt(Delta2*Delta2+Delta1*Delta1)); 

################################################################################


###LSST
period=10.0*year*24.0*3600.0;##second
semi= pow(period*period*G*Msun*1.5/(4.0*np.pi*np.pi),1.0/3.0)
#semi=float(100000.0*0.01*Rsun)
#period=np.sqrt(4.0*np.pi*np.pi/(G*Msun*1.5))*pow(semi,1.5)/(3600.0*24.0);#//days   
V=float(2.0*np.pi*semi/period) ##[m/s]
print("Duration(min),  v(km/s) ,  period(days): " ,float(2.0*0.01*Rsun/V/60.0), V*0.001,  period/24.0/3600.0 , semi/Rsun/0.01)
RE=np.sqrt(4.0*G*Msun*1.3*semi)/velocity+1.0e-50;
print ("tE(hours): ",  RE/V/60.0/60.0)
for i in range(20): 
    magg=float(15.0+10.0*i/20.0)
    errm=ErrorLSST(magg,2,1.2)
    errA=np.abs((np.power(10.0,-0.4*errm)-1.0)) 
    print("maggs:  ",  magg, errm,  errA  )
print ("*********************************************************")    


###Roman
period=62.0*24.0*3600.0;##second
semi=pow(period*period*G*Msun*1.5/(4.0*np.pi*np.pi),1.0/3.0)
V=float(2.0*np.pi*semi/period)##[m/s]
print("Duration(min),  v(km/s) ,  period(days): " ,float(2.0*0.01*Rsun/V/60.0), V*0.001,  period/24.0/3600.0 , semi/Rsun/0.01)
RE=np.sqrt(4.0*G*Msun*1.3*semi)/velocity+1.0e-50;
print("tE(min): ",  RE/V/60.0)
print("*********************************************************")   

 
###TESS
period=27.4*24.0*3600.0;##second
semi=pow(period*period*G*Msun*1.5/(4.0*np.pi*np.pi),1.0/3.0)
V=float(2.0*np.pi*semi/period)##[m/s]
print("Duration(min),  v(km/s) ,  period(days): " ,float(2.0*0.01*Rsun/V/60.0), V*0.001,  period/24.0/3600.0 , semi/Rsun/0.01)
RE=np.sqrt(4.0*G*Msun*1.3*semi)/velocity+1.0e-50;
print("tE(min): ",  RE/V/60.0/60.0)
print("*********************************************************")    
    
    
input("Enter a number ")



nc=int(31)
nx=int(10)
Nl=int(20000)#filter_LSST.txt
direc="./files/"
Num=int(3)

fif=open(direc+"D_t.dat","w")
fif.close() 

ppp=np.loadtxt("./filterLSST.txt") 
airm=np.zeros((Nl))
filt=np.zeros((Nl))
airm, filt=ppp[:,0], ppp[:,1]

################################################################################

ndet=int(0)
for ll in range(1): 
    f1=open(direc+"C_{0:d}.dat".format(Num),"r")
    nf= sum(1 for line in f1)  
    par =np.zeros((nf,nc))
    par=np.loadtxt(direc+"C_{0:d}.dat".format(Num)) 
    mar=np.zeros((nf*6,nx))
    mar=np.loadtxt(direc+"D_{0:d}.dat".format(Num)) 
    
    Mabs=np.zeros((6));  Maps=np.zeros((6));  fb=np.zeros((6))
    Mabl=np.zeros((6));  Mapl=np.zeros((6));  nsbl=np.zeros((6)); ratio=np.zeros((6))
    base=np.zeros((6));  Ext =np.zeros((6));  a1=np.zeros((6));   a2=np.zeros((6))
    
    for i in range(nf):
        count,numl, nums, lat, lon      =int(par[i,0]),par[i,1], par[i,2], par[i,3], par[i,4]
        Dist,mass,Teffs,Rstar,limbs     =par[i,5],par[i,6], par[i,7], par[i,8], par[i,9]
        Loggs,MBH,RBH,Loggl,Teffl,inc   =par[i,10],par[i,11],par[i,12],par[i,13],par[i,14],par[i,15]
        teta,ecen,period,lsemi,tp ,limbl=par[i,16],par[i,17],par[i,18], par[i,19],par[i,20],par[i,21]
        smagG,smagBP,smagRP,lmagG,lmagBP=par[i,22],par[i,23], par[i,24],par[i,25],par[i,26]
        lmagRP,Flags, ages, metals      =par[i,27],int(par[i,28]),  par[i,29],par[i,30]
        
        for j in range(6):
            Mabs[j],Maps[j],Mabl[j],Mapl[j],fb[j],nsbl[j],base[j],Ext[j]=mar[i*6+j,2:]
            ratio[j]=np.power(10.0,-0.4*float(Mapl[j]-Maps[j]));#FluxL/FluxS 
            a1[j]=1.000000/(1.0+ratio[j])
            a2[j]=ratio[j]/(1.0+ratio[j])
            
        ntran=float(Tobs/period)
        emt0 =ErrorLSST(base[2], 2, 1.2)
        deltaA0=np.abs((np.power(10.0,-0.4*emt0)-1.0))
        l1=-1; l2=-1;   o1=-1;  o2=-1 
        ta1=-1;ta2=-1; tb1=-1; tb2=-1
        snr=np.zeros((2));
        kin=np.zeros((2));         
        rol=np.zeros((2)); 
        dem=np.zeros((2));                 
        ros=np.zeros((2));  
        imp=np.zeros((2));
        dep=np.zeros((2));
        dur=np.zeros((2));
        nro=np.zeros((2));
        flagd=np.zeros((2)) 
        
        stt=int(np.random.rand(1)*(Nl-2))   
        nm=-1;  nk=0;   timm=0.0;  ea=-1; 
        #######################################################################
        if(Flags>0): 
            print("***************Count:  ", count, Flags)
            f2=open(direc+'M_{0:d}.dat'.format(int(count)))
            nm=int(len(f2.readlines()))  
            mod=np.zeros((nm, 9))
            ast=np.zeros((nm,2,6)) 
            lef=np.zeros((nm,3,6)) 
            dat=np.zeros((nm,5))
            dat[:,0]=dat[:,0]-1.0
            mod=np.loadtxt(direc+'M_{0:d}.dat'.format(int(count)))
            for j in range(nm):
                selfl=int(mod[j,8])
                if(selfl==1 and l1<0):  l1=int(j)
                if(selfl==1 and l1>=0): l2=int(j)
                if(selfl==2 and o1<0):  o1=int(j)
                if(selfl==2 and o1>=0): o2=int(j)
                if(selfl==1): 
                    ast[j,0,:]=fb*((mod[j,2]-mod[j,3])*a1+a2)+1.0-fb;#IRS_method Overal flux
                    ast[j,1,:]=fb*((mod[j,1]-mod[j,3])*a1+a2)+1.0-fb;#Valerio_model  Overal flux
                    lef[j,0,:]=fb*((1.0-mod[j,3])*a1+a2)+1.0-fb;#IRS_method, Occultation
                    lef[j,1,:]=fb*((mod[j,2]-0.0)*a1+a2)+1.0-fb;#IRS_method, Self_lensing
                    lef[j,2,:]=fb*((mod[j,1]-0.0)*a1+a2)+1.0-fb;#Valerio_method, Self_lensing 
                    nro[0]+=1.0
                    rol[0]+=float(RBH/mod[j,4])
                    emtt=ErrorLSST(base[2]-2.5*np.log10(ast[j,1,2]),2,1.2)  
                    dem[0]+=np.abs(np.power(10.0,-0.4*emtt)-1.0)*ast[j,1,2]
                elif(selfl==2):
                    ast[j,0,:]=fb*((mod[j,2]-mod[j,3])*a2+a1)+1.0-fb;#IRS_method   Overal flux
                    ast[j,1,:]=fb*((mod[j,1]-mod[j,3])*a2+a1)+1.0-fb;#Valerio_model Overal flux  
                    lef[j,0,:]=fb*((1.0-mod[j,3])*a2+a1)+1.0-fb;#IRS_method, Occultation
                    lef[j,1,:]=fb*((mod[j,2]-0.0)*a2+a1)+1.0-fb;#IRS_method, Self_lensing
                    lef[j,2,:]=fb*((mod[j,1]-0.0)*a2+a1)+1.0-fb;#Valerio_method, Self_lensing
                    nro[1]+=1.0  
                    rol[1]+=float(Rstar/mod[j,4])
                    emtt=ErrorLSST(base[2]-2.5*np.log10(ast[j,1,2]),2,1.2)  
                    dem[1]+=np.abs(np.power(10.0,-0.4*emtt)-1.0)*ast[j,1,2]
                else:
                    ast[j,0,:]=1.0; ast[j,1,:]=1.0;  lef[j,0,:]=1.0;  lef[j,1,:]=1.0;  lef[j,2,:]=1.0 
                if(selfl==1 and ta1<0  and abs(ast[j,1,2]-1.0)>=float(deltaA0*0.5)): ta1=int(j)
                if(selfl==1 and ta1>=0 and abs(ast[j,1,2]-1.0)>=float(deltaA0*0.5)): ta2=int(j)
                if(selfl==2 and tb1<0  and abs(ast[j,1,2]-1.0)>=float(deltaA0*0.5)): tb1=int(j)
                if(selfl==2 and tb1>=0 and abs(ast[j,1,2]-1.0)>=float(deltaA0*0.5)): tb2=int(j)
                
                hh=float(mod[j,0])
                if(hh>=year):
                    while(hh>=year): 
                        hh=float(hh-year) 
                if(j>0 and mod[j,0]<=Tobs and hh<=sea):##and mod[j-1,8]>0 and mod[j,8]>0): 
                    timm += abs(mod[j,0]-mod[j-1,0]);  
                    if(timm>cadence):
                        timm=float(timm-cadence)
                        stt+=1
                        if(int(stt+1)>Nl): stt=0
                        fi=int(filt[stt])
                        Astar=ast[j,1,fi]
                        amass=float(airm[stt])
                        maga =float(base[fi]-2.5*np.log10(Astar))
                        emt  =ErrorLSST(maga, int(fi), amass)
                        deltaA=np.abs((np.power(10.0,-0.4*emt)-1.0)*Astar);
                        dat[nk,0]=float(mod[j,0])
                        dat[nk,1]=float(ast[j,1,2]+np.random.normal(0.0,deltaA,1)); 
                        dat[nk,2]=float(deltaA)
                        dat[nk,3]=fi
                        dat[nk,4]=int(mod[j,8])
                        nk+=1
                        if(timm>cadence or Astar<0.0 or Astar>100 or deltaA<0 or maga<0 or emt<0.0):
                            print("Error(1): ", timm/cadence, Astar, deltaA, mod[j-1,0], mod[j,0], maga, emt, timm, nk, count)
                            input("Enter a number ")   
            print("Number of data", count, nk, dat[:nk,0], dat[:nk, 1],dat[:nk, 2],dat[:nk,3], dat[:nk,4] )                 
            ###################################################################
            kin[0]=-1;  kin[1]=-1; 
            rol[0]=float(rol[0]/nro[0])  
            rol[1]=float(rol[1]/nro[1])    
            dem[0]=float(dem[0]/nro[0])  
            dem[1]=float(dem[1]/nro[1])    
            if(l1>=0 and l2>=0):
                ros[0]=np.mean(np.abs(mod[l1:l2,5]));
                imp[0]= np.min(np.abs(mod[l1:l2,6]))
                dep[0]= np.abs(np.max(np.abs(ast[l1:l2,1,2]-1.0))/dem[0])## depth/error
                nm1=int(np.argmax(np.abs(ast[l1:l2,1,2]-1.0))+l1)
                if(ast[nm1,1,2]>=1.0): kin[0]=0#self-lensing
                else:                  kin[0]=1#occultation
                snr[0]=float(np.sqrt(ntran*1.0)*dep[0])+1.0e-9
                if(ta1>=0 and ta2>=0): dur[0]=np.abs(mod[ta2,0]-mod[ta1,0])/cadence
            if(o1>=0 and o2>=0):                
                ros[1]=np.mean(np.abs(mod[o1:o2,5]))  
                imp[1]= np.min(np.abs(mod[o1:o2,6]))
                dep[1]= np.abs(np.max(np.abs(ast[o1:o2,1,2]-1.0))/dem[1])
                nm2=int(np.argmax(np.abs(ast[o1:o2,1,2]-1.0))+o1)
                if(ast[nm2,1,2]>=1.0): kin[1]=0#self-lensing
                else:                  kin[1]=1#occultation
                snr[1]=float(np.sqrt(ntran*1.0)*dep[1])+1.0e-9
                if(tb1>=0 and tb2>=0): dur[1]=np.abs(mod[tb2,0]-mod[tb1,0])/cadence            
            flagd[0]=0; flagd[1]=0
            if(snr[0]>=3.0 and ntran>=1.0 and dep[0]>=2.0 and dur[0]>=1.0):  flagd[0]=1; ea=0 
            if(snr[1]>=3.0 and ntran>=1.0 and dep[1]>=2.0 and dur[1]>=1.0):  flagd[1]=1; ea=1 
            if(flagd[0]>0 or flagd[1]>0):
                ndet+=1
                Stat=str(r"$\rm{Detectable};~\rm{SNR}=$")##+str(round(snr[ea],1)))
                col='g' 
            if(flagd[0]<1 and flagd[1]<1): 
                Stat=str(r"$\rm{Not}~\rm{detectable};~\rm{SNR}=$")##+str(round(0.5*(snr[0]+snr[1]),1)))
                col='r'                 
            if(l1>=0 and o1>=0 and (l2<0 or o2<0 or nm1<l1 or nm1>l2 or nm2<o1 or nm2>o2 or kin[0]<0 or kin[1]<0 or dep[0]<0.0 or 
            dep[1]<0 or imp[0]>31.0 or imp[1]>31.0) or deltaA0<0.0):   
                print("Big error, ", l1, l2, o1, o2, nm1, nm2, ta1, ta2, tb1, tb2 )
                print(deltaA0, dur, snr, rol, ros, dep, imp, kin)
                input("Enter a number(iiii)")
                
        fif=open(direc+"D_t.dat","a")
        test=np.zeros((nc+26))#57
        test[:nc]=par[i,:]
        test[nc:]=np.array([l1, l2, o1, o2, ros[0],ros[1], rol[0],rol[1], imp[0],imp[1], dep[0],dep[1], dur[0],dur[1], snr[0],snr[1], 
        kin[0],kin[1], flagd[0],flagd[1], ea, deltaA0,emt0, ntran, dem[0], dem[1]])
        np.savetxt(fif,test.reshape((-1,nc+26)),fmt="%d  %d  %d  %.5f %.5f %.6f %.6f  %.2f  %.6f  %.3f  %.4f %.6f %.6f %.5f %.5f %.5f %.5f  %.4f  %.7f  %.8f  %.6f  %.5f  %.5f  %.5f  %.5f  %.5f  %.5f  %.5f  %d %.5f %.5f %d  %d  %d  %d  %.7f  %.7f  %.7f  %.7f  %.7f %.7f  %.9f  %.9f  %.9f  %.9f  %.5f  %.5f  %d  %d  %d  %d  %d  %.7f  %.7f  %.5f  %.7f  %.7f")
        fif.close()       
        #######################################################################    
        if(o1>=0 and l1>=0 and o2>=0 and l2>=0 and Flags>0 and (mod[l2,0]<mod[o1,0] or mod[o2,0]<mod[l1,0])): 
            nml=int(np.argmax(lef[l1:l2,2,2])+l1)
            nmo=int(np.argmax(lef[o1:o2,2,2])+o1)
            sft=0.0
            if(mod[l2,0]<mod[o1,0]):
                lim= np.array([limbs,limbl])
                sft=-mod[o1,0]+mod[l2,0]; stat=1
                tt=[int(l1+(l2-l1)*0.1),int(nml),int(l1+(l2-l1)*0.85),int(l2),int(o1+(o2-o1)*0.15),int(nmo),int(o1+(o2-o1)*0.9)] 
            elif(mod[o2,0]<mod[l1,0]):        
                ros=np.array([ros[1],ros[0]]);  imp=np.array([imp[1],imp[0]]); dep=np.array([dep[1],dep[0]]); 
                rol=np.array([rol[1],rol[0]]);  lim=np.array([limbl,limbs]);   dur=np.array([dur[1],dur[0]]);
                snr=np.array([snr[1],snr[0]]); 
                sft=-mod[o2,0]+mod[l1,0]; stat=2
                tt=[int(o1+(o2-o1)*0.1),int(nmo),int(o1+(o2-o1)*0.85),int(l1),int(l1+(l2-l1)*0.15),int(nml),int(l1+(l2-l1)*0.9)]
            else:  
                print(l1, l2, o1, o2, mod[l1,0], mod[l2,0], mod[o1,0], mod[o2,0], count)  
                print(mod[o2,8], mod[o1,8])                 
                input("Error: o1, o2, l1, l2 ")
            print(l1, l2, o1, o2, Flags, nm, nml, nmo, tt)    
            labs=[str(round(mod[tt[0],0],2)),str(round(mod[tt[1],0],2)),str(round(mod[tt[2],0],2)), str('...'),
                  str(round(mod[tt[4],0],2)),str(round(mod[tt[5],0],2)),str(round(mod[tt[6],0],2))]
            ###################################################################    
            plt.clf()
            plt.cla()
            fig=plt.figure(figsize=(8,6))
            ax=fig.add_subplot(111)
            plt.plot(mod[l1:l2,0],lef[l1:l2,0,2],color='darkred',ls='-',lw=1.9)
            plt.plot(mod[l1:l2,0],lef[l1:l2,2,2],color='darkgreen',ls='--',lw=1.9)
            plt.plot(mod[l1:l2,0],ast[l1:l2,1,2],'k',ls='-.', lw=1.7)
            plt.plot(mod[o1:o2,0]+sft,lef[o1:o2,0,2],'r-',label=r"$\rm{Finite}-\rm{Lens}$",lw=1.9)   
            plt.plot(mod[o1:o2,0]+sft,lef[o1:o2,2,2],'g--',label=r"$\rm{Self}-\rm{Lensing}$",lw=1.9)
            plt.plot(mod[o1:o2,0]+sft,ast[o1:o2,1,2],'k-.', label=r"$\rm{Overall}-\rm{Flux}$",lw=1.5) 
            shift=0.0
            for j in range(nk):
                nco=abs(int(dat[j,3]))
                if(dat[j,4]==1 or dat[j,4]==2):##self==1,2
                    if(dat[j,0]>=mod[l1,0] and dat[j,0]<=float(mod[l2,0]+cadence)): shift=0;
                    elif(dat[j,0]>=mod[o1,0] and dat[j,0]<=float(mod[o2,0]+cadence)): shift=sft;
                    else: 
                        print("time: ", dat[:nk,0], dat[j,0], j, nk)
                        print(mod[l1:l2,0],  mod[o1:o2,0])
                        input("Data is out of the range")
                    plt.errorbar(dat[j,0]+shift,dat[j,1],yerr=dat[j,2],fmt=".",markersize=9,color=cm(nco),ecolor=cm(nco),elinewidth=0.3, capsize=0,alpha=0.75)
            plt.title(
            r"$M_{1}(M_{\odot})=$"+'{0:.1f}'.format(MBH)+
            r"$;~M_{2}(M_{\odot})=$"+'{0:.1f}'.format(mass)+
            r"$;~T(\rm{days})=$"+'{0:.1f}'.format(period)+
            r"$;~\mathcal{F}=$"+'{0:.2f}'.format(ratio[2])+
            r"$;~f_{\rm{b}}=$"+'{0:.2f}'.format(fb[2])+
            r"$;~m_{\rm{b}}(\rm{mag})=$"+'{0:.1f}'.format(base[2])+"\n"+
            r"$\Delta t/\tau=$"+'{0:.1f}, {1:.1f}'.format(dur[0],dur[1])+
            r"$;~\Delta\rm{F}/$"+r"${\sigma}_{\rm{A}}=$"+'{0:.1f}, {1:.1f}'.format(dep[0],dep[1])+
            r"$;~\rho_{\star}=$"+'{0:.1f}, {1:.1f}'.format(ros[0],ros[1])+
            r"$;~\rho_{\rm{l}}=$"+'{0:.1f}, {1:.1f}'.format(rol[0],rol[1])+
            r"$;~u_{0}/\rho_{\star}=$"+'{0:.1f}, {1:.1f}'.format(imp[0],imp[1]),fontsize=14.0,color='k')
            if(stat==1): 
                plt.xlim(mod[l1,0], mod[o2,0]+sft)
                ticc=np.array([mod[tt[0],0],mod[tt[1],0],mod[tt[2],0],mod[tt[3],0],mod[tt[4],0]+sft,mod[tt[5],0]+sft,mod[tt[6],0]+sft])
            if(stat==2):  
                plt.xlim(mod[o1,0]+sft, mod[l2,0])
                ticc=np.array([mod[tt[0],0]+sft,mod[tt[1],0]+sft,mod[tt[2],0]+sft,mod[tt[3],0],mod[tt[4],0],mod[tt[5],0], mod[tt[6],0]])
            ax.set_xticks(ticc,labels=labs)
            rcParams['xtick.major.pad']='-3.0'
            plt.ylabel(r"$\rm{Normalized}~\rm{Flux}$",fontsize=20)
            plt.xlabel(r"$\rm{time}(\rm{days})$",fontsize=20)
            plt.xticks(fontsize=17, rotation=30)
            plt.yticks(fontsize=17, rotation=0)
            legend=ax.legend(prop={"size":12},loc='best',frameon=True, fancybox = True,shadow=True,framealpha=0.5)
            legend.get_frame().set_facecolor('whitesmoke')
            dd=ax.legend(title=Stat+str(round(snr[0],1))+',~'+str(round(snr[1],1)) )
            plt.setp(legend.get_title(),color=col)
            fig=plt.gcf()
            fig.tight_layout()  
            fig.savefig("./figs/LightCurveB{0:d}.jpg".format(count),dpi=200)
            #######################################################################
        if(Flags>0):#(l1<0oro1<0or(l1>=0 and l2>=0 and o1>=0 and o2>=0 and not(mod[l2,0]<mod[o1,0] or mod[o2,0]<mod[l1,0]) ))): 
            plt.cla()
            plt.clf()
            fig=plt.figure(figsize=(8,6))
            ax=fig.add_subplot(111)
            plt.plot(mod[:,0],lef[:,0,2],'r-.',label=r"$\rm{Finite}-\rm{Lens}$",lw=1.7)   
            plt.plot(mod[:,0],lef[:,2,2],'g--',label=r"$\rm{Self}-\rm{Lensing}$",lw=1.4)
            plt.plot(mod[:,0],ast[:,1,2],'k:', label=r"$\rm{Overall}-\rm{Flux}$",lw=1.4)
            for j in range(nk):
                nco=int(dat[j,3])
                plt.errorbar(dat[j,0],dat[j,1],yerr=dat[j,2],fmt=".",markersize=12,color=cm(nco),ecolor=cm(nco),elinewidth=0.3, capsize=0,alpha=0.75)
            plt.title(
            r"$M_{1}(M_{\odot})=$"+'{0:.1f}'.format(MBH)+
            r"$;~M_{2}(M_{\odot})=$"+'{0:.1f}'.format(mass)+
            r"$;~T(\rm{days})=$"+'{0:.1f}'.format(period)+
            r"$;~\mathcal{F}=$"+'{0:.2f}'.format(ratio[2])+
            r"$;~m_{\rm{b}}(\rm{mag})=$"+'{0:.1f}'.format(base[2])+"\n"+
            r"$f_{\rm{b}}=$"+'{0:.2f}'.format(fb[2])+
            r"$;~\Delta t/\tau=$"+'{0:.1f},{1:.1f}'.format(dur[0], dur[1])+
            r"$;~\Delta\rm{F}/$"+r"${\sigma}_{\rm{A}}=$"+'{0:.1f},{1:.1f}'.format(dep[0], dep[1])+
            r"$;~\rho_{\star}=$"+'{0:.1f},{1:.1f}'.format(ros[0],ros[1])+
            #r"$;~\rho_{\rm{l}}=$"+'{0:.2f},{1:.2f}'.format(rol[0],rol[1])+
            r"$;~u_{0}/\rho_{\star}=$"+'{0:.1f},{1:.1f}'.format(imp[0],imp[1]),fontsize=14.0,color='k')
            plt.xticks(fontsize=18, rotation=0)
            plt.yticks(fontsize=18, rotation=0)
            plt.xlabel(r"$\rm{time}(\rm{days})$", fontsize=18)
            plt.ylabel(r"$\rm{Normalized}~\rm{Flux}$",fontsize=19)
            legend=ax.legend(prop={"size":12.5},loc='best',frameon=True, fancybox = True,shadow=True,framealpha=0.5)
            legend.get_frame().set_facecolor('whitesmoke')
            dd=ax.legend(title=Stat+str(round(snr[0],1))+',~'+str(round(snr[1],1)) )
            plt.setp(legend.get_title(),color=col)
            fig=plt.gcf()
            fig.tight_layout()
            fig.savefig("./figs/Lighttot{0:d}.jpg".format(count), dpi=200)
        #######################################################################
            print("==========================================================")    
            print("Information:  ")
            print(count, imp, rol, ros, snr,  kin, flagd,  dur,    dep/deltaA0) 
            print(ta1, ta2, tb1, tb2, l1, l2, o1, o2, base, emt0, deltaA0, ntran)
            print("Light curve is plotted ", count)
            print("==========================================================")    
            
print("Efficiecy:  ", float(ndet*100.0/nf), ndet , nf)









