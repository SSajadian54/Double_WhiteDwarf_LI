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

###############################################################################
def tickfun(x, start, dd0):
    return((start+x*dd0) )


def smooth(y, box_pts):
    box = np.ones(box_pts)/box_pts
    y_smooth = np.convolve(y, box, mode='valid')
    return (y_smooth)


def ErrorTESS(maga):
    emt=-1.0;     
    if(maga<7.5):       emt=float(0.22*maga-5.850); 
    elif(maga<12.5):    emt=float(0.27*maga-6.225);  
    else:               emt=float(0.31*maga-6.725);    
    #emt=float(emt+np.random.normal(0.0,0.05,1))
    if(emt<-5.0): emt=-5.0 
    emt=np.power(10.0,emt)
    if(emt<0.00001 or emt>0.5 or maga<0.0):
        print("Error emt:  ", emt, maga); 
    return(emt); 
################################################################################
nc=int(32+1+7) 
direc="./files/"
Num=int(3)
Tobs=float(2.0*13.7)## days
cadence=float(2.0/60.0/24.0)## days
fij=open("./HistoTESS/detparam.dat","w")
fij.close(); 

################################################################################
ndet=int(0)
for ll in range(1): 
    f1=open(direc+"C_{0:d}.dat".format(Num),"r")
    nf= sum(1 for line in f1)  
    par =np.zeros((nf,nc))
    pard=np.zeros((nf,nc))
    dete=np.zeros((nf,12))##count, snr, depth, cdpp, ntran, errA, duration/cadence, impact, rho, mbase, errorM, flag_detection, 
    par=np.loadtxt(direc+"C_{0:d}.dat".format(Num)) 
    par=par[par[:,0].argsort()] 
    for i in range(nf):
        count,numl, nums, lat, lon       =par[i,0],par[i,1], par[i,2], par[i,3], par[i,4]
        Dist,mass,Teffs,Rstar,limbs      =par[i,5],par[i,6], par[i,7], par[i,8], par[i,9]
        Lumis,Loggs,Maps,Mabs,magb,fb,Ext=par[i,10],par[i,11],par[i,12],par[i,13],par[i,14],par[i,15],par[i,16]
        MBH, RBH, Lumil, Loggl, Teffl    =par[i,17],par[i,18], par[i,19],par[i,20],par[i,21]
        Mapl,Mabl,inc,teta,ecen          =par[i,22],par[i,23], par[i,24],par[i,25],par[i,26]
        period,lsemi,tp,ratio,cdpp,limbl =par[i,27],par[i,28], par[i,29],par[i,30],par[i,31],par[i,32]
        smagG,smagBP,smagRP,lmagG,lmagBP,lmagRP,Flags=par[i,33],par[i,34], par[i,35],par[i,36],par[i,37],par[i,38], int(par[i,39])         
        a1=1.000/(1.0+ratio)
        a2=ratio/(1.0+ratio) 
        base=float(-2.5*np.log10(np.power(10.0,-0.4*Maps)+np.power(10.0,-0.4*Mapl)))
        #print("baseline magnitude:  ",  base, magb, count)
        ########################################################################
        l1=-1;l2=-1
        o1=-1;o2=-1 
        ta1=-1; ta2=-1
        tb1=-1; tb2=-1
        count=int(count)
        nm=-1;  
        if(Flags>0):
            f2=open(direc+'M_{0:d}.dat'.format(count))
            nm=int(len(f2.readlines()))  
            mod=np.zeros((nm,9))#tim, Astar, Astar2, finl, l.RE/Rsun, s.ros, u/s.ros, disp/l.a, self);//9 
            ast=np.zeros((nm,2)) 
            lef=np.zeros((nm,3)) 
            dat=np.zeros((nm,4))
            dat[:,0]=dat[:,0]-1.0
            mod=np.loadtxt(direc+'M_{0:d}.dat'.format(count))
            timm=0.0; nk=0;
            emt0=np.abs(ErrorTESS(base))
            deltaA0=np.abs((np.power(10.0,-0.4*emt0)-1.0));
            for j in range(nm):
                selfl=int(mod[j,8])
                if(selfl==1 and l1<0): l1=int(j)
                if(selfl==1 and l1>0): l2=int(j)
                if(selfl==2 and o1<0): o1=int(j)
                if(selfl==2 and o1>0): o2=int(j)
                if(selfl==1): 
                    ast[j,0]=fb*((mod[j,2]-mod[j,3])*a1+a2)+1.0-fb;#IRS_method Overal flux
                    ast[j,1]=fb*((mod[j,1]-mod[j,3])*a1+a2)+1.0-fb;#Valerio_model  Overal flux
                    lef[j,0]=fb*((1.0-mod[j,3])*a1+a2)+1.0-fb;#IRS_method, Occultation
                    lef[j,1]=fb*((mod[j,2]-0.0)*a1+a2)+1.0-fb;#IRS_method, Self_lensing
                    lef[j,2]=fb*((mod[j,1]-0.0)*a1+a2)+1.0-fb;#Valerio_method, Self_lensing 
                elif(selfl==2): 
                    ast[j,0]=fb*((mod[j,2]-mod[j,3])*a2+a1)+1.0-fb;#IRS_method   Overal flux
                    ast[j,1]=fb*((mod[j,1]-mod[j,3])*a2+a1)+1.0-fb;#Valerio_model Overal flux  
                    lef[j,0]=fb*((1.0-mod[j,3])*a2+a1)+1.0-fb;#IRS_method, Occultation
                    lef[j,1]=fb*((mod[j,2]-0.0)*a2+a1)+1.0-fb;#IRS_method, Self_lensing
                    lef[j,2]=fb*((mod[j,1]-0.0)*a2+a1)+1.0-fb;#Valerio_method, Self_lensing
                else:
                    ast[j,0]=1.0; ast[j,1]=1.0;  lef[j,0]=1.0;  lef[j,1]=1.0;  lef[j,2]=1.0 
                if(selfl==1 and ta1<0 and abs(ast[j,1]-1.0)>=float(deltaA0*0.5)): ta1=int(j)
                if(selfl==1 and ta1>0 and abs(ast[j,1]-1.0)>=float(deltaA0*0.5)): ta2=int(j)
                if(selfl==2 and tb1<0 and abs(ast[j,1]-1.0)>=float(deltaA0*0.5)): tb1=int(j)
                if(selfl==2 and tb1>0 and abs(ast[j,1]-1.0)>=float(deltaA0*0.5)): tb2=int(j)
                if(j>0 and mod[j,0]<=Tobs and (mod[j,0]<12.7 or (mod[j,0]>=13.7 and mod[j,0]<26.4)) and mod[j-1,8]>0 and mod[j,8]>0): 
                    timm=timm+abs(mod[j,0]-mod[j-1,0]); nn=0; 
                    if(timm>cadence):
                        nn+=1
                        timm=float(timm-cadence)
                        Astar=ast[j,1]## abs(ast[j-1,1]+(ast[j,1]-ast[j-1,1])*cadence*nn/(mod[j,0]-mod[j-1,0]))
                        if(timm>cadence): 
                            print(Astar, ast[j-1,0], ast[j,1], mod[j-1,0], mod[j,0], timm/cadence,count,abs(mod[j,0]-mod[j-1,0])*24.0*60)
                            print(mod[j,8],   mod[j-1,8], j )
                            input("Enter a number")
                        maga=float(base-2.5*np.log10(Astar))
                        emt=np.abs(ErrorTESS(maga))
                        deltaA=np.abs((np.power(10.0,-0.4*emt)-1.0)*Astar);
                        dat[nk,0]=float(mod[j,0])##+cadence*nn) 
                        dat[nk,1]=float(Astar+np.random.normal(0.0,deltaA,1)); 
                        dat[nk,2]=float(deltaA)
                        nk+=1
                        if(Astar<0.0 or Astar>100.0 or deltaA<0.0 or deltaA>1.0 or mod[j,0]==mod[j-1,0] or maga<0.0 or maga>30.0 
                            or emt>1.0 or emt<0.0): 
                            print("Error: ", Astar, deltaA, mod[j-1,0], mod[j,0], maga, emt, timm, nk, nn, count)
                            input("Enter a number ")    
            ################################################################### 
            dur1=-10.0; dur2=-10.0; dur=0.0;  
            rhos= np.mean(np.abs(mod[:,5]))  
            impact=np.min(np.abs(mod[:,6]))#[u/rho*]
            par[i,4]=impact
            ntran=float(Tobs/period)
            depth=np.max(np.abs(ast[:,1]-1.0))+0.00001
            nmax=int(np.argmax(np.abs(ast[:,1]-1.0)))
            if(ast[nmax,1]>=1.0):   par[i,3]=0## self-lensing
            elif(ast[nmax,1]<1.0):  par[i,3]=1## occultation
            snr=float(np.sqrt(ntran*1.0)*depth*1000000.0/cdpp)+0.001
            if(nk>0): errA=np.mean(np.abs(dat[:nk,2]))
            else:     errA=deltaA0
            if(ta1>=0 and ta2>=0): dur1=abs(mod[ta2,0]-mod[ta1,0])##days
            if(tb1>=0 and tb2>=0): dur2=abs(mod[tb2,0]-mod[tb1,0])##days
            if(dur1>=0.0 or dur2>=0.0): dur=np.max(np.array([dur1,dur2]))/cadence
            else: dur=0.0/cadence     
            flag=0;
            if(snr>=3.0 and ntran>=1.0 and depth>=abs(2.0*errA) and dur>=1.0):
                pard[ndet,:]=par[i,:]
                flag=1;  ndet+=1
                Stat=str(r"$\rm{Detectable};~\rm{SNR}=$"+str(round(snr,1)))
                col='g'
            else: 
                flag=-1
                Stat=str(r"$\rm{Not}~\rm{detectable};~\rm{SNR}=$"+str(round(snr,1)))
                col='r' 
            dete[i,:]=np.array([count,np.log10(snr),np.log10(depth),cdpp,np.log10(ntran),np.log10(errA),dur,impact,rhos,base,emt0,flag]) 
            fij=open("./detparam.dat","a")
            np.savetxt(fij,dete[i,:].reshape((-1,12)),fmt="%d  %.7f  %.7f  %.7f  %.6f  %.6f  %.6f  %.9f  %.9f  %.8f  %.8f  %d") 
            fij.close()
            if(nmax<0 or nmax>int(nm-1) or ast[nmax,1]<0.0 or ast[nmax,1]>100.0 or par[i,3]<0 or par[i,3]>1 or depth<0.0 or 
                flag==0 or errA<=0.0 or snr<=0.0 or dur<0.0 or impact<=0.0):   
                print ("Big error, ", nmax,   ast[nmax,1], par[i,3], count, nk, depth, np.abs(ast[:,1]-1.0) ,flag, dur, errA )
                print (ta1, ta2, tb1, tb2, dur,  impact,  snr)
                input("Enter a number(iiii)")
            print("==============================================")    
            print("Information about each light curve:  ")
            print(count, rhos, impact , depth, snr, ntran,  errA, dur1, dur2, dur, flag,  base, emt0) 
            print(ta1, ta2, tb1, tb2, deltaA0,   depth/deltaA0, depth/errA/3.0, dur )   
            ####################################################################
            ##tim, Astar, Astar2, finl, l.RE/Rsun, s.ros, u/s.ros, disp/l.a, self);//9 
            '''
            if(o1>=0 and l1>=0 and o2>=0 and l2>=0 and int(count)%1==0):
                ros1= np.mean(mod[l1:l2,5])      
                imp1= np.min( mod[l1:l2,6])    
                fin1= RBH/np.mean(mod[l1:l2,4])    
                ros2= np.mean(mod[o1:o2,5])      
                imp2= np.min( mod[o1:o2,6])   
                fin2= Rstar/np.mean(mod[o1:o2,4]) 
                nml=int(np.argmax(lef[l1:l2,2])+l1)
                nmo=int(np.argmax(lef[o1:o2,2])+o1)
                
                sft=0.0
                if(mod[l2,0]<mod[o1,0]):
                    ros=np.array([ros1,ros2]); imp=np.array([imp1,imp2]); fin=np.array([fin1,fin2]);  lim=np.array([limbs,limbl])
                    sft=-mod[o1,0]+mod[l2,0]; stat=1
                    tt=[int(l1+(l2-l1)*0.1),int(nml),int(l1+(l2-l1)*0.85),int(l2),int(o1+(o2-o1)*0.15),int(nmo),int(o1+(o2-o1)*0.9)] 
                    
                elif(mod[o2,0]<mod[l1,0]):        
                    ros=np.array([ros2,ros1]); imp=np.array([imp2,imp1]); fin=np.array([fin2,fin1]); lim=np.array([limbl,limbs])    
                    sft=-mod[o2,0]+mod[l1,0]; stat=2
                    tt=[int(o1+(o2-o1)*0.1),int(nmo),int(o1+(o2-o1)*0.85),int(l1),int(l1+(l2-l1)*0.15),int(nml),int(l1+(l2-l1)*0.9)]
                else:  
                    print(l1, l2, o1, o2, mod[l1,0], mod[l2,0], mod[o1,0], mod[o2,0] , count)                   
                    input("Error: o1, o2, l1, l2 ")
                    
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
                shift=0.0
                for j in range(nk):
                    if(dat[j,0]>=mod[l1,0]   and dat[j,0]<=float(mod[l2,0]+cadence)):   shift=0;
                    elif(dat[j,0]>=mod[o1,0] and dat[j,0]<=float(mod[o2,0]+cadence)): shift=sft;
                    else: 
                        print("time: ", dat[:nk,0], dat[j,0], j, nk)
                        print(mod[l1:l2,0],    mod[o1:o2,0]  )
                        input("Data is out of the range")
                    plt.errorbar(dat[j,0]+shift,dat[j,1],yerr=dat[j,2],fmt=".",markersize=9,color='m',ecolor='gray',elinewidth=0.3, capsize=0,alpha=0.75)
                plt.title(
                r"$M_{1}(M_{\odot})=$"+'{0:.1f}'.format(MBH)+
                r"$;~M_{2}(M_{\odot})=$"+'{0:.1f}'.format(mass)+
                r"$;~T(\rm{days})=$"+'{0:.1f}'.format(period)+
                r"$;~\mathcal{F}=$"+'{0:.2f}'.format(ratio)+
                #r"$;~i(\rm{deg})=$"+'{0:.2f}'.format(inc)+"\n"+
                r"$;~m_{\rm{b}}(\rm{mag})=$"+'{0:.1f}'.format(base)+"\n"+
                r"$\Delta t/\tau=$"+'{0:.1f}'.format(dur)+
                r"$;~\Delta\rm{F}/$"+r"${\sigma}_{\rm{A}}=$"+'{0:.1f}'.format(depth/errA)+
                r"$;~\rho_{\star}=$"+'{0:.1f},{1:.1f}'.format(ros[0],ros[1])+
                r"$;~\rho_{\rm{l}}=$"+'{0:.2f},{1:.2f}'.format(fin[0],fin[1])+
                r"$;~u_{0}/\rho_{\star}=$"+'{0:.1f},{1:.1f}'.format(imp[0],imp[1]),fontsize=16.0,color='k')
                if(stat==1): 
                    plt.xlim(mod[l1,0], mod[o2,0]+sft)
                    ticc=np.array([mod[tt[0],0],mod[tt[1],0],mod[tt[2],0],mod[tt[3],0],mod[tt[4],0]+sft,mod[tt[5],0]+sft,mod[tt[6],0]+sft])
                if(stat==2):  
                    plt.xlim(mod[o1,0]+sft, mod[l2,0])
                    ticc=np.array([mod[tt[0],0]+sft,mod[tt[1],0]+sft,mod[tt[2],0]+sft,mod[tt[3],0],mod[tt[4],0],mod[tt[5],0], mod[tt[6],0]])
                ax.set_xticks(ticc,labels=labs)
                rcParams['xtick.major.pad']='-3.0'
                #y_vals=ax.get_yticks()
                #if(abs(np.max(y_vals)-np.min(y_vals))<0.0001): 
                #    plt.ylim(1.0-0.0005, 1.0+0.0005)
                plt.ylabel(r"$\rm{Normalized}~\rm{Flux}$",fontsize=20)
                plt.xlabel(r"$\rm{time}(\rm{days})$",fontsize=20)
                plt.xticks(fontsize=17, rotation=30)
                plt.yticks(fontsize=17, rotation=0)
                legend=ax.legend(prop={"size":12.5},loc='best',frameon=True, fancybox = True,shadow=True,framealpha=0.5)
                legend.get_frame().set_facecolor('whitesmoke')
                dd=ax.legend(title=Stat)##,prop={"size":14.0})
                plt.setp(legend.get_title(),color=col)
                fig=plt.gcf()
                fig.tight_layout()  
                fig.savefig("./figs/LightCurveB{0:d}.jpg".format(count),dpi=200)
            print("Light curve is plotted:===========  ", count)
            '''
        #else: 
        #    print("No data file ", Flags)                
        #print("HHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHH")
print("Efficiecy:  ", float(ndet*100.0/nf))

###############################################################################
ndet=int(ndet)
col=['b','g']# Lensing(0); Occultation(1)
co=[]
for i in range(int(ndet)): 
    co.append(col[int(pard[i,3])])
    
    
    
    
plt.cla()
plt.clf()
fig = plt.figure(figsize=(8,6))
gs = GridSpec(4,4)
ax_sc = fig.add_subplot(gs[1:4,0:3])
ax_hy = fig.add_subplot(gs[0:1,0:3], sharex=ax_sc)
ax_hx = fig.add_subplot(gs[1:4,3], sharey=ax_sc)

ax_sc.scatter(par[:,17],par[:,6],marker='o',c='k',s=7.5,label=r"$\rm{All}~\rm{Events}$")
ax_sc.scatter(pard[:ndet,17],pard[:ndet,6],marker='*',c=co[:ndet],s=10.5,label=r"$\rm{Detectable}~\rm{Events}$")
ax_sc.set_xlabel(r"$M_{\rm{l}}(M_{\odot})$",fontsize=19)
ax_sc.set_ylabel(r"$M_{\star}(M_{\odot})$",fontsize=19)
#tic=np.array([ 0.3, 0.5, 0.8, 1.0 ,1.2 ])
#ax_sc.set_xticks(tic , labels=np.round(tic,1)) 
#ax_sc.set_yticks(tic , labels=np.round(tic,1)) 
ax_sc.set_xlim([0.17, 1.4])
ax_sc.set_ylim([0.17, 1.4])
plt.subplots_adjust(wspace=-20.0)
plt.subplots_adjust(hspace=-20.0)
legend=ax_sc.legend(prop={"size":14}, loc='best' ,fancybox=True, shadow=True,framealpha=0.5)
legend.get_frame().set_facecolor('whitesmoke')
##########################################
ax_hy.hist( par[:,17],35,    density=False,histtype='bar',ec='k',facecolor='k', alpha=0.5,rwidth=1.0)
ax_hy.hist(pard[:ndet,17],35,density=False,histtype='bar',ec='m',facecolor='m', alpha=0.5,rwidth=1.0)
y_vals=ax_hy.get_yticks()
ax_hy.set_yticks(y_vals)
ax_hy.set_yticklabels(['{:.2f}'.format(float(1.0*x*(1.0/nf))) for x in y_vals]) 
y_vals=ax_hy.get_yticks()
plt.ylim([np.min(y_vals), np.max(y_vals)])
#ax_hy.set_xticks(tic,labels=np.round(tic,1)) 
#ax_hy.set_xlim([0.17, 1.4])
plt.subplots_adjust(wspace=-10.0)
plt.subplots_adjust(hspace=-10.0)
##########################################
ax_hx.hist( par[:,6],35,    density=False,histtype='bar',ec='k',facecolor='k',alpha=0.5,rwidth=1,orientation ='horizontal')
ax_hx.hist(pard[:ndet,6],35,density=False,histtype='bar',ec='m',facecolor='m',alpha=0.5,rwidth=1,orientation ='horizontal')
y_vals =ax_hx.get_yticks()
ax_hx.set_yticks(y_vals)
ax_hx.set_yticklabels(['{:.2f}'.format(float(1.0*x*(1.0/nf))) for x in y_vals]) 
y_vals=ax_hx.get_yticks()
plt.ylim([np.min(y_vals), np.max(y_vals)])
#ax_hx.set_xticks(tic,labels=np.round(tic,1)) 
#ax_hx.set_xlim([0.17, 1.4])
#ax_hx.set_xlim([0.17, 1.4])
plt.subplots_adjust(wspace=-2.0)
plt.subplots_adjust(hspace=-2.0)
##########################################
plt.xticks(fontsize=16, rotation=0)
plt.yticks(fontsize=16, rotation=0)
fig.tight_layout(pad=0.1)
fig=plt.gcf()
fig.savefig("./HistoTESS/AMaps1.jpg", dpi=200)







###############################################################################
par[:,1]=par[:,17]+par[:,6]  # Ml+Mstar
par[:,2]=par[:,17]/par[:,6]  # Ml/Mstar

pard[:ndet,1]=pard[:ndet,17]+pard[:ndet,6]  # Ml+Mstar
pard[:ndet,2]=pard[:ndet,17]/pard[:ndet,6]  # Ml/Mstar


plt.cla()
plt.clf()
fig = plt.figure(figsize=(8,6))
gs = GridSpec(4,4)
ax_sc = fig.add_subplot(gs[1:4,0:3])
ax_hy = fig.add_subplot(gs[0:1,0:3], sharex=ax_sc)
ax_hx = fig.add_subplot(gs[1:4,3], sharey=ax_sc)
ax_sc.scatter(par[:,1],par[:,2],marker='o',c='k',s=9.5,label=r"$\rm{All}~\rm{Events}$")
ax_sc.scatter(pard[:ndet,1],pard[:ndet,2],marker='*',c=co[:ndet],s=10.5,label=r"$\rm{Detectable}~\rm{Events}$")
ax_sc.set_xlabel(r"$M_{\rm{l}}(M_{\odot})+M_{\star}(M_{\odot})$",fontsize=19)
ax_sc.set_ylabel(r"$q$",fontsize=19)
ticx=np.array([ 0.3*2, 0.5*2, 0.8*2, 1.0*2 ,1.2*2 ])
ticy=np.array([0.3, 0.5, 0.7, 0.9, 1.0])
ax_sc.set_xticks(ticx , labels=np.round(ticx,1)) 
ax_sc.set_yticks(ticy , labels=np.round(ticy,1)) 
ax_sc.set_xlim([0.17*2, 1.4*2])
ax_sc.set_ylim([0.0, 1.])
plt.subplots_adjust(wspace=-20.0)
plt.subplots_adjust(hspace=-20.0)
legend=ax_sc.legend(prop={"size":15}, loc=4 ,fancybox=True, shadow=True)
legend.get_frame().set_facecolor('whitesmoke')
##########################################
ax_hy.hist( par[:,1],30,    density=False,histtype='bar',ec='k',facecolor='k', alpha=0.5,rwidth=1.0)
ax_hy.hist(pard[:ndet,1],30,density=False,histtype='bar',ec='m',facecolor='m', alpha=0.5,rwidth=1.0)
y_vals=ax_hy.get_yticks()
ax_hy.set_yticks(y_vals)
ax_hy.set_yticklabels(['{:.2f}'.format(float(1.0*x*(1.0/nf))) for x in y_vals]) 
y_vals=ax_hy.get_yticks()
plt.ylim([np.min(y_vals), np.max(y_vals)])
ax_hy.set_xticks(ticx,labels=np.round(ticx,1)) 
ax_hy.set_xlim([0.17*2, 1.4*2])
plt.subplots_adjust(wspace=-10.0)
plt.subplots_adjust(hspace=-10.0)
##########################################
ax_hx.hist( par[:,2],30,density=False,histtype='bar',ec='k',facecolor='k',alpha=0.5,rwidth=1,orientation ='horizontal')
ax_hx.hist(pard[:ndet,2],30,density=False,histtype='bar',ec='m',facecolor='m',alpha=0.5,rwidth=1,orientation ='horizontal')
y_vals =ax_hx.get_xticks()
ax_hx.set_xticks(y_vals)
ax_hx.set_xticklabels(['{:.2f}'.format(float(1.0*x*(1.0/nf))) for x in y_vals]) 
y_vals=ax_hx.get_yticks()
plt.ylim([np.min(y_vals), np.max(y_vals)])
ax_hx.set_yticks(ticy,labels=np.round(ticy,1)) 
#ax_hx.set_xlim([0.17, 1.4])
plt.subplots_adjust(wspace=-2.0)
plt.subplots_adjust(hspace=-2.0)
##########################################
plt.xticks(fontsize=16, rotation=0)
plt.yticks(fontsize=16, rotation=0)
fig.tight_layout(pad=0.1)
fig=plt.gcf()
fig.savefig("./HistoTESS/AMaps3.jpg", dpi=200)


################################################################################
#par[:,27]=np.log10(par[:,27])## period
#pard[:ndet,27]=np.log10(pard[:ndet,27])

#par[:,4]=np.log10(par[:,4]) #impact_parameter
#pard[:ndet,4]=np.log10(pard[:ndet,4])

plt.cla()
plt.clf()
fig = plt.figure(figsize=(8,6))
gs = GridSpec(4,4)
ax_sc=fig.add_subplot(gs[1:4,0:3])
ax_hy=fig.add_subplot(gs[0:1,0:3],sharex=ax_sc)
ax_hx=fig.add_subplot(gs[1:4,3],  sharey=ax_sc)

ax_sc.scatter(par[:,4],par[:,27],marker='o',c='k',s=7.5,label=r"$\rm{All}~\rm{Events}$")
ax_sc.scatter(pard[:ndet,4],pard[:ndet,27],marker='*',c=co[:ndet],s=10.5,label=r"$\rm{Detectable}~\rm{Events}$")
ax_sc.set_xlabel(r"$\log_{10}[u_{0}/\rho_{\star}]$",fontsize=18)
ax_sc.set_ylabel(r"$\log_{10}[T(\rm{days})]$",fontsize=18)
plt.subplots_adjust(wspace=-20.0)
plt.subplots_adjust(hspace=-20.0)
#ticx=np.array([ -2.0, -1.0, 0.0, 1.0 , 1.5 ])
#ticy=np.array([ 0.3, 0.8, 1.2, 1.5, 1.8 ])
#ax_sc.set_xticks(ticx,labels=np.round(ticx,1)) 
#ax_sc.set_yticks(ticy,labels=np.round(ticy,1)) 
#ax_sc.set_ylim([0.0 ,2.0])
#ax_sc.set_xlim([-2.2,2.0])
legend=ax_sc.legend(prop={"size":15}, loc=4 ,fancybox=True, shadow=True)
legend.get_frame().set_facecolor('whitesmoke')



ax_hy.hist( par[:,4],35,    density=True,histtype='bar',ec='k',facecolor='k', alpha=0.5,rwidth=1.0)
ax_hy.hist(pard[:ndet,4],35,density=True,histtype='bar',ec='m',facecolor='m', alpha=0.5,rwidth=1.0)
#y_vals=ax_hy.get_yticks()
#ax_hy.set_yticks(y_vals)
#ax_hy.set_yticklabels(['{:.2f}'.format(float(1.0*x*(1.0/nf))) for x in y_vals]) 
y_vals=ax_hy.get_yticks()
plt.ylim([np.min(y_vals), np.max(y_vals)])
#ax_hy.set_xticks(ticx,labels=np.round(ticx,1) ) 
#ax_hy.set_xlim([-2.2,2.0])
plt.subplots_adjust(wspace=-10.0)
plt.subplots_adjust(hspace=-10.0)
#ax_hy.sharex(ax_sc)
#plt.tick_params('x', labelbottom=False)
#ax_hy.set_xticklabels([])



ax_hx.hist(par[:,27],35,     density=True,histtype='bar',ec='k',facecolor='k',alpha=0.5,rwidth=1,orientation ='horizontal')
ax_hx.hist(pard[:ndet,27],35,density=True,histtype='bar',ec='m',facecolor='m',alpha=0.5,rwidth=1,orientation ='horizontal')
#y_vals =ax_hx.get_xticks()
#ax_hx.set_xticks(y_vals)
#ax_hx.set_xticklabels(['{:.2f}'.format(float(1.0*x*(1.0/nf))) for x in y_vals]) 
y_vals=ax_hx.get_yticks()
plt.ylim([np.min(y_vals), np.max(y_vals)])
#ax_hx.set_yticks(ticy,labels=np.round(ticy,1) ) 
#ax_hx.set_xlim([0.0,2.0])
plt.subplots_adjust(wspace=-2.0)
plt.subplots_adjust(hspace=-2.0)
#ax_hx.sharey(ax_sc)
#plt.tick_params('y', labelbottom=False)

plt.xticks(fontsize=16, rotation=0)
plt.yticks(fontsize=16, rotation=0)
#plt.legend(prop={"size":15}, loc='best')
#plt.subplots_adjust(wspace=.0)
#plt.subplots_adjust(hspace=.0)
fig.tight_layout(pad=0.1)
fig=plt.gcf()
fig.savefig("./HistoTESS/AMaps2.jpg", dpi=200)
print("****  two maps are plotted **************")
################################################################################
nam=[r"$\log_{10}[\rm{SNR}]$", r"$\log_{10}[\Delta F]$", r"$\rm{CDPP}$", r"$\log_{10}[N_{\rm{tran}}]$", r"$\log_{10}[\sigma_{\rm{A}}]$", r"$\rm{Duration}/\rm{Cadence}$", r"$u_{0}/\rho_{\star}$",r"$\rho_{\star}$", r"$m_{\rm{base}}(\rm{mag})$", r"$\sigma_{\rm{m}}(\rm{mag})$",r"$\rm{Detection}~\rm{Flag}$"]

##[count, np.log10(snr), np.log10(depth), cdpp, ntran, np.log10(errA), dur/cadence, impact, rhos, base, errorM, flag])    
## 0       1                  2             3     4      5                  6           7    8,   9 ,   10      11
#dete[:,1]=np.log10(dete[:,1])
#dete[:,2]=np.log10(dete[:,2])
#dete[:,3]=np.log10(dete[:,3])
#dete[:,5]=np.log10(dete[:,5])
#dete[:,6]=np.log10(dete[:,6])
#dete[:,7]=np.log10(dete[:,7])
j=0; 
dete2=np.zeros((nf,12))
for i in range(nf):
    if(dete[i,11]>0):  
        dete2[j,:]=dete[i,:]
        j+=1 

for ii in range(11):
    i=ii+1
    plt.clf()
    plt.cla()
    fig=plt.figure(figsize=(8,6))
    ax= plt.gca()              
    plt.hist(dete[:,i],  density=True,bins=30,histtype='bar',ec='k',facecolor='k',alpha=0.4,rwidth=1.0,label=r"$\rm{All}~\rm{Simulated}~\rm{Targets}$")
    plt.hist(dete2[:j,i],density=True,bins=30,histtype='step',color='m',lw=1.9,ls='-',label=r"$\rm{Detectable}~\rm{Targets}$")
    #y_vals =ax.get_yticks()
    #ax.set_yticks(y_vals)
    #ax.set_yticklabels(['{:.2f}'.format(float(1.0*x*(1.0/nf))) for x in y_vals]) 
    y_vals = ax.get_yticks()
    plt.ylim([np.min(y_vals), np.max(y_vals)])
    ax.set_ylabel(r"$\rm{Normalized}~\rm{Distribution}$",fontsize=18,labelpad=0.1)
    ax.set_xlabel(str(nam[ii]),fontsize=18,labelpad=0.1)
    plt.xticks(fontsize=18, rotation=0)
    plt.yticks(fontsize=18, rotation=0)
    plt.grid("True")
    plt.grid(linestyle='dashed')
    if(ii==0): 
        plt.legend()
        plt.legend(loc='best',fancybox=True, shadow=True)
        plt.legend(prop={"size":18})
    fig=plt.gcf()
    fig.tight_layout()
    fig.savefig("./HistoTESS/HCTLFF{0:d}.jpg".format(i),dpi=200)
    print("****  detect params are plotting **************", i)


###############################################################################
plt.cla() 
plt.clf()
plt.figure(figsize=(8,6))
plt.plot(dete[:,9],dete[:,10],"ro", alpha=0.95)
plt.xticks(fontsize=17,rotation=0)
plt.yticks(fontsize=17,rotation=0)
plt.yscale('log')
plt.xlabel(r"$m_{\rm{base}}(\rm{mag})$", fontsize=18)
plt.ylabel(r"$\sigma_{\rm{m}}(\rm{mag})$", fontsize=18)
plt.grid("True")
plt.legend()
fig=plt.gcf()
fig.savefig("./HistoTESS/ErrorMag.jpg", dpi=200)

###############################################################################



nam0=["count", "l.num", "s.num",  "s.lat", "s.lon", "Dist1", "s.mass", "s.Teff", "s.Rstar", "s.limb", "s.Lumi", "s.Logg", "s.Map", "s.Mab", "s.magb", "s.blend", "s.Ai", "l.MBH", "l.RBH", "l.Lumi", "l.Logg", "l.Teff" , "l.Map", "l.Mab" , "l.inc", "l.tet", "l.ecen", "l.period", "log10(l.a/(s.Rstar*Rsun))", "l.tp", "l.ratio", "s.cdpp", "l.limb"] #33
   
for i in range(nc):
    plt.clf()
    fig= plt.figure(figsize=(8,6))
    ax= plt.gca()              
    plt.hist(par[:,i],30,histtype='bar',ec='darkgreen',facecolor='green',alpha=0.6, rwidth=1.5)
    plt.hist(pard[:ndet,i],30,histtype='bar',ec='darkred',facecolor='red',alpha=0.4, rwidth=1.5)
    y_vals = ax.get_yticks()
    ax.set_yticks(y_vals)
    ax.set_yticklabels(['{:.2f}'.format(float(1.0*x*(1.0/nf))) for x in y_vals]) 
    y_vals = ax.get_yticks()
    plt.ylim([np.min(y_vals), np.max(y_vals)])
    ax.set_ylabel(r"$\rm{Normalized}~\rm{Distribution}$",fontsize=19,labelpad=0.1)
    ax.set_xlabel(str(nam0[i]),fontsize=19,labelpad=0.1)
    plt.xticks(fontsize=17, rotation=0)
    plt.yticks(fontsize=17, rotation=0)
    plt.grid("True")
    plt.grid(linestyle='dashed')
    fig=plt.gcf()
    fig.savefig("./HistoTESS/hsimC{0:d}.jpg".format(i),dpi=200)
    print ("**** histos are plotted ***********************" , i)   

#############################################################










