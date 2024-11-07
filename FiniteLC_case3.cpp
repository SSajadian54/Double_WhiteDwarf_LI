#include <stdlib.h>
#include <stdio.h>
#include <string>
#include <math.h>
#include <fstream>
#include <iostream>
#include <sys/time.h>
#include <sys/timeb.h>
#include <cmath>
#include "VBBinaryLensingLibrary.h"
using std::cout;
using std::endl;
using std::cin;

///=============================================================================

const double RA=180.0/M_PI;
const double KP=3.08568025*pow(10.,19); // in meter.
const double G= 6.67384*pow(10.,-11.0);// in [m^3/s^2*kg].
const double velocity=299792458.0;//velosity of light  m/s
const double Msun=1.98892*pow(10.,30.0); //in [kg].
const double Mjupiter=1.898*pow(10,27.0); 
const double Mearth=  5.9722*pow(10.0,24.0);
const double AU=1.495978707*pow(10.0,11.0);
const double Rsun=6.9634*pow(10.0,8.0); ///solar radius [meter]
const double Teffsun= 5772.0;//kelvin
const double Avks=double(8.20922);
///=============================================================================
const int Nli=int(798);  
const double thre=0.001; 
const int    NB=int(1000); 
const int    nbh=int(30);  
const int    Np=106; ///rows in CDPPMAG.txt
const int    Nl=110; /// wavelength TESS throughput 
const int    nx=10551; 
const int    ny=10551;  
const int    nw=1772;///rows in WDCat.dat
const double wave[4]= {0.673,0.7865,0.532,0.797};//https://en.wikipedia.org/wiki/Photometric_system  G, T, BP, RP
const double AlAv[4]= {0.791986524645539, 0.617155245862836  ,1.0386670640894,0.601810722049874};
const double sigma[4]={0.017, 0.02,0.02, 0.02};// G, T, BP, RP  Table (2) Cardeli

//x=1.0/wave    ///https://heasarc.gsfc.nasa.gov/docs/tess/the-tess-space-telescope.html
//y=x-1.82
//a=1.0+0.17699*y-0.50447*y*y-0.02427*y*y*y+0.72085*y*y*y*y+ 0.01979*y*y*y*y*y-0.7753*y*y*y*y*y*y+0.32999*y*y*y*y*y*y*y
//b=1.41338*y+2.28305*y*y+1.07233*y*y*y-5.38434*y*y*y*y-0.62251*y*y*y*y*y+5.3026*y*y*y*y*y*y-2.09002*y*y*y*y*y*y*y
//AlAv=a+b/3.1

///=============================================================================

struct source{
    int num;  
    double Ds;
    double nsbl, blend, magb, Ai, Mab, Map;
    double Teff, Rstar, mass, Logg;
    double ros, limb, lon, lat;
    double magG, magBP, magRP;
    double cdp, Lumi;  
    double tmag[Np], cdpp[Np];
};
struct limbD{
    double Logg[Nli],Tef[Nli], Limb[Nli];  
};

struct lens{
    int num;  
    double ecen, inc, Lumi, tet, tp, period, a; 
    double phi, RE, lon, lat, limb;   
    double magG, magBP, magRP;
    double ratio, q, MBH, RBH, Map, Mab, Teff, Logg;
    double dx,dy,xi,yi,xsi, ysi, xsc, ysc, num0, num1, Dls, Dl;
    double mass[nw], radius[nw], logg[nw], teff[nw], G[nw], BP[nw], RP[nw], dist[nw], lgal[nw], bgal[nw];
    int    flag;  
};
struct extinc{
   double dis[100];///distance
   double Extks[100];///ks-band extinction
   double Aks;
   int    flag; 
};
struct doppler{
  double wave[Nl], throu[Nl]; 
  double waven, Fpl0, Fpl1;  
};
///=============================================================================
int    Extinction(extinc & ex, double, double );
double ErrorTESS(double maga); 
double Interpol(double ds, extinc & ex);
double RandN(double sigma,double);
double RandR(double down, double up);
double Fluxlimb(double limb, double rstar);  
double Kepler(double phi, double ecen); 
double Bessel(int n,double x); 
double CDPPM(source & s, double); 
double DopplerF(doppler & dp, double , double); 
double Fplank(double , double);  
double Ellipsoid(double, double, double, double, double, double);
void   LensEq2(source & s, lens & l, double, double );
void   FiniteLens(source & s, lens & l, double , double , double);
double LimbF(limbD & li, double, double); 
time_t  _timeNow;
unsigned int _randVal;
unsigned int _dummyVal;
FILE * _randStream;
///===========================================================================//
///                                                                           //
///                  Main program                                             //
///                                                                           //
///===========================================================================//	
int main()
{
   time(&_timeNow);
   _randStream = fopen("/dev/urandom", "r");
   _dummyVal = fread(&_randVal, sizeof(_randVal), 1, _randStream);
   srand(_randVal);
   time( &_timeNow);
   printf("START time:   %s",ctime(&_timeNow));
      
   
   VBBinaryLensing vbb;
   vbb.Tol=1.e-5;
   vbb.a1 =0.0;  
   vbb.LoadESPLTable("./ESPL.tbl");
  
  
   source s;
   lens l;
   extinc ex;
   doppler dp;
   limbD li;  
    
    
   FILE* film;  
   FILE* distr;
   FILE* cdppf;
   FILE* dopp;
   FILE* wdcat; 
///HHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHH
   long int ID; 
   char   filenam0[40], filenam2[40];
   double ksi, x0, y0, x1, y1, z1, dis; 
   double phase, u, us, Astar, As, disp, dt, dt0;   
   double Rho, RE, proj, tim, finl, SourceA, Astar2, Roche; 
   double Dist1, Dist2, color, extG, extBP, extRP, Av, emax, paral;  
   int count, self, nstep, Flag;    
   
  

   cdppf=fopen("./files/CDPPMAG.txt","r");
   for(int i=0; i<Np; ++i){
   fscanf(cdppf,"%lf  %lf\n",&s.tmag[i], &s.cdpp[i]);
   if((s.tmag[i]<s.tmag[i-1] and i>0) or (i>0 and s.cdpp[i]<s.cdpp[i-1])){ 
   cout<<"Error counter:  "<<i<<"\t tmag:  "<<s.tmag[i]<<"\t cdpp:  "<<s.cdpp[i]<<endl;   int uue; cin>>uue;}}
   fclose(cdppf); 
   cout<<"********** File CDPPMAG.dat was read ************"<<endl;    


    
  
   dopp=fopen("./files/TESS_Throught.txt","r");
   for(int i=0;  i<Nl;  ++i){
   fscanf(dopp,"%lf  %lf\n",&dp.wave[i], &dp.throu[i]);}
   fclose(dopp); 
   cout<<"********** File TESS_Throught.txt was read ************"<<endl;    
   

   
   
   dopp=fopen("./files/WDlimb.txt","r");
   for(int i=0; i<Nli;  ++i){
   fscanf(dopp,"%lf  %lf  %lf\n",&li.Logg[i], &li.Tef[i], &li.Limb[i]); }
   fclose(dopp);     
   cout<<"********** File TESS_Throught.txt was read ************"<<endl;    
   
  
        
   wdcat=fopen("./files/WDCat.dat","r");
   if(!wdcat){ cout<<"cannot read wdcat.dat "<<endl; exit(0);}
   for(int i=0; i<nw; ++i){//ID, Teff, logg, mass, parallax, G, BP,  RP, radius, lgal, bgal
   fscanf(wdcat,"%d   %lf  %lf  %lf  %lf  %lf  %lf  %lf  %lf  %lf  %lf\n",
   &ID, &l.teff[i], &l.logg[i], &l.mass[i], &paral, &l.G[i], &l.BP[i], &l.RP[i], &l.radius[i], &l.lgal[i], &l.bgal[i]);  
   l.G[i]  -=5.0*log10(100.0/paral);  
   l.BP[i] -=5.0*log10(100.0/paral);  
   l.RP[i] -=5.0*log10(100.0/paral);
   l.dist[i]=double(1.0/paral);}
   fclose(wdcat); 
   cout<<"**************** File WDcat.dat was read **************"<<endl;    
   
   
   
      
   sprintf(filenam0,"./files/light/lcF2/%c%c%d.dat",'C','_',2);
   distr=fopen(filenam0,"a+"); 
   fclose(distr);              
   
///=============================================================================      
   for(count=455; count<=550; ++count){  
   sprintf(filenam2,"./files/light/lcF2/%c%c%d.dat",'M','_',int(count));
   film=fopen(filenam2,"w");  
   
 
   do{
   l.num=int(RandR(0.0,nw-1.0));
   l.MBH= l.mass[l.num];
   l.RBH= l.radius[l.num];
   l.Logg=l.logg[l.num];
   l.Teff=l.teff[l.num]; 
   l.lon= l.lgal[l.num];
   l.lat= l.bgal[l.num];  
   if(l.lon<=0.0) l.lon+=360.0;
   l.Lumi=pow(l.Teff/Teffsun,4.0)*pow(l.RBH,2.0);  
      
   s.num=int(RandR(0.0,nw-1.0));
   s.mass= l.mass[s.num];
   s.Rstar=l.radius[s.num];
   s.Logg= l.logg[s.num]; 
   s.Teff= l.teff[s.num];  
   s.lon=  l.lgal[s.num];
   s.lat=  l.bgal[s.num];  
   if(s.lon<=0.0) s.lon+=360.0;
   s.Lumi=pow(s.Teff/Teffsun,4.0)*pow(s.Rstar,2.0); 

   l.a=RandR(log10(3.0*s.Rstar),log10(10000.0*s.Rstar) );
   l.a=pow(10.0,l.a)*Rsun;
   l.period=sqrt(4.0*M_PI*M_PI/(G*Msun*(s.mass+l.MBH)))*pow(l.a,1.5)/(3600.0*24.0);//days   
   l.q=double(l.MBH/s.mass);
   Roche=l.a-l.a*0.49*pow(l.q,2.0/3.0)/(0.6*pow(l.q,2.0/3.0)+log(1.0+pow(l.q,1.0/3.0)));
   if(Roche<0.0 or l.period<0.0 or l.MBH<0.0 or s.mass<0.0 or s.num<0 or l.num<0 or l.num>nw or s.num>nw){
   cout<<"Error Roche:  "<<Roche/s.Rstar/Rsun<<"\t q:  "<<l.q<<"\t s.num: "<<s.num<<endl; 
   cout<<"l.a:  "<<l.a<<"\t MBH:  "<<l.MBH<<"\t num_l: "<<l.num<<endl; int uue; cin>>uue; }
   }while(Roche<double(s.Rstar*Rsun) or l.period<5.0 or l.period>50.0 or l.MBH>0.25 or s.mass>0.45); 
   s.limb=LimbF(li, s.Teff, s.Logg);
   l.limb=LimbF(li, l.Teff, l.Logg);
     
     
     
   emax=double(0.8-8.0*exp(-pow(6.0*l.period,0.35)));//Fig 7
   if(emax<0.0)  emax=0.0; 
   if(emax>1.0)  emax=1.0;
   if(emax<0.0 or emax>1.0){cout<<"Error emax:  "<<emax<<"\t period:  "<<l.period<<endl; int yyw; cin>> yyw; } 
   //emax=0.1;  
   l.ecen=RandR(0.0,emax);
   l.tp=  RandR(0.0,l.period); 
   l.inc= RandR(0.0,0.2)*M_PI/180.0;//rad
   l.tet= RandR(0.1,359.9)*M_PI/180.0;//rad
  
   
   
   Dist1=double(l.dist[s.num]);  
   ex.flag=-1;
   ex.flag=Extinction(ex, double(s.lon), double(s.lat) );
   if(ex.flag>0) Av=double(Interpol(Dist1,ex)*Avks);
   else          Av=double(0.7*Dist1); 
   if(Av<0.0)    Av=0.0;
   extG= fabs(Av*AlAv[0])+RandN(sigma[0],1.5);//G
   extBP=fabs(Av*AlAv[2])+RandN(sigma[2],1.5);//BP
   extRP=fabs(Av*AlAv[3])+RandN(sigma[3],1.5);//RP
   s.Ai= fabs(Av*AlAv[1])+RandN(sigma[1],1.5);//Tband
   if(s.Ai<0.0)  s.Ai =0.0; 
   if(extG<0.0)  extG =0.0;
   if(extBP<0.0) extBP=0.0;
   if(extRP<0.0) extRP=0.0;
   s.magG=  l.G[s.num]-extG;  
   s.magBP=l.BP[s.num]-extBP;  
   s.magRP=l.RP[s.num]-extRP;  
   color=double(s.magBP-s.magRP); 
   if(color<=6.0 and color>=-1.0)
   s.Mab=s.magG-0.00522555*pow(color,3.0)+0.0891337*pow(color,2.0)-0.633923*color+0.0324473+RandN(0.006,1.5); 
   else s.Mab=s.magG-0.430+RandN(0.6,1.5);
   s.Map=s.Mab+s.Ai+5.0*log10(Dist1*100.0);     
   s.blend=1.0;    
   s.magb=double(s.Map+2.5*log10(s.blend));
   s.nsbl=double(1.0/s.blend);   
   s.cdp=CDPPM(s,s.magb);   
   
   
   Dist2=double(l.dist[l.num]); 
   ex.flag=-1;
   ex.flag=Extinction(ex, double(l.lon), double(l.lat) ); 
   if(ex.flag>0)Av=double(Interpol(Dist2,ex)*Avks);
   else         Av=double(0.7*Dist2); 
   if(Av<0.0)   Av=0.0;
   extG= fabs(Av*AlAv[0])+RandN(sigma[0],1.5);//G
   extBP=fabs(Av*AlAv[2])+RandN(sigma[2],1.5);//BP
   extRP=fabs(Av*AlAv[3])+RandN(sigma[3],1.5);//RP
   if(extG<0.0)  extG =0.0;
   if(extBP<0.0) extBP=0.0;
   if(extRP<0.0) extRP=0.0;
   l.magG=  l.G[l.num]-extG;  
   l.magBP=l.BP[l.num]-extBP;  
   l.magRP=l.RP[l.num]-extRP;  
   color=double(l.magBP-l.magRP); 
   if(color<=6.0 and color>=-1.0)
   l.Mab=l.magG-0.00522555*pow(color,3.0)+0.0891337*pow(color,2.0)-0.633923*color+0.0324473+RandN(0.006,1.5); 
   else l.Mab=l.magG-0.430+RandN(0.6,1.5);
   l.Map=l.Mab+s.Ai+5.0*log10(Dist1*100.0);     
   l.ratio=pow(10.0,-0.4*fabs(l.Map-s.Map)); 
   dt=double(l.period/5500.0);///days
   RE=sqrt(4.0*G*Msun*(l.MBH+s.mass)*0.5*l.a)/velocity+1.0e-50;//meter 
   Rho=double((l.RBH+s.Rstar)*Rsun*0.5/RE);
   l.dx=33.5435*Rho/nx/1.00;//[RE]
   l.dy=33.5435*Rho/ny/1.00;//[RE]
   dt0=dt; Flag=-1;
  
  
  
  
   nstep=0;
   for(tim=0.0;   tim<=l.period; tim+=dt0){
   nstep+=1;
   l.phi=double((tim-l.tp)*2.0*M_PI/l.period); 
   if(l.ecen<0.01) ksi=l.phi;
   else            ksi=Kepler(l.phi , l.ecen);
   x0=l.a*(cos(ksi)-l.ecen);//major axis [m]
   y0=l.a*sin(ksi)*sqrt(1.0-l.ecen*l.ecen);//minor axis [m]
   y1=              y0*cos(l.tet)+x0*sin(l.tet);//[m]
   x1= cos(l.inc)*(-y0*sin(l.tet)+x0*cos(l.tet));//[m]
   z1=-sin(l.inc)*(-y0*sin(l.tet)+x0*cos(l.tet));//[m]
   dis= sqrt(x1*x1 + y1*y1 + z1*z1)+1.0e-50;//[m] 
   disp=sqrt(y1*y1 + z1*z1)+1.0e-50;///meter
   phase=acos(-x1/dis);
   Astar=Astar2=1.0;
   l.num0=l.num1=finl=0.0;
   self=0; 
     
     
     
   if(x1<-0.0001){//Lensing of source star by lens
   l.Dl=Dist1*KP;//meter
   s.Ds=Dist1*KP+fabs(x1);//meter
   l.Dls=fabs(x1);//[m]
   proj=double(l.Dl/s.Ds);
   l.RE=sqrt(4.0*G*Msun*l.MBH)*sqrt(l.Dls*proj)/velocity+1.0e-50;
   s.ros=fabs(s.Rstar*Rsun*proj/l.RE); 
   SourceA=double(M_PI*s.ros*s.ros*(1.0-s.limb/3.0)/(l.dx*l.dy));
   u=fabs(disp/l.RE);
   l.xsc=double(y1/l.RE);  
   l.ysc=double(z1/l.RE);
   if(u<float(25.0*s.ros)){
   self=1;
   if(Flag!=1){dt0=dt/35.0; Flag=1;} 
   if(s.ros>100.0){//Valerio_Magnification
   if(u<s.ros){Astar=double(1.0+2.0/s.ros/s.ros);        }
   else       {Astar=double(u*u+2.0)/sqrt(u*u*(u*u+4.0));}}
   else{vbb.a1=s.limb; Astar=vbb.ESPLMag2(u, s.ros);}
   if(u<float(15.0*s.ros)){
   FiniteLens(s,l, double(l.MBH), double(l.RBH), double(s.limb) );
   finl=   double(l.num1/SourceA);
   Astar2= double(l.num0/SourceA);}
   else{finl=double(u*u+2.0)/(u*sqrt(u*u+4.0))-1.0;}}}
   
   
   
   
   if(x1>0.0001){//Lensing of Lens by source star
   l.Dl=Dist1*KP-fabs(x1);//[m]
   s.Ds=Dist1*KP;//[m]
   l.Dls=fabs(x1);//[m]
   proj=double(l.Dl/s.Ds);
   l.RE=sqrt(4.0*G*Msun*s.mass)*sqrt(l.Dls*proj)/velocity+1.0e-50;
   s.ros=fabs(l.RBH*Rsun*proj/l.RE); 
   SourceA=double(M_PI*s.ros*s.ros*(1.0-l.limb/3.0)/(l.dx*l.dy));
   u=fabs(disp/l.RE);
   l.xsc=double(-y1/l.RE);  
   l.ysc=double(-z1/l.RE);
   if(u<float(25.0*s.ros)){
   self=2;
   if(Flag != 2){dt0=dt/35.0; Flag=2;} 
   if(s.ros>100.0){//Valerio_Magnification
   if(u<s.ros){Astar=double(1.0+2.0/s.ros/s.ros);        }
   else       {Astar=double(u*u+2.0)/sqrt(u*u*(u*u+4.0));}}
   else{vbb.a1=l.limb; Astar=vbb.ESPLMag2(u,s.ros);}
   if(u<float(15.0*s.ros)){
   FiniteLens(s,l, double(s.mass), double(s.Rstar), double(l.limb) );
   finl=   double(l.num1/SourceA);
   Astar2= double(l.num0/SourceA);}
   else{finl=double(u*u+2.0)/(u*sqrt(u*u+4.0))-1.0;}}}
   if(self==0) dt0=dt; 
   
   if(nstep%50==0){  
   cout<<"nstep:  "<<nstep<<"\t dt0: "<<dt0/dt<<"\t period  "<<l.period<<endl;
   cout<<"self: "<<self<<"\t ro_star: "<<s.ros<<"\t u/ros: "<<u/s.ros<<endl;
   cout<<"A_VBB:  "<<Astar<<"\t A_IRS: "<<Astar2<<"\t FinL: "<<finl<<endl;
   cout<<"num0:  "<<l.num0<<"\t num1: "<<l.num1<<"\t sourceA: "<<SourceA<<endl;
   cout<<"u:  "<<u<<"\t RE:   "<<l.RE/(s.Rstar*Rsun)<<"\t inc(deg):  "<<l.inc*RA<<endl;
   cout<<"semi/Rsun: "<<l.a/Rsun<<endl;
   cout<<"*********************************************************"<<endl;}
   if((x1<-0.01 and fabs(phase)>M_PI/2.0) or l.RE<-0.01 or s.ros<-0.01 or dis<-0.01 or x1>dis or disp<-0.01 or 
   phase<-0.00001 or phase>float(M_PI*1.01) or s.limb>1.01 or Astar<0.9 or l.num0<0.0 or s.limb<0.0 or 
   l.limb<0.0 or l.limb>1.01){ 
   cout<<"ERROx1:  "<<x1<<"\t phase: "<<phase<<"\t ros:  "<<s.ros<<endl;
   cout<<"RE:    "<<RE<<"\t dis:  "<<dis<<"disp:  "<<disp<<endl;
   cout<<"u:  "<<u<<"\t Astar:  "<<Astar<<"\t limb:  "<<s.limb<<"\t l.limb: "<<l.limb<<endl;}
  
  
   if(x1<0.0) As=double(Astar-finl+l.ratio)/(1.0+l.ratio); 
   else       As=double(1.0+(Astar-finl)*l.ratio)/(1.0+l.ratio); 
   fprintf(film,"%.4lf   %.6lf   %.6lf  %.6lf  %.4lf  %.4lf  %.4lf  %.5lf  %.5lf  %.4lf  %.4lf  %d\n",
   tim,Astar,Astar2,finl,x1/l.a,y1/l.a,z1/l.a,l.RE/Rsun,s.ros,u/s.ros,disp/l.a, self);//12
   }//time loop
   fclose(film);
   
   
 
   distr=fopen(filenam0,"a+"); 
   fprintf(distr,
   "%d  %d  %d  %.5lf  %.5lf  "///5
   "%.6lf  %.6lf  %.2lf  %.6lf  %.3lf  %.4lf  %.4lf  %.4lf   %.4lf %.4lf  %.4lf  %.4lf  "//17
   "%.6lf  %.6lf  %.6lf  %.5lf  %.5lf  %.5lf  %.5lf  "//24
   "%.5lf  %.5lf  %.6lf  %.7lf  %.8lf  %.6lf  %.8lf  %.5lf   %.5lf \n",//32
   count,l.num, s.num,  s.lat, s.lon, //5
   Dist1, s.mass, s.Teff, s.Rstar, s.limb, s.Lumi, s.Logg, s.Map, s.Mab, s.magb, s.blend, s.Ai,//17
   l.MBH, l.RBH, l.Lumi, l.Logg, l.Teff,l.Map,l.Mab,//24
   l.inc*RA, l.tet*RA, l.ecen, l.period, log10(l.a/(s.Rstar*Rsun)), l.tp, l.ratio, s.cdp, l.limb);//32
   fclose(distr);      
   cout<<"=============================================================="<<endl;
   cout<<"Count:  "<<count<<endl;
   cout<<"RE/Rsun: "<<RE/Rsun<<"\t Rho*: "<<Rho<<"\t ImpactP/R*: "<<sin(l.inc)*l.a/s.Rstar/Rsun<<endl;
   cout<<"latit:  "<<s.lat<<"\t longt: "<<s.lon<<"\t nstep:  "<<nstep<<endl;
   cout<<"l.dx:   "<<l.dx<<"\t l.dy:  "<<l.dy<<endl;
   cout<<"s.Lumi: "<<s.Lumi<<"\t l.Lumi:  "<<l.Lumi<<"\t Ds:  "<<Dist1<<endl;
   cout<<"inc(deg):    "<<l.inc*RA<<"\t  teta(deg):  "<<l.tet*RA<<"\t dt:  "<<dt<<endl;
   cout<<"l.Mab:  "<<l.Mab<<"\t l.Map: "<<l.Map<<"\t ratio:  "<<l.ratio<<endl;
   cout<<"s.Mab:  "<<s.Mab<<"\t s.Map: "<<s.Map<<"\t period: "<<l.period<<endl;
   cout<<"l.MBH:  "<<l.MBH<<"\t l.RBH: "<<l.RBH<<"\t semi/Rsun: "<<l.a/Rsun<<endl;
   cout<<"mass:   "<<s.mass<<"\t s.Rstar:  "<<s.Rstar<<"\t blend:  "<<s.blend<<endl;
   cout<<"l.a/AU: "<<l.a/AU<<"\t Ratio:  "<<l.ratio<<"\t ecen:  "<<l.ecen<<endl;
   cout<<"=============================================================="<<endl;
   }     
   fclose(_randStream);
   return(0);
}
///HHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHH
double CDPPM( source & s ,  double Mags){

   double cdp=-1.0, shib;  

   if(Mags<s.tmag[0])  cdp=s.cdpp[0];  
   else if(Mags>=s.tmag[Np-1]){
   shib=double(s.cdpp[Np-1]-s.cdpp[Np-2])/(s.tmag[Np-1]-s.tmag[Np-2]);  
   cdp=double(s.cdpp[Np-1]+shib*(Mags-s.tmag[Np-1]));}
   else{
   for(int i=1; i<Np; ++i){
   if((Mags-s.tmag[i])*(Mags-s.tmag[i-1])<=0.0) {
   shib=double(s.cdpp[i]-s.cdpp[i-1])/(s.tmag[i]-s.tmag[i-1]);  
   cdp=s.cdpp[i-1]+shib*(Mags-s.tmag[i-1]); 
   break;}}}
   if(cdp<0.0 or cdp>100000.0 or cdp<1.0){
   cout<<"Error   cdpp:  "<<cdp<<"\t Mags:  "<<Mags<<endl; int uue; cin>>uue;} 
   return(cdp); 
}
///HHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHH
double LimbF(limbD & li, double tstar, double logg){
    int num=-1,num2=-1,i;double distm, dist;
    
    if(logg>li.Logg[Nli-1])   num=int(Nli-1); 
    else if(logg<li.Logg[0])  num=0;  
    else{
    for(int j=1; j<Nli; ++j){
    if(double((logg-li.Logg[j])*(logg-li.Logg[j-1]))<=0.0){num=int(j-1); break;}}}

    distm=100000000.0; 
    for(int j=-20; j<20; ++j){
    i=int(num+j);
    if(i>=0 and i<Nli){
    dist=sqrt(pow(tstar-li.Tef[i],2.0)+pow(logg-li.Logg[i],2.0)); 
    if(dist<distm){distm=dist;  num2=i;}}}
    
    
    if(num<0 or num2<0 or num>=Nli or num2>=Nli or li.Limb[num2]<0.0 or 
    li.Limb[num2]>1.0 or fabs(tstar-li.Tef[num2])>2500){
    cout<<"Error num: "<<num<<"\t  num2:  "<<num2<<endl;  
    cout<<"TeffC:    "<<li.Tef[num2]<<"\tLoggC:  "<<li.Logg[num2]<<endl;
    cout<<"Tstar:   "<<tstar<<"\t logg:  "<<logg<<endl;  
    int yye;  cin>>yye;}
    
    cout<<"Tstar:   "<<tstar<<"\t       logg:  "<<logg<<endl;
    cout<<"TeffC:   "<<li.Tef[num2]<<"\tLoggC: "<<li.Logg[num2]<<endl;
    cout<<"limb_Drakening:   "<<li.Limb[num2]<<"\tnum2: "<<num2<<endl;
    cout<<"****************************************************"<<endl;
    return(li.Limb[num2]);  
}
///HHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHH
double Kepler(double phi, double ecen){  
    double ksi=0;   
    double term, term0;  
    phi=double(phi*RA); 
    while(phi>360.0) phi=phi-360.0; 
    while(phi<0.0)   phi=phi+360.0;      
    if(phi>180)      phi=double(phi-360.0);
    if(phi<-181.0 or phi>181.0){ 
    cout<<"Error :  Phi:  "<<phi<<"\t ecent:  "<<ecen<<endl;   int yye;  cin>>yye;}
    phi=double(phi/RA);
    ksi=phi; 
    for(int i=1; i<NB; ++i){
    term= Bessel(i,i*ecen)*sin(i*phi)*2.0/i;  
    ksi+=term; 
    if(i==1) term0=fabs(term); 
    if(fabs(term)<double(thre*term0) and i>5)  break;}        
    return(ksi); 
}    
///#############################################################################
double ErrorTESS(double maga){
   double emt=-1.0, m;     
   
   if(maga<7.5)       emt=double(0.22*maga-5.850); 
   else if(maga<12.5) emt=double(0.27*maga-6.225);  
   else               emt=double(0.31*maga-6.725);    
   emt=emt+RandN(0.1,3.0);
   if(emt<-5.0) emt=-5.0;  
   emt=pow(10.0,emt);
   if(emt<0.00001 or emt>0.5 or maga<0.0){
   cout<<"Error emt:  "<<emt<<"\t maga:  "<<maga<<endl;}
   return(emt); 
}
///#############################################################################
double RandN(double sigma, double nn){
   double rr,f,frand;
   do{
   rr=RandR(-sigma*nn , sigma*nn); ///[-N sigma:N sigma]
   f= exp(-0.5*rr*rr/(sigma*sigma));
   frand=RandR(0.0 , 1.0);
   }while(frand>f);
   return(rr);
}
///#############################################################################
double RandR(double down, double up){
   double p =(double)rand()/((double)(RAND_MAX)+(double)(1.0));
   return(p*(up-down)+down);
}
///#############################################################################
double Fluxlimb(double limb, double rstar){
    return ( double(1.0-limb*(1.0-sqrt(fabs(1.0-rstar*rstar)))) );
}
///#############################################################################
double Bessel(int n,double x){
    double j1=0.00000001,tet;
    int kmax=10000;
    for(int k=0; k<kmax; ++k){
    tet=double(k*M_PI/kmax);
    j1+=double(M_PI/kmax/1.0)*cos(n*tet-x*sin(tet)); }
    return(j1/M_PI);
}    
///#############################################################################
void FiniteLens(source & s, lens & l, double lmass, double lradius, double limbs){
    double diss, mu, Fsour;  
    l.num0=0.0; l.num1=0.0;
    for(int i=0; i<nx; ++i){
    for(int j=0; j<ny; ++j){
    l.xi=float(i-nx/2.0)*l.dx*l.RE;//[m]
    l.yi=float(j-ny/2.0)*l.dy*l.RE;//[m]
    LensEq2(s,l, lmass, lradius);
    diss=sqrt((l.xsi-l.xsc)*(l.xsi-l.xsc)+(l.ysi-l.ysc)*(l.ysi-l.ysc));//##[RE]
    if(diss<s.ros or diss==s.ros){
    mu=sqrt(fabs(1.0-diss*diss/(s.ros*s.ros)));   
    Fsour=1.0-limbs*(1.0-mu);//-s.limb2*(1.0-mu)*(1.0-mu); 
    l.num0+=Fsour;//area_of_images
    if(l.flag<1) l.num1+=Fsour;//part of images' area occultated
    if(Fsour<0.0 or mu>1.0 or s.ros<=0.0 or diss<0.0 or l.flag<0){ 
    cout<<"diss:  "<<diss<<"\t ros:  "<<s.ros<<"\t mu:  "<<mu<<"\t Fosour:  "<<Fsour<<endl;
    cout<<"num0:  "<<l.num0<<"\t num1:  "<<l.num1<<"\t flag: "<<l.flag<<endl; 
    cout<<"xsi:  "<<l.xsi<<"\t xsc:  "<<l.xsc<<endl;
    cout<<"ysi:  "<<l.ysi<<"\t ysc:  "<<l.ysc<<endl;  int uue;    cin>>uue; }
    }}}  
} 
//////##########################################################################
void LensEq2(source & s , lens & l, double lmass, double lradius){
    l.flag=-1; 
    double b, tanb, beta, xs0, ys0, d2, angle, tant, ttm;  
    
    b=    sqrt(l.xi*l.xi+l.yi*l.yi)+1.0e-50;//Impact parameter[m]
    angle=fabs(4.0*G*lmass*Msun/(velocity*velocity*b));//#radian  
    tant= double(b/l.Dl);
    ttm=  double(tan(angle)-tant)/(1.0+tan(angle)*tant );//tan(alfa-teta)
    tanb=tant -(tant + ttm)*l.Dls/s.Ds;
        
    beta= double(tanb*s.Ds);//#[m] in lens plane
    l.xsi=double(beta*l.xi/b)/l.RE;//[RE]
    l.ysi=double(beta*l.yi/b)/l.RE;//[RE]
    d2=   double(b*b/(l.RE*l.RE)); 
    xs0=  double(l.xi/l.RE-l.xi/l.RE/d2);
    ys0=  double(l.yi/l.RE-l.yi/l.RE/d2);
    if(double(angle*180.0/M_PI)>10.0 or fabs(l.xsi-xs0)>2.1 or fabs(l.ysi-ys0)>2.1){
    cout<<"new xs:  "<<l.xsi<<"\t ys:   "<<l.ysi<<endl;
    cout<<"old xs0: "<<xs0<<"\t   ys0: "<<ys0<<endl;
    cout<<"alpha: "<<angle*180.0/M_PI<<"\t b: "<<b<<endl;
    cout<<"xi/RE:  "<<l.xi/l.RE<<"\t yi/RE:  "<<l.yi/l.RE<<endl;
    cout<<"Dls:  "<<l.Dls<<"\t Ds(KP):  "<<s.Ds/KP<<"\t Dls/Ds "<<l.Dls/s.Ds<<endl;
    cout<<"lmass:  "<<lmass<<"\t lradius:  "<<lradius<<endl;
    cout<<"RE/Rstar: "<<l.RE/(lradius*Rsun)<<"\t Delta:  "<<fabs(tan(angle)-angle)<<endl;
    cout<<"tanb:  "<<tanb<<"\t tant:   "<<tant<<endl;}
    //int yye; cin>>yye;}
    if(b<double(lradius*Rsun) or b==(lradius*Rsun)) l.flag=0;///##Occultation
    else                                             l.flag=1;
    //cout<<"LENSEQ xsi:  "<<l.xsi<<"\t xsc:  "<<l.xsc<<endl;
    //cout<<"ysi:  "<<l.ysi<<"\t ysc:  "<<l.ysc<<endl;
}
///#############################################################################
///==============================================================//
///                                                              //
///                  EXtinction                                 //
///                                                              //
///==============================================================//
int Extinction(extinc & ex,  double lon, double lat )
{
   double sig,Lon,Lat;
   int uue, flag=0;
   if(lon<0.0){ sig=-1.0;cout<<"Strange!!!!longtitude is negative: lon "<<lon<<endl; cin>>uue;}
   else sig=1.0;
   double delt=fabs(lon)-floor(fabs(lon));

     
   if(delt>1.0 or delt<0.0){cout<<"ERROR longtitude: delt: "<<delt<<"\t lon: "<<lon<<endl;  cin>>uue; }
   else if(delt<0.25) Lon=(floor(fabs(lon))+0.00)*sig;
   else if(delt<0.50) Lon=(floor(fabs(lon))+0.25)*sig;
   else if(delt<0.75) Lon=(floor(fabs(lon))+0.50)*sig;
   else               Lon=(floor(fabs(lon))+0.75)*sig;
   if(fabs(lon)<0.24999999)      Lon=360.00;
   if(fabs(lon-360.0)<0.2499999) Lon=360.00;




   if(lat<0.0) sig=-1.0;
   else sig=1.0;
   delt=fabs(lat)-floor(fabs(lat));
   if(delt>1.0 or delt<0.0) {cout<<"ERROR latitude: delt: "<<delt<<"\t lon: "<<lat<<endl;  cin>>uue;}
   else if(delt<0.25)  Lat=(floor(fabs(lat))+0.00)*sig;
   else if(delt<0.50)  Lat=(floor(fabs(lat))+0.25)*sig;
   else if(delt<0.75)  Lat=(floor(fabs(lat))+0.50)*sig;
   else                Lat=(floor(fabs(lat))+0.75)*sig;   
   if(fabs(Lon)<0.2499999) Lon=360.00; 
   if(Lat==-0.00)  Lat=0.00;
   if(fabs(Lon)<0.24999999)    Lon=360.00;
   
   
   cout<<"Lon:    "<<Lon<<"\t     Lat:  "<<Lat<<endl;
     
   char filename[40];
   FILE *fpd;
   sprintf(filename,"./files/Ext/%c%c%c%.2lf%c%.2lf.dat",'E','x','t',float(Lat),'_',float(Lon) );
   fpd=fopen(filename,"r");
   if(!fpd){
   cout<<"cannot open (extinction) file long : "<<Lon<<"\t latit: "<<Lat<<endl;
   //FILE *SD;
   //SD=fopen("./files/Ext/saved_direction.txt","r");
   //for(int i=0; i<64881; ++i) {
   //fscanf(SD,"%lf %lf \n",&latit,&lonti);
   //if(fabs(Lat-latit)<0.1 and fabs(Lon-lonti)<0.1){
   //cout<<"ERROR  long : "<<Lon<<"\t latit: "<<Lat<<endl;
   //cout<<"Saved: Latii: "<<latit<<"\t lonti: "<<lonti<<endl; cin>>uue;}}
   flag=-1;}
   else{
   flag=1;
   for(int i=0; i<100; ++i){
   fscanf(fpd,"%lf  %lf\n",&ex.dis[i],&ex.Extks[i]);////Just extinctin in [Ks-band]
   if(ex.dis[i]<0.2  or ex.dis[i]>50.0 or ex.Extks[i]<0.0){
   cout<<"dis: "<<ex.dis[i]<<"\t extI: "<<ex.Extks[i]<<"\t i: "<<i<<endl;
   cout<<"filename: "<<filename<<endl;  ex.Extks[i]=0.0;}}
   fclose(fpd);}
   //cout<<">>>>>>>>>>>>>>> END OF EXTINCTION FUNCTION <<<<<<<<<<<<<<<<<<"<<endl
   return(flag);
}
///#############################################################################
///==============================================================//
///                                                              //
///                  Linear interpolarion                        //
///                                                              //
///==============================================================//
double Interpol(double ds, extinc & ex)
{
  double F=-1.0;
  if(ds<ex.dis[0])        F=ex.Extks[0];
  else if(ds>=ex.dis[99]) F=ex.Extks[99];
  else{ 
  for(int i=0; i<99; ++i){
  if(ex.dis[i]>=ex.dis[i+1]){
  cout<<"ERROR dis[i]: "<<ex.dis[i]<<"\t disi+1: "<<ex.dis[i+1]<<endl;  int yye; cin>>yye; }
  if(ds>=ex.dis[i] and ds<ex.dis[i+1]){
  F = ex.Extks[i]+(ds-ex.dis[i])*(ex.Extks[i+1]-ex.Extks[i])/(ex.dis[i+1]-ex.dis[i]);
  break;}}}
  if(F==-1.0 or F<0.0){cout<<"ERROR big Extinction(ds): "<<F<<"\t ds: "<<ds<<endl; exit(0); }
  return(F);
}
///#############################################################################


