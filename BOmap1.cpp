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

///#############################################################################
const int n1=70;  
const int n2=70;
const double G= 6.67430*pow(10.0,-11.0);
const double AU=1.495978707*pow(10.0,11);
const double Msun=1.989*pow(10.0,30.0);
const double Rsun=6.9634*pow(10.0,8.0);
const double KPC =3.0857*pow(10.0,19.0);
const double velocity=299792458.0;
const double dx=double(0.003903875);
const double dy=double(0.003903875);

///#############################################################################
struct param{
    int    nx, ny, flag, Fla;
    double period, Dl, Rout;  
    double Mlens, Mstar, Rstar, Rlens, semi, RE, Rhol, Rhos, Rin, xsiz, ysiz;  
    double Astar0, Astar1, obscur,image, occult, Fbase, limb, Xsc, Ysc;
    double diss, xi, yi, xs, ys, xs0, ys0, u, ur;          
};
void LensEq2(param & p);
void InRayShoot(param & p); 
 ///############################################################################ 
int main()
{
    VBBinaryLensing vbb;
    vbb.Tol=1.e-5;
    vbb.LoadESPLTable("./ESPL.tbl");
    param p;   
    char filenam0[40];
    FILE* fil0; 
    p.limb=0.45;
    vbb.a1=p.limb;  
    p.Dl=double(1.0*KPC);///[m]
    p.ur=0.0001;
    int peri;
    
    
for(int k=1; k<=50; ++k){    
    peri=int(k*1.0);//period [days]
    sprintf(filenam0,"./BOmaps1/%c%c%c%c%d.dat",'B','m','a','p',peri);
    fil0=fopen(filenam0,"w"); 
    fclose(fil0);              
    p.period=double(peri*24.0*3600.0);
    for(int in=0; in<n1; ++in){//Mstar
        p.Mstar=double(0.17+(1.4-0.17)*in/n1);//[Msun]
        p.Rstar=double(0.01125*sqrt(pow(p.Mstar/1.454,-2.0/3.0)-pow(p.Mstar/1.454,2.0/3.0)));//Run radius 
    for(int jn=0; jn<n2; ++jn){///Lens_type
        p.Mlens=double(0.17+(1.4-0.17)*jn/n2);//[Msun]
        p.Rlens=double(0.01125*sqrt(pow(p.Mlens/1.454,-2.0/3.0)-pow(p.Mlens/1.454,2.0/3.0)));//[Rsun]
        p.semi =pow(p.period*p.period*G*(p.Mlens+p.Mstar)*Msun/(4.0*M_PI*M_PI),1.0/3.0);//#[m]
        p.RE=   sqrt(4.0*G*p.Mlens*Msun)*sqrt(p.Dl*p.semi/(p.Dl+p.semi))/velocity;//[m] 
        p.Rhos= double(p.Rstar*Rsun*p.Dl/(p.RE*(p.Dl+p.semi))); 
        p.Rhol= double(p.Rlens*Rsun/p.RE); 
        p.Fbase=double(M_PI*p.Rhos*p.Rhos*(1.0-p.limb/3.0)/(dx*dy));
        p.u=    double(p.ur*p.Rhos);//[RE]
        p.Rin= fabs(p.Rhos-sqrt(p.Rhos*p.Rhos+4.0))*0.5;//[RE]
        p.Rout=fabs(p.Rhos+sqrt(p.Rhos*p.Rhos+4.0))*0.5;//[RE]
        p.Xsc=p.u;
        p.Ysc=0.0;
        p.xsiz=double(2.0*p.ur+5.5)*p.Rhos;//[RE]
        p.ysiz=double(2.0*p.ur+5.5)*p.Rhos;//[RE]
        p.nx = int(p.xsiz/dx+1.0); 
        p.ny = int(p.ysiz/dy+1.0); 
        InRayShoot(p);
        p.Astar0=vbb.ESPLMag2(p.u, p.Rhos); 
        p.Astar1=double( p.image/p.Fbase); 
        p.occult=double(p.obscur/p.Fbase); 
        fil0=fopen(filenam0,"a+"); 
        fprintf(fil0,"%d %d %.5lf %.4lf %.5lf %.5lf %.4lf %.5lf %.5lf  %.5lf  %.5lf  %.8lf  %.8lf  %.8lf\n",
        in,jn,p.Mstar,p.Rstar,p.Rhos,p.Mlens,p.Rlens,p.Rhol,p.Rin,p.Rout,p.RE/(Rsun*0.01),p.Astar0,p.Astar1,p.occult);//14
        fclose(fil0);
        cout<<"******************************************************"<<endl; 
        cout<<"Number:  "<<p.nx<<"\t size:   "<<p.xsiz<<endl;
        cout<<"Mlens:  "<<p.Mlens<<"\t Rlens:  "<<p.Rlens<<"\t semi(Rsun): "<<p.semi/Rsun<<endl;
        cout<<"RE(Rsun):  "<<p.RE/Rsun<<"\t Rhos: "<<p.Rhos<<"\t p.u: "<<p.u<<endl;
        cout<<"Rhol:  "<<p.Rhol<<"\t Rlens/Rin: "<<p.Rlens*Rsun/(p.RE*p.Rin)<<endl;
        cout<<"Rin:  "<<p.Rin<<"\t Rout: "<<p.Rout<<"\t ur: "<<p.ur<<endl;
        cout<<"Astar0:  "<<p.Astar0<<"\t Astar1:  "<<p.Astar1<<"\t Occul:  "<<p.occult<<endl;
        cout<<"*****************************************************"<<endl; 
        }}
        }
    return(0);
}
//#############################################################################
void LensEq2(param & p){
    double b,angle, tana, tant, tanb, beta;
    b= sqrt(p.xi*p.xi+p.yi*p.yi)+1.0e-50;//#[m] Impact parameter
    angle=double(4.0*G*p.Mlens*Msun/(velocity*velocity*b));//#radian  
    tant= double(b/p.Dl);//## tan(theta)
    tana= double(tan(angle));//## tan(alpha)
    tanb= tant-(tant + (tana-tant)/(1.0+ tant*tana))*p.semi/(p.Dl+p.semi); 
    beta= double(tanb*p.Dl);//#[m]
    p.xs= double(beta*p.xi/b)/p.RE;//#[RE]
    p.ys= double(beta*p.yi/b)/p.RE;//#[RE]
    p.xs0=double(p.xi/p.RE)*(1.0 - 1.0/(b*b/(p.RE*p.RE)));
    p.ys0=double(p.yi/p.RE)*(1.0 - 1.0/(b*b/(p.RE*p.RE)));
    //if(double(angle*180.0/M_PI)>10.0 or fabs(p.xs-p.xs0)>5.0 or fabs(p.ys-p.ys0)>5.0){
    //cout<<"Error deflection angle: "<<double(angle*180.0/M_PI)<<endl;
    //cout<<"new ccordinate : "<<p.xs<<"\t "<<p.ys<<endl;
    //cout<<"old coordinate:  "<<p.xs0<<"\t "<<p.ys0<<endl;}
    p.Fla=-1;
    if(b<double(p.Rlens*Rsun) or b==double(p.Rlens*Rsun)) p.Fla=0;
    else{ p.Fla=1;}
}
//##############################################################################
void InRayShoot(param & p){
    double mu; 
    p.image=0.0;  
    p.obscur=0.0; 
    for(int i=0; i<p.nx;  ++i){
    for(int j=0; j<p.ny;  ++j){
    p.xi=(double(i-p.nx/2.0)*dx+dx/1.996834354+0.00527435617465)*p.RE;//[m]
    p.yi=(double(j-p.ny/2.0)*dy+dy/1.996834354+0.00527435617465)*p.RE;//[m]
    LensEq2(p);
    p.diss=sqrt((p.xs-p.Xsc)*(p.xs-p.Xsc)+(p.ys-p.Ysc)*(p.ys-p.Ysc));
    if(p.diss<p.Rhos or p.diss==p.Rhos){
    mu=sqrt(fabs(1.0-p.diss*p.diss/(p.Rhos*p.Rhos)));   
    p.image+=               fabs(1.0-p.limb*(1.0-mu));
    if(p.Fla<0.9){p.obscur+=fabs(1.0-p.limb*(1.0-mu));}
    //else{
    //if(double(p.Rlens*Rsun)>double(p.Rout*p.RE)){
    //cout<<"** b/Rsun: "<<sqrt(p.xi*p.xi+p.yi*p.yi)/Rsun<<"\t Rlens: "<<p.Rlens<<"\t Rout/Rsun: "<<p.Rout*p.RE/Rsun<<endl;
    //int iie; cin>>iie;}}
    }
    }}
}    
//##############################################################################









