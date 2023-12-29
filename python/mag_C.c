#include <math.h>
#include <stdlib.h>
#include <stdio.h>


/* Method developped by Lee et al. 2018 */
/* Implimented based on arxiv: 0901.1316 */
/* This technique uses simpson integration. */

/* Disk */
double u1(double t, double u, double rho)
{
    double ans;
    if(u<=rho){
        ans=0;
    }else if(t<=asin(rho/u)){
        ans=u*cos(t)-pow(rho*rho-u*u*sin(t)*sin(t), 0.5);
    }else{
        ans=0;
    }
    return ans;
}

double u2(double t, double u, double rho)
{
    double ans;
    if(u<=rho){
        ans=u*cos(t)+pow(rho*rho-u*u*sin(t)*sin(t), 0.5);
    }else if(t<=asin(rho/u)){
        ans=u*cos(t)+pow(rho*rho-u*u*sin(t)*sin(t), 0.5);
    }else{
        ans=0;
    }
    return ans;
}

double f(double t, double u, double rho)
{
    double ans, xu1, xu2;
    xu1 = u1(t, u, rho);
    xu2 = u2(t, u, rho);
    ans = xu2*pow(xu2*xu2+4.0, 0.5) - xu1*pow(xu1*xu1+4.0, 0.5);
    return ans;
}

double A_disk_scalar(double u, double rho, int n)
{
    int k;
    double ans, pref, first, second, third;
    double pi=3.14159265358979323846264338;
    
    if(u<=rho){
        pref=1.0/pi/rho/rho * pi/2/n;
        first=((u+rho)*pow(pow(u+rho,2)+4, 0.5)-(u-rho)*pow(pow(u-rho,2)+4, 0.5))/3.0;
        second=0;
        for(k=1;k<=n-1;k++){
            second=second+f(2*k*pi/2/n, u, rho);
        }
        second=second*2.0/3.0;
        third=0;
        for(k=1;k<=n;k++){
            third=third+f((2*k-1)*pi/2.0/n, u, rho);
        }
        third=third*4.0/3.0;
        ans=pref*(first+second+third);
    }else{
        pref=1.0/pi/rho/rho * asin(rho/u)/n;
        first=((u+rho)*pow(pow(u+rho,2)+4,0.5)-(u-rho)*pow(pow(u-rho,2)+4,0.5))/3.0;
        second=0;
        for(k=1;k<=n/2.0-1;k++){
            second=second+f(2*k*asin(rho/u)/n, u, rho);
        }
        second=second*2.0/3.0;
        third=0;
        for(k=1;k<=n/2.0;k++){
            third=third+f((2*k-1)*asin(rho/u)/n, u, rho);
        }
        third=third*4.0/3.0;
        ans=pref*(first+second+third);
    }
    return ans;
}
void A_disk(double u[], double rho, double a[], int n, int Ngrid)
{
    double xu;
    int i;
    for(i=0;i<Ngrid;i++){
        xu=u[i];
        a[i]=A_disk_scalar(xu, rho, n);
    }
}

/* Limb darkening */
double s_limb(double u, double rho, double order)
{
    double ans=0;
    double pi=3.14159265358979323846264338;
    if(u<=rho){
        ans=(1+order/2.0)/pi/rho/rho * pow(1.0-pow(u/rho, 2), order/2.0);
    }
    return ans;
}
double A_limb_scalar(double u, double rho, int n1, int n2, double order)
{
    int k1, k2;
    double ans, t, xu, xu1, xu2, r;
    double pi=3.14159265358979323846264338;
    
    ans=0;
    for(k1=0;k1<=n1;k1++){
        t=pi*k1/n1;
        xu1=u1(t, u, rho);
        xu2=u2(t, u, rho);
        for(k2=0;k2<=n2;k2++){
            xu=xu1+k2*(xu2-xu1)/n2;
            r=pow(xu*xu+u*u-2*xu*u*cos(t), 0.5);
            ans=ans+(xu*xu+2)/pow(xu*xu+4, 0.5)*s_limb(r, rho, order) * (xu2-xu1)/n2 * pi/n1;
        }
    }
    ans=ans*2;
    return ans;
}
void A_limb(double u[], double rho, double a[], int n1, int n2, int Ngrid, double order)
{
    int i;
    for(i=0;i<Ngrid;i++){
        a[i]=A_limb_scalar(u[i], rho, n1, n2, order);
    }
}