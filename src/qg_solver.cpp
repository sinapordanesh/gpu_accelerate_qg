// qg_solver.cpp  
#include "qg_solver.h"  
#include <iostream>  
#include <fstream>  
#include <cmath>  
#include <filesystem>  
#include <cuda_runtime.h>  
#include <device_launch_parameters.h>  
#include <curand.h>  
#include <cufft.h>  
#include <cufftXt.h>  
#include <helper_cuda.h>  

// CUDA kernel to compute the linear operator L  
__global__ void setupLinearOperatorKernel(double *L, double nu, int sh, int N) {  
    int i = blockIdx.x * blockDim.x + threadIdx.x;  
    int j = blockIdx.y * blockDim.y + threadIdx.y;  
    
    if (i < N && j < N) {  
        int idx = j * N + i;  
        int ki = (i <= N/2) ? i : i - N;  
        int kj = (j <= N/2) ? j : j - N;  
        
        double kr = sqrt(ki*ki + kj*kj);  
        L[idx] = -nu * pow(kr, 2*sh);  
    }  
}  

// CUDA kernel to compute stochastic forcing spectrum  
__global__ void setupStochasticForcingKernel(cufftDoubleComplex *Fk, double epsf, double kf,   
                                            double sigf, double alpf, int N) {  
    int i = blockIdx.x * blockDim.x + threadIdx.x;  
    int j = blockIdx.y * blockDim.y + threadIdx.y;  
    
    if (i < N && j < N) {  
        int idx = j * N + i;  
        int ki = (i <= N/2) ? i : i - N;  
        int kj = (j <= N/2) ? j : j - N;  
        
        double kr = sqrt(ki*ki + kj*kj);  
        double phi = (ki != 0) ? atan2(kj, ki) : 0.0;  
        
        double ampl = epsf * (1.0 + alpf * cos(2.0 * phi));  
        double value = (ampl * kr) * exp(-0.5 * (kr-kf)*(kr-kf)/(sigf*sigf)) / (sqrt(2.0*M_PI) * sigf);  
        
        Fk[idx].x = value;  
        Fk[idx].y = 0.0;  
    }  
    
    if (i == 0 && j == 0) {  
        Fk[0].x = 0.0;  
        Fk[0].y = 0.0;  
    }  
}  

// CUDA kernel to compute wind stress  
__global__ void setupWindStressKernel(double *tau, double tau0, double tdel,   
                                     double *Y, int N, const char* forcing_type) {  
    int i = blockIdx.x * blockDim.x + threadIdx.x;  
    int j = blockIdx.y * blockDim.y + threadIdx.y;  
    
    if (i < N && j < N) {  
        int idx = j * N + i;  
        double y = Y[j];  
        
        if (strcmp(forcing_type, "sin") == 0) {  
            tau[idx] = tau0 * sin(y) * sin(y);  
        } else if (strcmp(forcing_type, "sinh") == 0) {  
            tau[idx] = tau0 * 1.0 / (cosh(y/tdel) * cosh(y/tdel));  
        } else if (strcmp(forcing_type, "constant") == 0) {  
            tau[idx] = tau0;  
        }  
    }  
}  

// CUDA kernel to convert wind stress to spectral space  
__global__ void windStressToSpectralKernel(cufftDoubleComplex *tauk, double *ky,   
                                          cufftDoubleComplex *tau_hat, int N) {  
    int i = blockIdx.x * blockDim.x + threadIdx.x;  
    int j = blockIdx.y * blockDim.y + threadIdx.y;  
    
    if (i < N && j < N) {  
        int idx = j * N + i;  
        double ky_val = ky[j];  
        
        tauk[idx].x = -ky_val * tau_hat[idx].y;  // -i * ky * tau_hat  
        tauk[idx].y = ky_val * tau_hat[idx].x;  
    }  
}  

// CUDA kernel for initializing the vorticity field  
__global__ void initializeVorticityKernel(double *qp, int N, unsigned long seed) {  
    int i = blockIdx.x * blockDim.x + threadIdx.x;  
    int j = blockIdx.y * blockDim.y + threadIdx.y;  
    
    if (i < N && j < N) {  
        int idx = j * N + i;  
        // Use a simple PRNG for initialization  
        unsigned long a = 16807;  
        unsigned long m = 2147483647;  
        unsigned long x = (seed + idx) % m;  
        x = (a * x) % m;  
        double r = (double)x / m;  
        qp[idx] = 0.01 * (2.0*r - 1.0);  // Random numbers in [-0.01, 0.01]  
    }  
}  

// CUDA kernel for computing RHS  
__global__ void computeRHSKernel(cufftDoubleComplex *q, cufftDoubleComplex *psi,  
                               double *kx, double *ky, double *laplacian,  
                               double beta, double dd, cufftDoubleComplex *RHS, int N) {  
    int i = blockIdx.x * blockDim.x + threadIdx.x;  
    int j = blockIdx.y * blockDim.y + threadIdx.y;  
    
    if (i < N && j < N) {  
        int idx = j * N + i;  
        double kx_val = kx[i];  
        
        // Compute stream function: psi = q / laplacian  
        if (i == 0 && j == 0) {  
            psi[idx].x = 0.0;  
            psi[idx].y = 0.0;  
        } else {  
            double lap = laplacian[idx];  
            psi[idx].x = q[idx].x / lap;  
            psi[idx].y = q[idx].y / lap;  
        }  
        
        // Beta effect and linear damping  
        RHS[idx].x = -beta * kx_val * psi[idx].y - dd * q[idx].x;  
        RHS[idx].y = beta * kx_val * psi[idx].x - dd * q[idx].y;  
    }  
}  

// CUDA kernel for upscaling to 3/2 grid  
__global__ void upscaleToThreeHalvesKernel(cufftDoubleComplex *in, cufftDoubleComplex *out,  
                                         int N, int N_3by2) {  
    int i = blockIdx.x * blockDim.x + threadIdx.x;  
    int j = blockIdx.y * blockDim.y + threadIdx.y;  
    
    if (i < N_3by2 && j < N_3by2) {  
        int out_idx = j * N_3by2 + i;  
        out[out_idx].x = 0.0;  
        out[out_idx].y = 0.0;  
    }  
    
    if (i < N/2+1 && j < N/2+1) {  
        int in_idx = j * N + i;  
        int out_idx = j * N_3by2 + i;  
        out[out_idx].x = (9.0/4.0) * in[in_idx].x;  
        out[out_idx].y = (9.0/4.0) * in[in_idx].y;  
    }  
    
    if (i < N/2+1 && j > N/2+1 && j < N) {  
        int in_idx = j * N + i;  
        int out_idx = (j + N_3by2 - N) * N_3by2 + i;  
        out[out_idx].x = (9.0/4.0) * in[in_idx].x;  
        out[out_idx].y = (9.0/4.0) * in[in_idx].y;  
    }  
    
    if (i > N/2+1 && i < N && j < N/2+1) {  
        int in_idx = j * N + i;  
        int out_idx = j * N_3by2 + (i + N_3by2 - N);  
        out[out_idx].x = (9.0/4.0) * in[in_idx].x;  
        out[out_idx].y = (9.0/4.0) * in[in_idx].y;  
    }  
    
    if (i > N/2+1 && i < N && j > N/2+1 && j < N) {  
        int in_idx = j * N + i;  
        int out_idx = (j + N_3by2 - N) * N_3by2 + (i + N_3by2 - N);  
        out[out_idx].x = (9.0/4.0) * in[in_idx].x;  
        out[out_idx].y = (9.0/4.0) * in[in_idx].y;  
    }  
}  

// CUDA kernel for computing the velocities and vorticity gradients  
__global__ void computeVelocitiesAndGradientsKernel(double *u, double *v, double *qx, double *qy,  
                                                 double *DX, double *DY,   
                                                 cufftDoubleComplex *Psi_hat,   
                                                 cufftDoubleComplex *Q_hat, int N_3by2) {  
    int i = blockIdx.x * blockDim.x + threadIdx.x;  
    int j = blockIdx.y * blockDim.y + threadIdx.y;  
    
    if (i < N_3by2 && j < N_3by2) {  
        int idx = j * N_3by2 + i;  
        double dx_val = DX[i];  
        double dy_val = DY[j];  
        
        // u = -d(psi)/dy  
        u[idx] = -dy_val * Psi_hat[idx].y;  // real part of -i*dy*Psi  
        
        // v = d(psi)/dx  
        v[idx] = dx_val * Psi_hat[idx].y;   // real part of i*dx*Psi  
        
        // qx = d(q)/dx  
        qx[idx] = -dx_val * Q_hat[idx].y;   // real part of i*dx*Q  
        
        // qy = d(q)/dy  
        qy[idx] = -dy_val * Q_hat[idx].y;   // real part of i*dy*Q  
    }  
}  

// CUDA kernel for computing the Jacobian (u*qx + v*qy)  
__global__ void computeJacobianKernel(double *jaco_real, double *u, double *v,   
                                    double *qx, double *qy, int N_3by2) {  
    int i = blockIdx.x * blockDim.x + threadIdx.x;  
    int j = blockIdx.y * blockDim.y + threadIdx.y;  
    
    if (i < N_3by2 && j < N_3by2) {  
        int idx = j * N_3by2 + i;  
        jaco_real[idx] = u[idx] * qx[idx] + v[idx] * qy[idx];  
    }  
}  

// CUDA kernel for downscaling from 3/2 grid  
__global__ void downscaleFromThreeHalvesKernel(cufftDoubleComplex *in, cufftDoubleComplex *out,  
                                             int N, int N_3by2) {  
    int i = blockIdx.x * blockDim.x + threadIdx.x;  
    int j = blockIdx.y * blockDim.y + threadIdx.y;  
    
    if (i < N && j < N) {  
        int out_idx = j * N + i;  
        out[out_idx].x = 0.0;  
        out[out_idx].y = 0.0;  
    }  
    
    if (i < N/2+1 && j < N/2+1) {  
        int out_idx = j * N + i;  
        int in_idx = j * N_3by2 + i;  
        out[out_idx].x = (4.0/9.0) * in[in_idx].x;  
        out[out_idx].y = (4.0/9.0) * in[in_idx].y;  
    }  
    
    if (i < N/2+1 && j > N/2+1 && j < N) {  
        int out_idx = j * N + i;  
        int in_idx = (j - N + N_3by2) * N_3by2 + i;  
        out[out_idx].x = (4.0/9.0) * in[in_idx].x;  
        out[out_idx].y = (4.0/9.0) * in[in_idx].y;  
    }  
    
    if (i > N/2+1 && i < N && j < N/2+1) {  
        int out_idx = j * N + i;  
        int in_idx = j * N_3by2 + (i - N + N_3by2);  
        out[out_idx].x = (4.0/9.0) * in[in_idx].x;  
        out[out_idx].y = (4.0/9.0) * in[in_idx].y;  
    }  
    
    if (i > N/2+1 && i < N && j > N/2+1 && j < N) {  
        int out_idx = j * N + i;  
        int in_idx = (j - N + N_3by2) * N_3by2 + (i - N + N_3by2);  
        out[out_idx].x = (4.0/9.0) * in[in_idx].x;  
        out[out_idx].y = (4.0/9.0) * in[in_idx].y;  
    }  
}  

// CUDA kernel for subtracting Jacobian from RHS  
__global__ void subtractJacobianKernel(cufftDoubleComplex *RHS, cufftDoubleComplex *jaco_hat, int N) {  
    int i = blockIdx.x * blockDim.x + threadIdx.x;  
    int j = blockIdx.y * blockDim.y + threadIdx.y;  
    
    if (i < N && j < N) {  
        int idx = j * N + i;  
        RHS[idx].x -= jaco_hat[idx].x;  
        RHS[idx].y -= jaco_hat[idx].y;  
    }  
}  

// CUDA kernel for ARK4 first stage  
__global__ void ark4Stage1Kernel(cufftDoubleComplex *q, cufftDoubleComplex *k0,  
                              cufftDoubleComplex *l0, double *L, int N) {  
    int i = blockIdx.x * blockDim.x + threadIdx.x;  
    int j = blockIdx.y * blockDim.y + threadIdx.y;  
    
    if (i < N && j < N) {  
        int idx = j * N + i;  
        l0[idx].x = L[idx] * q[idx].x;  
        l0[idx].y = L[idx] * q[idx].y;  
    }  
}  

// CUDA kernel to compute M = 1/(1-.25*dt*L)  
__global__ void computeMKernel(double *L, cufftDoubleComplex *M, double dt, int N) {  
    int i = blockIdx.x * blockDim.x + threadIdx.x;  
    int j = blockIdx.y * blockDim.y + threadIdx.y;  
    
    if (i < N && j < N) {  
        int idx = j * N + i;  
        double denom = 1.0 - 0.25 * dt * L[idx];  
        M[idx].x = 1.0 / denom;  
        M[idx].y = 0.0;  
    }  
}  

// CUDA kernel for ARK4 second stage  
__global__ void ark4Stage2Kernel(cufftDoubleComplex *q, cufftDoubleComplex *q1,  
                              cufftDoubleComplex *k0, cufftDoubleComplex *l0,  
                              cufftDoubleComplex *M, double dt, int N) {  
    int i = blockIdx.x * blockDim.x + threadIdx.x;  
    int j = blockIdx.y * blockDim.y + threadIdx.y;  
    
    if (i < N && j < N) {  
        int idx = j * N + i;  
        q1[idx].x = M[idx].x * (q[idx].x + 0.5*dt*k0[idx].x + 0.25*dt*l0[idx].x);  
        q1[idx].y = M[idx].x * (q[idx].y + 0.5*dt*k0[idx].y + 0.25*dt*l0[idx].y);  
    }  
}  

// CUDA kernel for computing l1  
__global__ void computeL1Kernel(cufftDoubleComplex *q1, cufftDoubleComplex *l1,  
                             double *L, int N) {  
    int i = blockIdx.x * blockDim.x + threadIdx.x;  
    int j = blockIdx.y * blockDim.y + threadIdx.y;  
    
    if (i < N && j < N) {  
        int idx = j * N + i;  
        l1[idx].x = L[idx] * q1[idx].x;  
        l1[idx].y = L[idx] * q1[idx].y;  
    }  
}  

// CUDA kernel for ARK4 third stage  
__global__ void ark4Stage3Kernel(cufftDoubleComplex *q, cufftDoubleComplex *q2,  
                              cufftDoubleComplex *k0, cufftDoubleComplex *k1,  
                              cufftDoubleComplex *l0, cufftDoubleComplex *l1,  
                              cufftDoubleComplex *M, double dt, int N) {  
    int i = blockIdx.x * blockDim.x + threadIdx.x;  
    int j = blockIdx.y * blockDim.y + threadIdx.y;  
    
    if (i < N && j < N) {  
        int idx = j * N + i;  
        double a = 13861.0/62500.0;  
        double b = 6889.0/62500.0;  
        double c = 8611.0/62500.0;  
        double d = -1743.0/31250.0;  
        
        q2[idx].x = M[idx].x * (q[idx].x + dt*(a*k0[idx].x + b*k1[idx].x + c*l0[idx].x + d*l1[idx].x));  
        q2[idx].y = M[idx].x * (q[idx].y + dt*(a*k0[idx].y + b*k1[idx].y + c*l0[idx].y + d*l1[idx].y));  
    }  
}  

// CUDA kernel for computing l2  
__global__ void computeL2Kernel(cufftDoubleComplex *q2, cufftDoubleComplex *l2,  
                             double *L, int N) {  
    int i = blockIdx.x * blockDim.x + threadIdx.x;  
    int j = blockIdx.y * blockDim.y + threadIdx.y;  
    
    if (i < N && j < N) {  
        int idx = j * N + i;  
        l2[idx].x = L[idx] * q2[idx].x;  
        l2[idx].y = L[idx] * q2[idx].y;  
    }  
}  

// CUDA kernel for ARK4 fourth stage  
__global__ void ark4Stage4Kernel(cufftDoubleComplex *q, cufftDoubleComplex *q3,  
                              cufftDoubleComplex *k0, cufftDoubleComplex *k1,  
                              cufftDoubleComplex *k2, cufftDoubleComplex *l0,  
                              cufftDoubleComplex *l1, cufftDoubleComplex *l2,  
                              cufftDoubleComplex *M, double dt, int N) {  
    int i = blockIdx.x * blockDim.x + threadIdx.x;  
    int j = blockIdx.y * blockDim.y + threadIdx.y;  
    
    if (i < N && j < N) {  
        int idx = j * N + i;  
        double a0 = -0.04884659515311858;  
        double a1 = -0.1777206523264010;  
        double a2 = 0.8465672474795196;  
        double b0 = 0.1446368660269822;  
        double b1 = -0.2239319076133447;  
        double b2 = 0.4492950415863626;  
        
        q3[idx].x = M[idx].x * (q[idx].x + dt*(a0*k0[idx].x + a1*k1[idx].x + a2*k2[idx].x +  
                                            b0*l0[idx].x + b1*l1[idx].x + b2*l2[idx].x));  
        q3[idx].y = M[idx].x * (q[idx].y + dt*(a0*k0[idx].y + a1*k1[idx].y + a2*k2[idx].y +  
                                            b0*l0[idx].y + b1*l1[idx].y + b2*l2[idx].y));  
    }  
}  

// CUDA kernel for computing l3  
__global__ void computeL3Kernel(cufftDoubleComplex *q3, cufftDoubleComplex *l3,  
                             double *L, int N) {  
    int i = blockIdx.x * blockDim.x + threadIdx.x;  
    int j = blockIdx.y * blockDim.y + threadIdx.y;  
    
    if (i < N && j < N) {  
        int idx = j * N + i;  
        l3[idx].x = L[idx] * q3[idx].x;  
        l3[idx].y = L[idx] * q3[idx].y;  
    }  
}  

// CUDA kernel for ARK4 fifth stage  
__global__ void ark4Stage5Kernel(cufftDoubleComplex *q, cufftDoubleComplex *q4,  
                              cufftDoubleComplex *k0, cufftDoubleComplex *k1,  
                              cufftDoubleComplex *k2, cufftDoubleComplex *k3,  
                              cufftDoubleComplex *l0, cufftDoubleComplex *l1,  
                              cufftDoubleComplex *l2, cufftDoubleComplex *l3,  
                              cufftDoubleComplex *M, double dt, int N) {  
    int i = blockIdx.x * blockDim.x + threadIdx.x;  
    int j = blockIdx.y * blockDim.y + threadIdx.y;  
    
    if (i < N && j < N) {  
        int idx = j * N + i;  
        double a0 = -0.1554168584249155;  
        double a1 = -0.3567050098221991;  
        double a2 = 1.058725879868443;  
        double a3 = 0.3033959883786719;  
        double b0 = 0.09825878328356477;  
        double b1 = -0.5915442428196704;  
        double b2 = 0.8101210538282996;  
        double b3 = 0.2831644057078060;  
        
        q4[idx].x = M[idx].x * (q[idx].x + dt*(a0*k0[idx].x + a1*k1[idx].x + a2*k2[idx].x + a3*k3[idx].x +  
                                            b0*l0[idx].x + b1*l1[idx].x + b2*l2[idx].x + b3*l3[idx].x));  
        q4[idx].y = M[idx].x * (q[idx].y + dt*(a0*k0[idx].y + a1*k1[idx].y + a2*k2[idx].y + a3*k3[idx].y +  
                                            b0*l0[idx].y + b1*l1[idx].y + b2*l2[idx].y + b3*l3[idx].y));  
    }  
}  

// CUDA kernel for computing l4  
__global__ void computeL4Kernel(cufftDoubleComplex *q4, cufftDoubleComplex *l4,  
                             double *L, int N) {  
    int i = blockIdx.x * blockDim.x + threadIdx.x;  
    int j = blockIdx.y * blockDim.y + threadIdx.y;  
    
    if (i < N && j < N) {  
        int idx = j * N + i;  
        l4[idx].x = L[idx] * q4[idx].x;  
        l4[idx].y = L[idx] * q4[idx].y;  
    }  
}  

// CUDA kernel for ARK4 sixth stage  
__global__ void ark4Stage6Kernel(cufftDoubleComplex *q, cufftDoubleComplex *q5,  
                               cufftDoubleComplex *k0, cufftDoubleComplex *k1,  
                               cufftDoubleComplex *k2, cufftDoubleComplex *k3,  
                               cufftDoubleComplex *k4, cufftDoubleComplex *l0,  
                               cufftDoubleComplex *l2, cufftDoubleComplex *l3,  
                               cufftDoubleComplex *l4, cufftDoubleComplex *M,  
                               double dt, int N) {  
    int i = blockIdx.x * blockDim.x + threadIdx.x;  
    int j = blockIdx.y * blockDim.y + threadIdx.y;  
    
    if (i < N && j < N) {  
        int idx = j * N + i;  
        double a0 = 0.2014243506726763;  
        double a1 = 0.008742057842904184;  
        double a2 = 0.1599399570716811;  
        double a3 = 0.4038290605220775;  
        double a4 = 0.2260645738906608;  
        double b0 = 0.1579162951616714;  
        double b2 = 0.1867589405240008;  
        double b3 = 0.6805652953093346;  
        double b4 = -0.2752405309950067;  
        
        q5[idx].x = M[idx].x * (q[idx].x + dt*(a0*k0[idx].x + a1*k1[idx].x + a2*k2[idx].x +  
                                            a3*k3[idx].x + a4*k4[idx].x + b0*l0[idx].x +  
                                            b2*l2[idx].x + b3*l3[idx].x + b4*l4[idx].x));  
        q5[idx].y = M[idx].x * (q[idx].y + dt*(a0*k0[idx].y + a1*k1[idx].y + a2*k2[idx].y +  
                                            a3*k3[idx].y + a4*k4[idx].y + b0*l0[idx].y +  
                                            b2*l2[idx].y + b3*l3[idx].y + b4*l4[idx].y));  
    }  
}  

// CUDA kernel for computing l5  
__global__ void computeL5Kernel(cufftDoubleComplex *q5, cufftDoubleComplex *l5,  
                             double *L, int N) {  
    int i = blockIdx.x * blockDim.x + threadIdx.x;  
    int j = blockIdx.y * blockDim.y + threadIdx.y;  
    
    if (i < N && j < N) {  
        int idx = j * N + i;  
        l5[idx].x = L[idx] * q5[idx].x;  
        l5[idx].y = L[idx] * q5[idx].y;  
    }  
}  

// CUDA kernel for computing error for adaptive time stepping  
__global__ void computeErrorKernel(cufftDoubleComplex *k0, cufftDoubleComplex *l0,  
                                 cufftDoubleComplex *k2, cufftDoubleComplex *l2,  
                                 cufftDoubleComplex *k3, cufftDoubleComplex *l3,  
                                 cufftDoubleComplex *k4, cufftDoubleComplex *l4,  
                                 cufftDoubleComplex *k5, cufftDoubleComplex *l5,  
                                 cufftDoubleComplex *error, double dt, int N) {  
    int i = blockIdx.x * blockDim.x + threadIdx.x;  
    int j = blockIdx.y * blockDim.y + threadIdx.y;  
    
    if (i < N && j < N) {  
        int idx = j * N + i;  
        double a0 = 0.003204494398459;  
        double a2 = -0.002446251136679;  
        double a3 = -0.021480075919587;  
        double a4 = 0.043946868068572;  
        double a5 = -0.023225035410765;  
        
        error[idx].x = dt * (a0*(k0[idx].x+l0[idx].x) + a2*(k2[idx].x+l2[idx].x) +  
                            a3*(k3[idx].x+l3[idx].x) + a4*(k4[idx].x+l4[idx].x) +  
                            a5*(k5[idx].x+l5[idx].x));  
        error[idx].y = dt * (a0*(k0[idx].y+l0[idx].y) + a2*(k2[idx].y+l2[idx].y) +  
                            a3*(k3[idx].y+l3[idx].y) + a4*(k4[idx].y+l4[idx].y) +  
                            a5*(k5[idx].y+l5[idx].y));  
    }  
}  

// CUDA kernel for final ARK4 update  
__global__ void ark4FinalUpdateKernel(cufftDoubleComplex *q, cufftDoubleComplex *q_new,  
                                   cufftDoubleComplex *k0, cufftDoubleComplex *l0,  
                                   cufftDoubleComplex *k2, cufftDoubleComplex *l2,  
                                   cufftDoubleComplex *k3, cufftDoubleComplex *l3,  
                                   cufftDoubleComplex *k4, cufftDoubleComplex *l4,  
                                   cufftDoubleComplex *k5, cufftDoubleComplex *l5,  
                                   cufftDoubleComplex *xik, double dt, int N) {  
    int i = blockIdx.x * blockDim.x + threadIdx.x;  
    int j = blockIdx.y * blockDim.y + threadIdx.y;  
    
    if (i < N && j < N) {  
        int idx = j * N + i;  
        double a0 = 0.1579162951616714;  
        double a2 = 0.1867589405240008;  
        double a3 = 0.6805652953093346;  
        double a4 = -0.2752405309950067;  
        double a5 = 0.25;  
        
        q_new[idx].x = q[idx].x + dt*(a0*(k0[idx].x+l0[idx].x) + a2*(k2[idx].x+l2[idx].x) +  
                                  a3*(k3[idx].x+l3[idx].x) + a4*(k4[idx].x+l4[idx].x) +  
                                  a5*(k5[idx].x+l5[idx].x)) + sqrt(dt)*xik[idx].x;  
        q_new[idx].y = q[idx].y + dt*(a0*(k0[idx].y+l0[idx].y) + a2*(k2[idx].y+l2[idx].y) +  
                                  a3*(k3[idx].y+l3[idx].y) + a4*(k4[idx].y+l4[idx].y) +  
                                  a5*(k5[idx].y+l5[idx].y)) + sqrt(dt)*xik[idx].y;  
    }  
}  

// CUDA kernel for preparing stochastic forcing  
__global__ void prepareStochasticForcingKernel(cufftDoubleComplex *dW, cufftDoubleComplex *Fk,  
                                            cufftDoubleComplex *xik, int N) {  
    int i = blockIdx.x * blockDim.x + threadIdx.x;  
    int j = blockIdx.y * blockDim.y + threadIdx.y;  
    
    if (i < N && j < N) {  
        int idx = j * N + i;  
        xik[idx].x = N * N * dW[idx].x * sqrt(Fk[idx].x);  
        xik[idx].y = N * N * dW[idx].y * sqrt(Fk[idx].x);  
    }  
}  

// Function implementations  
QGSolver::QGSolver(int N_, double beta, int kf, double epsf, double sigf, double alpf,   
                 double dd, double tau0, double tdel, std::string forcing_type_)  
    : N(N_), forcing_type(forcing_type_), qlim(1.5E5) {  
    
    // Initialize parameters  
    h_params.beta = beta;  
    h_params.sh = 8;  // hyperdiffusion exponent  
    h_params.dd = dd;  
    h_params.N = N;  
    h_params.kf = kf;  
    h_params.epsf = epsf;  
    h_params.sigf = sigf;  
    h_params.alpf = alpf;  
    h_params.tau0 = tau0;  
    h_params.tdel = tdel;  
    
    // Set up hyperviscous PV dissipation coefficient  
    if (N == 128) {  
        if (epsf == 0) {  
            h_params.nu = 2.5e-28 * pow(50.0/50.0, (-2*h_params.sh+2.0/3.0));  
        } else {  
            h_params.nu = 2.5e-28 * pow(kf/50.0, (-2*h_params.sh+2.0/3.0));  
        }  
    } else if (N == 256) {  
        h_params.nu = 5e-33 * pow(kf/50.0, (-2*h_params.sh+2.0/3.0));  
        if (tau0 == 0) {  
            h_params.nu *= 100;  
        }  
    } else if (N == 512) {  
        h_params.nu = 1e-34 * pow(kf/50.0, (-2*h_params.sh+2.0/3.0));  
    }  
    
    // Time stepping parameters  
    t = 0.0;  
    dt = 1e-5;  // initial time step size  
    tol = 1e-1;  
    r0 = 0.8 * tol;  
    
    // Initialize data structures  
    initialize();  
}  

QGSolver::~QGSolver() {  
    cleanup();  
}  

void QGSolver::initialize() {  
    std::cout << "Initializing data structures..." << std::endl;  
    
    // Complex array size  
    complexSize = N * N * sizeof(cufftDoubleComplex);  
    
    // Initialize host arrays  
    h_q.resize(N * N);  
    h_qp.resize(N * N);  
    h_X.resize(N);  
    h_Y.resize(N);  
    
    // Create coordinate arrays  
    double dx = 2.0 * M_PI / N;  
    for (int i = 0; i < N; i++) {  
        h_X[i] = -M_PI + i * dx;  
        h_Y[i] = -M_PI + i * dx;  
    }  
    
    // Allocate device memory  
    cudaMalloc((void**)&d_q, complexSize);  
    cudaMalloc((void**)&d_psi, complexSize);  
    cudaMalloc((void**)&d_qp, N * N * sizeof(double));  
    cudaMalloc((void**)&d_L, N * N * sizeof(double));  
    cudaMalloc((void**)&d_Fk, complexSize);  
    cudaMalloc((void**)&d_tauk, complexSize);  
    cudaMalloc((void**)&d_RHS, complexSize);  
    
    // Allocate memory for ARK4 stages  
    cudaMalloc((void**)&d_k0, complexSize);  
    cudaMalloc((void**)&d_k1, complexSize);  
    cudaMalloc((void**)&d_k2, complexSize);  
    cudaMalloc((void**)&d_k3, complexSize);  
    cudaMalloc((void**)&d_k4, complexSize);  
    cudaMalloc((void**)&d_k5, complexSize);  
    
    cudaMalloc((void**)&d_l0, complexSize);  
    cudaMalloc((void**)&d_l1, complexSize);  
    cudaMalloc((void**)&d_l2, complexSize);  
    cudaMalloc((void**)&d_l3, complexSize);  
    cudaMalloc((void**)&d_l4, complexSize);  
    cudaMalloc((void**)&d_l5, complexSize);  
    
    cudaMalloc((void**)&d_q1, complexSize);  
    cudaMalloc((void**)&d_q2, complexSize);  
    cudaMalloc((void**)&d_q3, complexSize);  
    cudaMalloc((void**)&d_q4, complexSize);  
    cudaMalloc((void**)&d_q5, complexSize);  
    
    cudaMalloc((void**)&d_M, complexSize);  
    
    // Allocate memory for stochastic forcing  
    cudaMalloc((void**)&d_dW, complexSize);  
    cudaMalloc((void**)&d_xik, complexSize);  
    
    // Allocate memory for 3/2 sized arrays (for dealiasing)  
    int N_3by2 = (3 * N) / 2;  
    cudaMalloc((void**)&d_Psi_hat_3by2, N_3by2 * N_3by2 * sizeof(cufftDoubleComplex));  
    cudaMalloc((void**)&d_Q_hat_3by2, N_3by2 * N_3by2 * sizeof(cufftDoubleComplex));  
    cudaMalloc((void**)&d_u_3by2, N_3by2 * N_3by2 * sizeof(double));  
    cudaMalloc((void**)&d_v_3by2, N_3by2 * N_3by2 * sizeof(double));  
    cudaMalloc((void**)&d_qx_3by2, N_3by2 * N_3by2 * sizeof(double));  
    cudaMalloc((void**)&d_qy_3by2, N_3by2 * N_3by2 * sizeof(double));  
    cudaMalloc((void**)&d_jaco_real_3by2, N_3by2 * N_3by2 * sizeof(double));  
    cudaMalloc((void**)&d_Jaco_hat_3by2, N_3by2 * N_3by2 * sizeof(cufftDoubleComplex));  
    
    // Allocate memory for maximum finder (for adaptive time stepping)  
    cudaMalloc((void**)&d_max_array, N * N * sizeof(double));  
    cudaMalloc((void**)&d_r1, sizeof(double));  
    
    // Create CUDA FFT plans  
    cufftPlan2d(&fft_plan, N, N, CUFFT_D2Z);  
    cufftPlan2d(&ifft_plan, N, N, CUFFT_Z2D);  
    cufftPlan2d(&fft_plan_3by2, N_3by2, N_3by2, CUFFT_D2Z);  
    cufftPlan2d(&ifft_plan_3by2, N_3by2, N_3by2, CUFFT_Z2D);  
    
    // Initialize cuRAND generator  
    curandCreateGenerator(&rng, CURAND_RNG_PSEUDO_DEFAULT);  
    curandSetPseudoRandomGeneratorSeed(rng, time(NULL));  
    
    // Initialize vorticity field  
    dim3 blockSize(16, 16);  
    dim3 gridSize((N + blockSize.x - 1) / blockSize.x, (N + blockSize.y - 1) / blockSize.y);  
    
    initializeVorticityKernel<<<gridSize, blockSize>>>(d_qp, N, time(NULL));  
    
    // Transform to spectral space  
    cufftExecD2Z(fft_plan, (double*)d_qp, d_q);  
    
    // Setup operators  
    setupOperators();  
    
    // Setup stochastic forcing  
    setupStochasticForcing();  
    
    // Setup wind stress  
    setupWindStress();  
    
    // Create output folder  
    std::string fstr;  
    if (h_params.tdel == 0) {  
        fstr = "forcing-" + forcing_type + "_tau0-" + std::to_string(h_params.tau0);  
    } else {  
        fstr = "forcing-" + forcing_type + "_tau0-" + std::to_string(h_params.tau0) +   
               "_tdel" + std::to_string(h_params.tdel);  
    }  
    
    if (h_params.epsf != 0) {  
        if (h_params.alpf == 0) {  
            fstr = fstr + "_kf" + std::to_string(h_params.kf) + "_epsf" +   
                   std::to_string(h_params.epsf) + "_sigf" + std::to_string(h_params.sigf);  
        } else {  
            fstr = fstr + "_kf" + std::to_string(h_params.kf) + "_epsf" +   
                   std::to_string(h_params.epsf) + "_sigf" + std::to_string(h_params.sigf) +   
                   "_alpha" + std::to_string(h_params.alpf);  
        }  
    }  
    
    std::string casestr = "ARK4_N" + std::to_string(N) +   
                        "_tau" + std::to_string(h_params.tau0) +   
                        "_tdel" + std::to_string(h_params.tdel);  
    
    data_folder = "data/" + casestr;  
    std::filesystem::create_directories(data_folder);  
    
    // Open output files  
    std::string qh_filename = data_folder + "/qh.bin";  
    std::string t_filename = data_folder + "/t.bin";  
    
    qh_file = fopen(qh_filename.c_str(), "wb");  
    t_file = fopen(t_filename.c_str(), "wb");  
    
    // Initialize diagnostics arrays  
    int diagSize = N / 200;  // Diagnostic storage size (adjust as needed)  
    T.resize(diagSize, 0.0);  
    vb.resize(diagSize, 0.0);  
    utz.resize(N);  
    for (int i = 0; i < N; i++) {  
        utz[i].resize(diagSize, 0.0);  
    }  
    energy.resize(N/2+1);  
    for (int i = 0; i < N/2+1; i++) {  
        energy[i].resize(diagSize, 0.0);  
    }  
    enstrophy.resize(N/2+1);  
    for (int i = 0; i < N/2+1; i++) {  
        enstrophy[i].resize(diagSize, 0.0);  
    }  
    dtsize.resize(diagSize, 0.0);  
    
    count = 0;  
    
    // Save initial state  
    saveData();  
}  

void QGSolver::setupOperators() {  
    // Create wavenumber arrays  
    double *h_k = new double[N];  
    for (int i = 0; i < N/2+1; i++) {  
        h_k[i] = i;  
    }  
    for (int i = N/2+1; i < N; i++) {  
        h_k[i] = i - N;  
    }  
    
    double *d_k;  
    cudaMalloc((void**)&d_k, N * sizeof(double));  
    cudaMemcpy(d_k, h_k, N * sizeof(double), cudaMemcpyHostToDevice);  
    
    // Setup linear operator  
    dim3 blockSize(16, 16);  
    dim3 gridSize((N + blockSize.x - 1) / blockSize.x, (N + blockSize.y - 1) / blockSize.y);  
    
    setupLinearOperatorKernel<<<gridSize, blockSize>>>(d_L, h_params.nu, h_params.sh, N);  
    
    // Clean up  
    delete[] h_k;  
    cudaFree(d_k);  
}  

void QGSolver::setupStochasticForcing() {  
    dim3 blockSize(16, 16);  
    dim3 gridSize((N + blockSize.x - 1) / blockSize.x, (N + blockSize.y - 1) / blockSize.y);  
    
    setupStochasticForcingKernel<<<gridSize, blockSize>>>(d_Fk, h_params.epsf, h_params.kf,  
                                                       h_params.sigf, h_params.alpf, N);  
}  

void QGSolver::setupWindStress() {  
    // Create coordinate arrays  
    double dx = 2.0 * M_PI / N;  
    double *h_kx = new double[N];  
    double *h_ky = new double[N];  
    double *h_X = new double[N];  
    double *h_Y = new double[N];  
    
    for (int i = 0; i < N/2+1; i++) {  
        h_kx[i] = i;  
        h_ky[i] = i;  
    }  
    for (int i = N/2+1; i < N; i++) {  
        h_kx[i] = i - N;  
        h_ky[i] = i - N;  
    }  
    
    for (int i = 0; i < N; i++) {  
        h_X[i] = -M_PI + i * dx;  
        h_Y[i] = -M_PI + i * dx;  
    }  
    
    // Copy to device  
    double *d_X, *d_Y, *d_kx, *d_ky;  
    cudaMalloc((void**)&d_X, N * sizeof(double));  
    cudaMalloc((void**)&d_Y, N * sizeof(double));  
    cudaMalloc((void**)&d_kx, N * sizeof(double));  
    cudaMalloc((void**)&d_ky, N * sizeof(double));  
    
    cudaMemcpy(d_X, h_X, N * sizeof(double), cudaMemcpyHostToDevice);  
    cudaMemcpy(d_Y, h_Y, N * sizeof(double), cudaMemcpyHostToDevice);  
    cudaMemcpy(d_kx, h_kx, N * sizeof(double), cudaMemcpyHostToDevice);  
    cudaMemcpy(d_ky, h_ky, N * sizeof(double), cudaMemcpyHostToDevice);  
    
    // Create 2D grid of Y coordinates  
    double *d_Y_grid;  
    cudaMalloc((void**)&d_Y_grid, N * N * sizeof(double));  
    
    dim3 blockSize(16, 16);  
    dim3 gridSize((N + blockSize.x - 1) / blockSize.x, (N + blockSize.y - 1) / blockSize.y);  
    
    // Create wind stress  
    double *d_tau;  
    cudaMalloc((void**)&d_tau, N * N * sizeof(double));  
    
    setupWindStressKernel<<<gridSize, blockSize>>>(d_tau, h_params.tau0, h_params.tdel,   
                                                d_Y_grid, N, forcing_type.c_str());  
    
    // Convert to spectral space  
    cufftDoubleComplex *d_tau_hat;  
    cudaMalloc((void**)&d_tau_hat, N * N * sizeof(cufftDoubleComplex));  
    
    cufftExecD2Z(fft_plan, d_tau, d_tau_hat);  
    
    // Multiply by -i*ky  
    windStressToSpectralKernel<<<gridSize, blockSize>>>(d_tauk, d_ky, d_tau_hat, N);  
    
    // Clean up  
    delete[] h_kx;  
    delete[] h_ky;  
    delete[] h_X;  
    delete[] h_Y;  
    
    cudaFree(d_X);  
    cudaFree(d_Y);  
    cudaFree(d_kx);  
    cudaFree(d_ky);  
    cudaFree(d_Y_grid);  
    cudaFree(d_tau);  
    cudaFree(d_tau_hat);  
}  

void QGSolver::computeRHS(cufftDoubleComplex *q, cufftDoubleComplex *rhs) {  
    dim3 blockSize(16, 16);  
    dim3 gridSize((N + blockSize.x - 1) / blockSize.x, (N + blockSize.y - 1) / blockSize.y);  
    
    // Create wavenumber arrays  
    double *h_k = new double[N];  
    for (int i = 0; i < N/2+1; i++) {  
        h_k[i] = i;  
    }  
    for (int i = N/2+1; i < N; i++) {  
        h_k[i] = i - N;  
    }  
    
    double *d_kx, *d_ky, *d_laplacian;  
    cudaMalloc((void**)&d_kx, N * sizeof(double));  
    cudaMalloc((void**)&d_ky, N * sizeof(double));  
    cudaMalloc((void**)&d_laplacian, N * N * sizeof(double));  
    
    cudaMemcpy(d_kx, h_k, N * sizeof(double), cudaMemcpyHostToDevice);  
    cudaMemcpy(d_ky, h_k, N * sizeof(double), cudaMemcpyHostToDevice);  
    
    // Compute Laplacian  
    // ... (implementation of computing Laplacian operator)  
    
    // Compute psi and linear part of RHS  
    computeRHSKernel<<<gridSize, blockSize>>>(q, d_psi, d_kx, d_ky, d_laplacian,   
                                           h_params.beta, h_params.dd, rhs, N);  
    
    // Compute Jacobian using 3/2 rule for dealiasing  
    int N_3by2 = (3 * N) / 2;  
    dim3 gridSize_3by2((N_3by2 + blockSize.x - 1) / blockSize.x, (N_3by2 + blockSize.y - 1) / blockSize.y);  
    
    // Upscale to 3/2 grid  
    upscaleToThreeHalvesKernel<<<gridSize, blockSize>>>(q, d_Q_hat_3by2, N, N_3by2);  
    upscaleToThreeHalvesKernel<<<gridSize, blockSize>>>(d_psi, d_Psi_hat_3by2, N, N_3by2);  
    
    // Create 3/2 sized derivative operators  
    double *d_DX_3by2, *d_DY_3by2;  
    cudaMalloc((void**)&d_DX_3by2, N_3by2 * sizeof(double));  
    cudaMalloc((void**)&d_DY_3by2, N_3by2 * sizeof(double));  
    
    // ... (setup for 3/2 sized derivative operators)  
    
    // Compute velocities and vorticity gradients  
    computeVelocitiesAndGradientsKernel<<<gridSize_3by2, blockSize>>>(d_u_3by2, d_v_3by2,