// qg_solver.h  
#pragma once  

#include <string>  
#include <vector>  
#include <complex>  
#include <cuda_runtime.h>  
#include <cufft.h>  
#include <curand.h>  

// Parameter struct (will be copied to constant memory)  
struct QGParams {  
    double beta;  
    int sh;  
    double dd;  
    double nu;  
    int N;  
    int kf;  
    double epsf;  
    double sigf;  
    double alpf;  
    double tau0;  
    double tdel;  
};  

class QGSolver {  
public:  
    QGSolver(int N, double beta, int kf, double epsf, double sigf, double alpf,   
             double dd, double tau0, double tdel, std::string forcing_type);  
    ~QGSolver();  
    
    void run(size_t Nt);  

private:  
    // Host parameters  
    QGParams h_params;  
    std::string forcing_type;  
    double qlim;  // Limit for q values  
    
    // Dimensions  
    int N;  
    int complexSize;  // Size of complex arrays  
    
    // Host arrays for initialization and diagnostics  
    std::vector<std::complex<double>> h_q;  
    std::vector<double> h_qp;  
    std::vector<double> h_X, h_Y;  
    
    // Time tracking  
    double t;  
    double dt;  
    double tol;  
    double r0;  
    
    // Diagnostic arrays  
    std::vector<double> T;  
    std::vector<double> vb;  
    std::vector<std::vector<double>> utz;  
    std::vector<std::vector<double>> energy;  
    std::vector<std::vector<double>> enstrophy;  
    std::vector<double> dtsize;  
    
    // Statistics counters  
    int count;  
    
    // Device memory pointers  
    cufftDoubleComplex *d_q;        // Vorticity in spectral space  
    cufftDoubleComplex *d_psi;      // Stream function in spectral space  
    cufftDoubleComplex *d_tauk;     // Wind stress in spectral space  
    double *d_qp;                   // Vorticity in physical space  
    double *d_L;                    // Linear operator  
    cufftDoubleComplex *d_Fk;       // Stochastic forcing spectrum  
    
    // Intermediate arrays for ARK4 time stepping  
    cufftDoubleComplex *d_k0, *d_k1, *d_k2, *d_k3, *d_k4, *d_k5;  
    cufftDoubleComplex *d_l0, *d_l1, *d_l2, *d_l3, *d_l4, *d_l5;  
    cufftDoubleComplex *d_q1, *d_q2, *d_q3, *d_q4, *d_q5;  
    cufftDoubleComplex *d_M;        // Implicit operator  
    
    // Arrays for stochastic forcing  
    cufftDoubleComplex *d_dW;       // Random noise  
    cufftDoubleComplex *d_xik;      // Scaled noise  
    
    // RHS computation intermediates  
    cufftDoubleComplex *d_RHS;  
    
    // 3/2 sized arrays for dealiasing  
    cufftDoubleComplex *d_Psi_hat_3by2;  
    cufftDoubleComplex *d_Q_hat_3by2;  
    double *d_u_3by2, *d_v_3by2, *d_qx_3by2, *d_qy_3by2, *d_jaco_real_3by2;  
    cufftDoubleComplex *d_Jaco_hat_3by2;  
    
    // Maximum finder for adaptive time stepping  
    double *d_max_array;  
    double *d_r1;  
    
    // CUDA handles  
    cufftHandle fft_plan, ifft_plan;  
    cufftHandle fft_plan_3by2, ifft_plan_3by2;  
    curandGenerator_t rng;  
    
    // Output data  
    std::string data_folder;  
    FILE *qh_file, *t_file;  
    
    // Methods  
    void initialize();  
    void setupOperators();  
    void setupStochasticForcing();  
    void setupWindStress();  
    
    void computeRHS(cufftDoubleComplex *q, cufftDoubleComplex *rhs);  
    void timeStep();  
    void adaptTimeStep(double r1);  
    void computeDiagnostics(size_t step_idx);  
    void saveData();  
    void cleanup();  
};  