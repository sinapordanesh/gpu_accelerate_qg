// main.cpp  
#include <iostream>  
#include <cmath>  
#include <vector>  
#include <string>  
#include <chrono>  
#include "qg_solver.h"  

int main(int argc, char** argv) {  
    // Parse command line arguments or use default values  
    int N = 128;  // Default grid size  
    double beta = 0.05;  
    double dd = 5E-3;  // Linear Ekman damping  
    double epsf = 0.0;  
    int kf = 50;  
    double sigf = 1.0;  
    double alpf = 0.0;  
    double tau0 = 0.1;  // Default value, will be overridden if running parameter sweep  
    double tdel = 0.1;  // Default width of wind stress  
    std::string forcing_type = "sinh";  
    size_t Nt = 5e5;  // Number of time steps  

    std::cout << "Initializing QG Solver with N = " << N << std::endl;  
    
    // Create and initialize the solver  
    QGSolver solver(N, beta, kf, epsf, sigf, alpf, dd, tau0, tdel, forcing_type);  
    
    // Run the simulation  
    auto start_time = std::chrono::high_resolution_clock::now();  
    solver.run(Nt);  
    auto end_time = std::chrono::high_resolution_clock::now();  
    
    std::chrono::duration<double> elapsed = end_time - start_time;  
    std::cout << "Simulation completed in " << elapsed.count() << " seconds" << std::endl;  
    
    return 0;  
}  