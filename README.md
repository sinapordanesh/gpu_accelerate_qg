# A GPU-Accelerated code for Quasi-Geostrophic simulation 

CUDA-accelerated implementation of a Quasi-Geostrophic (QG) flow simulation using spectral methods with adaptive RK4 time integration. Originated from a MATLAB code implementation. 
The acceleration methodology follows the proposed method in [A GPU-accelerated simulation of rapid intensification of a tropical cyclone with observed heating](https://www.researchgate.net/publication/390668201_A_GPU-accelerated_simulation_of_rapid_intensification_of_a_tropical_cyclone_with_observed_heating_Journal_Title_XXX1-10).

## Requirements

- CUDA Toolkit 11.0+ (with cuFFT and cuRAND libraries)
- C++17 compatible compiler (GCC 7+ or equivalent)
- CMake 3.10+

## Build Instructions

- Build the project:

```bash
mkdir build
cd build
cmake ..
make
```

## Usage

Run the simulation with default parameters:

```bash
./qg_solver
```

Or specify custom parameters by modifying `main.cpp`.

### Key Parameters

The simulation can be configured with the following parameters:

- `N`: Grid size (default: 128)
- `beta`: Beta parameter (default: 0.05)
- `dd`: Ekman damping coefficient (default: 5E-3)
- `tau0`: Wind stress amplitude (default: 0.1)
- `tdel`: Wind stress width (default: 0.1)
- `forcing_type`: Type of forcing ("sin", "sinh", or "constant")
- `Nt`: Number of time steps (default: 5e5)

To implement command-line parameter handling, modify `main.cpp` to parse `argc`/`argv` arguments.

## Output

The simulation outputs are saved in a `data/` directory with the following structure:

- `qh.bin`: Vorticity field in spectral space (binary format)
- `t.bin`: Time values (binary format)
- `diagnostics.mat`: Energy, enstrophy, and other diagnostic quantities

## Performance

This CUDA implementation offers significant speedup compared to CPU-based solvers, particularly for large grid sizes. On an NVIDIA V100 GPU, expect approximately 10-20x performance improvement over equivalent CPU code.


