# Simulator-based surrogate optimisation

Data and code corresponding to the research paper: _Simulator-based surrogate optimisation employing adaptive uncertainty-aware sampling_ (https://doi.org/10.1016/j.compchemeng.2025.109243).

Two case studies are included to demonstrate the method:

- Case 1: Having access to the simulator - Blade design

  - Objective: Optimise the thermal design of a jet engine turbine blade

  - Simulator: Thermal analysis using Finite Element Analysis (FEA) via MATLAB's Partial Differential Equation Toolbox

  - Reference: Based on MATLAB's [Thermal Stress Analysis of Jet Engine Turbine Blade](https://se.mathworks.com/help/pde/ug/thermal-stress-analysis-of-jet-engine-turbine-blade.html) example

- Case 2: Being limited to pre-acquired datasets - Auto-thermal reformer

  - Objective: Optimise the operating conditions of an auto-thermal reformer

  - Data source: 2,800 pre-simulated data points from the [OMLT repository](https://github.com/cog-imperial/OMLT/blob/dfe44bd59232996d821d9f8eb6d1b8f2d8aa1c3a/docs/notebooks/data/reformer.csv)

  - Reference: Optimisation setup follows the [OMLT implementation](https://github.com/cog-imperial/OMLT/blob/dfe44bd59232996d821d9f8eb6d1b8f2d8aa1c3a/docs/notebooks/neuralnet/auto-thermal-reformer-relu.ipynb)
