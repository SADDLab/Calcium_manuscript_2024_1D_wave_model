<h1 align="center">Local traveling waves of cytosolic calcium elicited by defense signals or wounding are propagated by distinct mechanisms </h1>


<p align="center">
  <img src=".png" alt="" width="600">
</p>

## Details of experimental data analysis


## Details of modeling components

### Library Description /  Directory Organization Details

### A. main-standalone-notebook

The "main-standalone-notebook" folder contains a jupyter notebook "1D-CICR-model-v0.ipynb" for simulation of CICR mediated waves in 1D. Following python libraries are required:

- numpy (1.26.4), Scipy (1.11.4), Pandas (2.1.4), Matplotlib (3.8.0), Seaborn (0.12.2), Surrogate Modelling Toolbox (2.7.0), scikit-learn (1.2.2), numba (0.59.0)

### B. parameter-screening

The "parameter-screening" folder contains following:

1. `model-wo-sink/main_v4.py`: The code performs a Latin Hypercube Sampling (LHS) of model parameters. For each parameter set, we run the model and save the wave statistics inlcuding distance travelled by wave and wave velocity.  
2. `model-w-sink/main_v5.py`: In this code we use the model with a sink to vary the model parameters $V_{\text{sink, max}}$ and $K_M$ together within speciied ranges. We compute the wave velocity, distance travelled by the wave and the $R^2$ value of a linear model fit to the location of calcium channel against their activation time. 
2. `model-w-sink/main_v6.py`: In this codee we add sink to the model. A sink was added to all the parameter sets that generated wave velocitoes within biological range of 1um flg22 treatment fro a model withpout sink. We compute the wave velocity, distance travelled by the wave and the R2 value of a linear model fit to the location of calcium channel against their activation time.


### C. data-analysis-notebook

The "data-analysis-notebook" folder contains the following:

1. `analysis-notebook-1.ipynb`: The notebook contains plots for Figure 5 and Supplementary Figure 10 of the manuscript
2. `analysis-notebook-2.ipynb`: The notebook contains plots for Figure 6 and Supplementary Figure 11 of the manuscript


## Contact

For inquiries related to the code, please contact:

Dr. Weiwei Zhang
Senior Scientist
Plant Cell Biologist
Department of Biological Sciences
Purdue University, West Lafayette, IN
Email: zhan2190@purdue.edu