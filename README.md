# HADDL-DAS-Denoising

Official MATLAB implementation for the Hybrid Attention-Driven Deep Learning (HADDL) framework for Distributed Acoustic Sensing (DAS) data noise attenuation. 

This repository provides a fully reproducible and comprehensive workflow that executes **both the synthetic wavefield simulations and the diverse real-world field data applications** presented throughout the manuscript. It allows researchers to seamlessly evaluate framework scalability from controlled synthetic tests to high-volume field records.

## Reference
If you use this code, subroutines, or benchmark datasets in your research, please cite:
* Oboué, Y. A. S. I., Chen, Y., & Chen, Y. (2026). A Hybrid Attention-Driven Deep Learning Framework for Denoising DAS Data. *Geophysics* (Under Review).

---

## Repository Architecture

The package is organized into four primary directories to ensure modularity and ease of reproduction:

* **`Demos/`**: Contains the core high-level execution scripts for computing and plotting.
* **`Field_data/`**: Storage for real-world DAS data matrices (`.mat` format).
* **`Synth_data/`**: Storage for generated synthetic test profiles.
* **`Subroutines/`**: The complete backend engine containing filtering operators, metrics, and network helpers.

---

## Detailed File Descriptions

### 1. Main Plotting and Figure Generation Scripts (`Demos/`)
These scripts act as the wrapper environment to visualize the comparative performance across different benchmarking techniques:
* **`plot_haddl_synth_fig_5_8.m`**: Synthesizes and plots the multi-method benchmark (Figures 5, 6, 7, and 8) for synthetic datasets.
* **`plot_haddl_m..._fig_14_17.m`**: Generates benchmarking plots and comparative profiles for the `microDAS` dataset (Figures 14 to 17).
* **`plot_haddl_F...AS_fig_9_13.m`**: Generates benchmarking plots and processing profiles for the `FORGE_DAS` dataset (Figures 9 to 13).
* **`plot_haddl_jD..._fig_18_22.m`**: Generates benchmarking plots, removed noise profiles, and local similarity maps for the `jDAS` submarine dataset (Figures 18 to 22).

### 2. Framework Execution and Test Pipelines (`Demos/`)
Run these workflows to process the data matrices prior to plotting:

* **Proposed Framework:**
    * `test_haddl_synth.m` / `test_haddl_synth_fig1.m` / `test_haddl_synth_fig_3_4.m`: Runs the HADDL denoising pipeline on different synthetic data configurations.
    * `test_haddl_microDAS.m`: Applies HADDL to the high-frequency microDAS field records.
    * `test_haddl_FORGE_DAS.m`: Processes the deep geothermal reservoir DAS data from the FORGE site.
    * `test_haddl_field_jDAS.m`: Executes HADDL on the submarine jDAS dataset.

* **MHA-RN Baseline Evaluations:**
    * `test_mharn_synth.m`, `test_mharn_microDAS.m`, `test_mharn_FORGE_DAS.m`, `test_mharn_field_jDAS.m`: Run the Multi-Head Attention Residual Network processing baseline across all corresponding synthetic and field scenarios.

* **BP+SGK Baseline Evaluations:**
    * `test_sgk_synth.m`, `test_sgk_microDAS.m`, `test_sgk_field_jDAS.m`, `test_sgk_field_FORGE_DAS.m`: Execute the traditional Bandpass + Sliding Singular Value Decomposition (SGK) baseline filters.

> 💡 **Note on SSDL Benchmark Reproducibility:** The SSDL (Self-Supervised Deep Learning) comparative data fields (stored in `Output_Synth_SSDL/` and `Output_Field_*_SSDL/`) were generated using the official Python/Jupyter Notebook pipeline provided by Saad et al. (2024). The denoised outputs were exported as `.mat` matrices for seamless integration and calculation within this MATLAB evaluation suite.

### 3. Subroutines and Backend Functions (`Subroutines/`)
Core operators that drive the signal enhancement framework:
* `haddl_DL_Predict.m`: Handles deep learning inference and model execution.
* `haddl_localsimi.m`: Computes the 2D local similarity maps used to track noise orthogonality.
* `haddl_snr.m`: High-accuracy Signal-to-Noise Ratio calculation utility.
* `haddl_fk_dip.m` / `haddl_bandpass.m`: Classical f-k domain and frequency bandpass operators.
* `LO_adjnull.m` / `LO_banded_solve.m`: Mathematical solvers for localized orthogonalization routines.
* `haddl_patch.m` / `haddl_patch_inv.m` (2D & 3D): Implements optimal spatial windowing and overlapping patch reconstruction to mitigate boundary artifacts.

---

## Data Dependencies

### Field Data (`Field_data/`)
* `microDAS_data.mat`: High-density ambient and microseismic active records.
* `forgedAS_data.mat`: Wellbore monitoring dataset from the Utah FORGE geothermal site.
* `jDAS_data.mat`: Submarine dark fiber acoustic sensing profile.

### Synthetic Data (`Synth_data/`)
* `dsynthDAS3.mat` / `dsynthDAS4.mat`: Clean synthetic forward-modeled seismic wavefields.
* `dnhoriz.mat` / `dnoiseSynthDAS.mat`: Pre-computed complex random and coherent noise vectors.

---

## License
This project is licensed under the GNU General Public License v3.0 - see the `LICENSE` file for details.
