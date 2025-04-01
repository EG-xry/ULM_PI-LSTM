
###### DATE 2025.2.12-

**AUTHOR: Eric Gao**
Utilizing Python and MatLab, the research utilizes Physics Informed Neural Network (PINN) incorporated with a recurrent neural network structure, Long Short Term Memory (LSTM), to improve tracking accuracy in super resolution ulrasound localization micrscopy (srULM).

#### ABSTRACT

This research presents a novel approach to improving tracking for super-resolution ultrasound localization microscopy (srULM) by integrating Physics Informed Neural Networks (PINNs) with Long Short-Term Memory (LSTM) architectures, forming a Physics Informed LSTM (PI-LSTM) model. Recognizing the time-dependent nature of the tracking problem, our framework leverages the strengths of LSTM in capturing temporal dynamics while embedding the underlying physics, specifically, the Navier-Stokes equations, directly into the loss function. While recent advancements in localization, such as those utilizing Radial Symmetry and deep learning techniques, have significantly enhanced spatial accuracy, tracking remains a critical challenge. Traditional approaches, including nearest neighbor and Hungarian methods, are limited in robustness when dealing with complex, dynamic imaging scenarios. By incorporating physical principles into the neural network, our method not only refines the association process between successive frames but also improves overall tracking robustness and image accuracy, as quantified by the Jaccard Index. This PI-LSTM framework represents a promising step towards more accurate and reliable tracking in srULM and potentially other dynamic imaging modalities.

The research utilizes the framework and existing methodologies of data processing and localization methods of the following research:

###### DATE 2020.12.17-VERSION 1.1

**AUTHORS: Arthur Chavignon, Baptiste Heiles, Vincent Hingot. CNRS, Sorbonne Universite, INSERM.**

#### ACADEMIC REFERENCES TO BE CITED

Details of the code in the article by Heiles, Chavignon, Hingot, Lopez, Teston and Couture.
[*Performance benchmarking of microbubble-localization algorithms for ultrasound localization microscopy*, Nature Biomedical Engineering, 2022 (10.1038/s41551-021-00824-8)](https://www.nature.com/articles/s41551-021-00824-8).

General description of super-resolution in: Couture et al., [*Ultrasound localization microscopy and super-resolution: A state of the art*, IEEE UFFC 2018](https://doi.org/10.1109/TUFFC.2018.2850811
        
        
        
        ).

#### ABSTRACT

Ultrasound Localization Microscopy (ULM) is an ultrasound imaging technique that relies on the acoustic response of sub-wavelength ultrasound scatterers to map the microcirculation with an order of magnitude increase in resolution. Initially demonstrated in vitro, this technique has matured and sees implementation in vivo for vascular imaging of organs, and tumors in both animal models and humans. The performance of the localization algorithm greatly defines the quality of vascular mapping. We compiled and implemented a collection of ultrasound localization algorithms and devised three datasets in silico and in vivo to compare their performance through 18 metrics. We also present two novel algorithms designed to increase speed and performance. By openly providing a complete package to perform ULM with the algorithms, the datasets used, and the metrics, we aim to give researchers a tool to identify the optimal localization algorithm for their usage, benchmark their software and enhance the overall image quality in the field while uncovering its limits.

This article provides all materials and post-processing scripts and functions.

#### RELATED DATASET

Simulated and in vivo datasets are available on Zenodo [10.5281/zenodo.4343435
        
        ](https://doi.org/10.5281/zenodo.4343435
        
        
        
        ).

#### 1. PATH AND LOCATIONS

Before running scripts, two paths are required and have to be set in `PALA_SetUpPaths.m` to your computer environment:

- `PALA_addons_folder`: the addons folder with all dedicated functions for PALA
- `PALA_data_folder`: root path of your data folder

#### 2. EXAMPLE SCRIPT

Script name `/PALA_scripts/PALA_InVivoULM_example.m`
Data are loaded, microbubbles are detected, localized and paired to generate a list of trajectories.
The script generates 4 final images:

- Image density based on microbubbles counts: pixel intensity codes the number of microbubbles crossing this pixel
- Image density with axial color encoding: pixel intensity codes the number of microbubbles crossing upward/downward
- Velocity magnitude image: pixel intensity represents the average bubble velocity in _mm/s_
- PowerDoppler image for comparison

#### 3. MAIN SCRIPTS

For each major datasets (_In Silico PSF_, _In Silico Flow_, and _In Vivo Brain_), scripts are separated in two parts: one for processing and one for displaying. After computation, the processing script will launch displaying script. All scripts are located in `/PALA_scripts/` folder.

- Data processing scripts performing detection, localization (and tracking): (`PALA_SilicoPSF.m`, `PALA_SilicoFlow.m` , `PALA_VivoBrain.m`)
- Displaying scripts for results analysis: (`PALA_SilicoPSF_fig.m`, `PALA_SilicoFlow_fig.m`, `PALA_VivoBrain_fig.m`)

After running the 3 main scirpts, the gobal score can be computed and display with `PALA_GlobalScore.m`. This script loads scores from the 3 major datasets and computes the global score.

For the 3 supplementary in vivo datasets (_RatBrainBolus_, _MouseTumor_, and _RatKidney_), we added a routine script (`PALA_VivoMulti.m`) with few input parameters. The user can select each datasets by changing the value of `DataSetNumber`.

#### 4. TOOLBOX FOR ULTRASOUND LOCALIZATION MICROSCOPY

The 3 main functions required for ULM processing are provided in the folder `/PALA_addons/ULM_toolbox/`

- `ULM_localization2D.m`: localizing microbubbles in a stack of images
- `ULM_tracking2D.m`: pairing a list of microbubbles into a set of tracks
- `ULM_Track2MatOut.m`: generates a density image (`MatOut`) by accumulating tracks occurrences in a pixel grid

MATLAB toolboxes required to run the code:
_Communications_, _Bioinformatics_, _Image Processing_, _Curve Fitting_, _Signal Processing_, _Statistics and Machine Learning_, _Parallel Computing_, _Computer Vision Toolbox_.
