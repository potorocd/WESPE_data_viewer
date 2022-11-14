# WESPE data viewer
Graphical user interface for tr-XPS data treatment from WESPE endstation

## Introduction
This program is developed for convenient manipulation of time-resolved XPS data measured with time-of-flight analyzers of WESPE at the PG2 beamline of FLASH free-electron laser. The graphical user interface is developed using an open-source, cross-platform Python framework kivy. The application works on the basis of several python objects developed for reading hdf5 files, data handling, and the creation of different kinds of visualization. 

## Installation
1) Install [Anaconda]( https://www.anaconda.com/) python distribution
2) Install kivy (in anaconda prompt)
```conda
conda install kivy -c conda-forge
```
3) Install lmfit (in anaconda prompt)
```conda
conda install lmfit -c conda-forge
```
4) Install spyder (in anaconda prompt, if it is not installed as default)
```
conda install -c anaconda spyder
```

## How to use
The main window includes four sections followed by green execution buttons:
1. Upload runs
2. Calculate delay-energy map
3. Create delay-energy map visualization
4. Slice delay-energy map data

<p align="center">
    <img align="middle" src="https://github.com/potorocd/WESPE_data_viewer/blob/main/packages/readme/Main_menu.png" alt="Main menu"/>
</p>

### Section I - Upload runs
<p align="center">
    <img align="middle" src="https://github.com/potorocd/WESPE_data_viewer/blob/main/packages/readme/Main_menu_I.png" alt="Main menu I"/>
</p>

This section serves for loading hdf5 files without any data handling.

File directory – specify the directory where data is stored

Run numbers – specify run numbers separated by coma

The files are located as ‘FileDirectory\RunNumber\RunNumber_energy.mat’

DLD4Q/DLD1Q switch – select between two TOF analyzers

The ‘App settings’ button opens a popup window with application settings.

<p align="center">
    <img align="middle" src="https://github.com/potorocd/WESPE_data_viewer/blob/main/packages/readme/App_settings.png" alt="App settings"/>
</p>

The settings are stored in the ‘packages/config.json’ file. To save changes to the file, press ‘Save settings’. On press of ‘Load default settings’, the ‘packages \default_config.json’ file is read, it can not be changed via the application window. Graph parameters can be changed without restarting the application, the change of app parameters requires doing this.

The ‘Upload runs’ button reads out data from the specified runs and creates a summary popup window. On top of it, the results of three checks are shown: 1) check if all runs are static or delay stage scans; 2) check if runs contain data from similar energy regions; 3) check if the monochromator position is the same for all runs.

<p align="center">
    <img align="middle" src="https://github.com/potorocd/WESPE_data_viewer/blob/main/packages/readme/Upload_runs.png" alt="Upload runs"/>
</p>

### Section II - Calculate delay-energy map (computationally demanding part)
<p align="center">
    <img align="middle" src="https://github.com/potorocd/WESPE_data_viewer/blob/main/packages/readme/Main_menu_II.png" alt="Main menu II"/>
</p>

This section serves for the generation of a delay-energy map using data uploaded in **Section I**. Delay-energy map is a 2D array (2D image), where the y-axis corresponds to the time domain, and the x-axis corresponds to the energy domain.

The x-axis is described with two coordinates: 1) Kinetic energy – values taken from hdf5 files; 2) Binding energy calculated as ‘mono value – Kinetic energy - 4.5 (work function of spectrometer)’. For old data, no mono values are stored in hdf5 files, therefore it is set to 0 eV. As a result, Binding energy values are negative.

The y-axis is described with one coordinate as default: ‘Delay stage values’ taken from hdf5 files. For static runs, all electrons are assigned to delay stage values equal to the run number in ps. When T0 in **Section II** is ‘on’, the second coordinate called ‘Delay relative t0’ is created. The values are calculated as ‘time zero – delay stage values’.

#### Delay-energy map parameters
***T0: ON/OFF*** – when the switch is on, the ‘Delay relative t0’ coordinate is created

***Kinetic energy/Binding energy*** – toggle determining energy coordinate for the first delay-energy map visualization
#### Binning
***Energy step*** – determines binning in the energy domain (x-axis)

***Time step*** – determines binning in the time domain (y-axis)
#### Bunch filtering
***MacroB: ON/OFF*** – switch determining if MacroBunch ID filter is applied

Limits in % are specified as two values separated by coma

***MicroB: ON/OFF*** – switch determining if MicroBunch ID filter is applied

Limits in units are specified as two values separated by coma

***Mode ‘Time delay’/’MicroBunch’*** – switch to an alternative mode that creates a 2D image, where MicroBunch ID serves as the y-axis instead of Time delay. It can be used for comparison of pumped and unpumped MicroBunches.

### Section III - Create delay-energy map visualization
<p align="center">
    <img align="middle" src="https://github.com/potorocd/WESPE_data_viewer/blob/main/packages/readme/Main_menu_III.png" alt="Main menu III"/>
</p>

This section serves for the instant generation of delay-energy map visualizations using the array generated in **Section II**.

<p align="center">
    <img align="middle" src="https://github.com/potorocd/WESPE_data_viewer/blob/main/packages/readme/Map_example.png" alt="Map example" width="600"/>
</p>

#### Plot parameters
***T0: ON/OFF*** – switch between ‘Delay stage values’ and ‘Delay relative t0’ coordinates in the time domain

***Dif map: ON/OFF*** – switch to the difference map plot where averaged energy dispersive curve before -0.25 ps is subtracted from the whole map line by line. It helps to emphasize minor variations of intensity as a function of time delay.

***Kinetic energy/Binding energy*** – toggle to select the coordinate for the energy axis

#### Delay/Energy ROI
***Energy: ON/OFF*** – when on, the cutoff for the energy axis (x-axis) is activated, where limits are determined by two values separated by coma

***Delay: ON/OFF*** – when on, the cutoff for the time axis (y-axis) is activated, where limits are determined by two values separated by coma

#### Normalizing
***Total electrons: ON/OFF*** – switch on normalization to compensate for different acquisition times for every specific time delay. Every energy dispersive curve is normalized to the total sum of electrons within the whole energy window.

***[0,1]: ON/OFF*** – normalization of every energy dispersive curve to [0,1] as [min, max].

***[-1,1]: ON/OFF*** – normalization of every energy dispersive curve to the maximal value between abs(min) and abs(max). Suitable for the difference plots, which include negative values.

***Mode: PyQt/Save Fig*** – PyQt corresponds to plot visualization in a popup window, while Save Fig creates an image in the ‘FileDir\fig_output’ folder.

***Save ASCII*** – it saves the whole delay energy map from the current visualization to ‘FileDir/ASCII_output/Maps/DateTime’. Every time delay value is saved as a separate .dat file and contains corresponding energy dispersive curve.

### Section IV - Slice delay-energy map data
<p align="center">
    <img align="middle" src="https://github.com/potorocd/WESPE_data_viewer/blob/main/packages/readme/Main_menu_IV.png" alt="Main menu IV"/>
</p>

This section serves for the instant generation of 1D slices of the 2D delay-energy maps created in **Section III** either in the time or energy domain.

<p align="center">
    <img align="middle" src="https://github.com/potorocd/WESPE_data_viewer/blob/main/packages/readme/Slice_example.png" alt="Slice example" width="600"/>
</p>

#### Slice mode
***Time axis/Energy axis*** – switch between slices across the y- or x-axis

***Waterfall: ON/OFF*** – ‘on’ selection introduces y offset between curves to avoid overlapping

***Add map: ON/OFF*** – ‘on’ selection adds map plot to slice plot where the regions used for generation of slices are displayed

***Legend: ON/OFF*** – show or hide the legend of the slice plot

#### Slice parameters
***Position*** – specification of the slice positions separated with commas; ‘Main’ instead of a number automatically finds the position of the most intense feature; ‘SB, hv’ will determine the sideband position by the addition of ‘hv’ to the position of the most intense feature

***Width*** – specification of slice widths separated with commas; if one value is specified, it will be applied to every slice

***Difference: ON/OFF*** – ‘on’ selection adds additional curves to the plot calculated as a difference of every curve starting from the second one and the first one

#### Data treatment
***Norm [0,1]: ON/OFF*** – ‘on’ selection normalizes the curves to [0,1] as [min, max]

***Norm [-1,1]: ON/OFF*** – ‘on’ selection normalizes the curves to the maximal value between abs(min) and abs(max).

***Smoothing: ON/OFF*** – ‘on’ selection applies the Savitzky-Golay filter to all slices

***Derivative: ON/OFF*** – ‘on’ selection applies the first derivative to all slices, which can be helpful for determination of time zero

***Voigt fit*** – it performs fitting of the first slice with single Voigt fit, which helps to determine time zero

<p align="center">
    <img align="middle" src="https://github.com/potorocd/WESPE_data_viewer/blob/main/packages/readme/Voigt_fit_example.png" alt="Voigt fit example" width="600"/>
</p>

***Mode: PyQt/Save Fig*** – PyQt corresponds to plot visualization in a popup window, while Save Fig creates an image in the ‘FileDir\fig_output’ folder.

***Save ASCII*** – it saves all slices from the current visualization to ‘FileDir/ASCII_output/Cuts/DateTime’.

