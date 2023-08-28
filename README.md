# Plug-and-Play Collaboration Between Specialized Tsetlin Machines

This project introduces plug-and-play teaming of specialized Tsetlin machines (TMs), fostering collaboration. 
* The collaboration relies on the ability of TMs to specialize during learning and to assess their own expertise during inference.
* When teaming up, the TMs that are most confident step in and make decisions, relieving those less confident. In this manner, the team becomes significantly more competent than the members alone.
* As a demonstration, the project implements four TM specializations:
  * Histogram of Gradients;
  * Adaptive Thresholding w/10x10 convolution;
  * Color Thermometers w/3x3 convolution; and
  * Color Thermometers w/4x4 convolution.
* A team of four TMs increases accuracy on Fashion-MNIST by two percentage points, accuracy on CIFAR-10 by , and nine points for CIFAR-100.

In conclusion, the team-based approach sets the new state-of-the-art performance for TMs across the three datasets.

## Architecture
<p align="center">
  <img width="60%" src="https://github.com/olegranmo/Plug-and-Play-Collaboration-Between-Specialized-Tsetlin-Machines/blob/main/TeamOfSpecialists.png">
</p>

## Results

<p align="center">
  <img width="50%" src="https://github.com/olegranmo/Plug-and-Play-Collaboration-Between-Specialized-Tsetlin-Machines/blob/main/CIFAR100ThresholdingAnalysis.png">
</p>

<p align="center">
  <img width="50%" src="https://github.com/olegranmo/Plug-and-Play-Collaboration-Between-Specialized-Tsetlin-Machines/blob/main/CIFAR100ThermometersAnalysis.png">
</p>

<p align="center">
  <img width="60%" src="https://github.com/olegranmo/Plug-and-Play-Collaboration-Between-Specialized-Tsetlin-Machines/blob/main/Results.png">
</p>

<p align="center">
  <img width="50%" src="https://github.com/olegranmo/Plug-and-Play-Collaboration-Between-Specialized-Tsetlin-Machines/blob/main/CIFAR10Epochs.png">
</p>

## Paper

_Plug-and-Play Collaboration Between Specialized Tsetlin Machines_. Ole-Christoffer Granmo, 2023.
