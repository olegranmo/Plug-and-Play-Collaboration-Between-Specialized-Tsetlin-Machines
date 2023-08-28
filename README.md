# Plug-and-Play Collaboration Between Specialized Tsetlin Machines

This project introduces plug-and-play teaming of specialized Tsetlin machines (TMs), fostering collaboration. 
* The collaboration relies on a TM's ability to specialize during learning and to assess its competence during inference.
* When teaming up, the TMs that are most confident step in and make decisions, relieving the uncertain ones. In this manner, the team becomes significantly more competent than the members alone, benefitting from the specialization of each member.
* As a demonstration, the project implements four TM specializations:
  * Histogram of Gradients;
  * Adaptive Thresholding w/10x10 convolution;
  * Color Thermometers w/3x3 convolution;
  * Color Thermometers w/4x4 convolution.
* The above teaming gives the new state-of-the-art performance for TMs across the three datasets Fashion-MNIST, CIFAR-10, and CIFAR-100, increasing accuracy on Fashion-MNIST by two percentage points, CIFAR-10 by twelve points, and CIFAR-100 by nine points.

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
