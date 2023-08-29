# Plug-and-Play Collaboration Between Specialized Tsetlin Machines

This project introduces plug-and-play collaboration between Tsetlin machines (TMs). 
* The collaboration relies on a TM's ability to specialize during learning and to assess its competence during inference.
* When teaming up, the most confident TMs step in and make the decisions, relieving the uncertain ones. In this manner, the team becomes more competent than its members, benefitting from their specializations.
* The members can be combined in any manner, at any time, without any fine-tuning (plug-and-play).
* The project implements four TM specializations as a demonstration:
  * Histogram of Gradients;
  * Adaptive Thresholding w/10x10 convolution;
  * Color Thermometers w/3x3 convolution;
  * Color Thermometers w/4x4 convolution.
    
Working as a team increases accuracy on Fashion-MNIST by two percentage points, CIFAR-10 by twelve points, and CIFAR-100 by nine points, yielding new state-of-the-art results for TMs.

## Architecture

The plug-and-play architecture is shown below.
<p align="center">
  <img width="60%" src="https://github.com/olegranmo/Plug-and-Play-Collaboration-Between-Specialized-Tsetlin-Machines/blob/main/TeamOfSpecialists.png">
</p>

<p align="left">
As seen, each member TM first outputs its class sum <img src="http://latex.codecogs.com/svg.latex?\hat{y}^{i,c}" border="0" valign="middle"/> per class <i>i</i>, with the class sum signifying confidence:
</p>

<p align="center">
<img src="http://latex.codecogs.com/svg.latex?\hat{y}^{i,c} = \sum_{j=1}^{n/2} C_j^{i,+}(X) - \sum_{j=1}^{n/2} C_j^{i,-}(X)" border="0" valign="middle"/>.
</p>

<p align="left">
In the normal case, a TM outputs the class with the largest confidence:
</p>

<p align="center">
<img src="http://latex.codecogs.com/svg.latex?"\hat{y} = \mathrm{argmax}_{i}\left(\sum_{j=1}^{n/2} C_j^{i,+}(\mathbf{X}) - \sum_{j=1}^{n/2} C_j^{i,-}(\mathbf{X})\right)" border="0" valign="middle"/>.
</p>



In the ensuing normalization step, the TM's class sums are divided by the difference between the largest and smallest class sums. The normalized class sums, in turn, are added together, forming the class sums of the team as a whole. The maximum value of these decides the class output in the final step.

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
