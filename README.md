# TMComposite: Plug-and-Play Collaboration Between Specialized Tsetlin Machines

This project introduces plug-and-play collaboration between Tsetlin machines (TMs), referred to as a TM Composite. 
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

In the normal case, a TM <img src="http://latex.codecogs.com/svg.latex?t" border="0" valign="middle"/> outputs the class <img src="http://latex.codecogs.com/svg.latex?i" border="0" valign="middle"/> with the largest class sum (see [Tsetlin machine](https://github.com/cair/TsetlinMachine)):

<p align="center">
<img src="http://latex.codecogs.com/svg.latex?\hat{y}_t = \mathrm{argmax}_{i}\left(\sum_{j=1}^{n/2} C_{t,j}^{i,+}(\mathbf{X}) - \sum_{j=1}^{n/2} C_{t,j}^{i,-}(\mathbf{X})\right)" border="0" valign="middle"/>.
</p>

<p align="left">
When collaborating in a team, however, each TM team member outputs its class sums <img src="http://latex.codecogs.com/svg.latex?c^i_t" border="0" valign="middle"/>, with the class sum <img src="http://latex.codecogs.com/svg.latex?c^i_t" border="0" valign="middle"/> signifying the confidence of TM <img src="http://latex.codecogs.com/svg.latex?t" border="0" valign="middle"/> in class <img src="http://latex.codecogs.com/svg.latex?i" border="0" valign="middle"/>:
</p>

<p align="center">
<img src="http://latex.codecogs.com/svg.latex?c^i_t = \sum_{j=1}^{n/2} C_{t,j}^{i,+}(X) - \sum_{j=1}^{n/2} C_{t,j}^{i,-}(X)" border="0" valign="middle"/>.
</p>

In the ensuing normalization step, the class sums of each TM <img src="http://latex.codecogs.com/svg.latex?t" border="0" valign="middle"/> are divided by the difference <img src="http://latex.codecogs.com/svg.latex?\alpha_t" border="0" valign="middle"/> between the largest and smallest class sums in the data <img src="http://latex.codecogs.com/svg.latex?\mathcal{D}" border="0" valign="middle"/>: <img src="http://latex.codecogs.com/svg.latex?\alpha_t = \mathrm{max}_{d,i}(c^i_{t,d}) - \mathrm{min}_{d,i}(c^i_{t,d})" border="0" valign="middle"/>.

The normalized class sums, in turn, are added together, forming the class sums of the team as a whole. The maximum value of these decides the class output in the final step.

## Results

### Does the Tsetlin Machine Know When it Does Not Know?

<p align="center">
  <img width="50%" src="https://github.com/olegranmo/Plug-and-Play-Collaboration-Between-Specialized-Tsetlin-Machines/blob/main/CIFAR100ThresholdingAccuracyClassSum.png">
</p>

Using Adaptive Thresholding on CIFAR-100 as an example, the above figure relates the accuracy of the TM to its confidence. Along the x-axis, we rank the 10000 test images from CIFAR-100 from lowest to highest class sum (confidence). The y-axis shows the test accuracy for the x-axis confidence level and upwards. When confidence is low, the TM operates at low accuracy. As confidence increases, accuracy increases as well, up to 100% accuracy at the highest confidence levels.

### Is the Tsetlin Machine a Generalist or a Specialist?

<p align="center">
  <img width="50%" src="https://github.com/olegranmo/Plug-and-Play-Collaboration-Between-Specialized-Tsetlin-Machines/blob/main/CIFAR100ThresholdingAnalysis.png">
</p>
We next investigate a selection of individual images at the lower and higher ends of the confidence spectrum. The images demonstrate that the Adaptive Thresholding TM has specialized in recognizing larger pixel structures, operating at high confidence and accuracy. For images that are mainly characterized by color texture, on the other hand, it suffers from low confidence and accuracy.

### Can Specialist Tsetlin Machines Successfully Collaborate?

<p align="center">
  <img width="50%" src="https://github.com/olegranmo/Plug-and-Play-Collaboration-Between-Specialized-Tsetlin-Machines/blob/main/CIFAR100ThermometersAnalysis.png">
</p>

Now consider the confidence and accuracy of the Color Thermometers TM in the figure above. Notice how it specializes to get high confidence and accuracy in recognizing color texture. The weakness of the Adaptive Thresholding TM seems to be the strength of the Color Thermometers TM, and vice versa. 

<p align="center">
  <img width="60%" src="https://github.com/olegranmo/Plug-and-Play-Collaboration-Between-Specialized-Tsetlin-Machines/blob/main/Results.png">
</p>

We are now ready to deploy our plug-and-play architecture for TM collaboration. The above table shows the accuracy of various team compositions for Fashion-MNIST, CIFAR-10, and CIFAR-100. Observe how adding team members improves accuracy. Indeed, the full team increased accuracy on Fashion-MNIST by two percentage points, CIFAR-10 by twelve points, and CIFAR-100 by nine points, yielding new state-of-the-art results for TMs.

<p align="center">
  <img width="50%" src="https://github.com/olegranmo/Plug-and-Play-Collaboration-Between-Specialized-Tsetlin-Machines/blob/main/CIFAR10Epochs.png">
</p>

Considering accuracy epoch-by-epoch for CIFAR-10, the team performance is superior already from the first epoch and stays ahead throughout the learning process.

## Further Research

The plug-and-play collaboration between TMs opens several research paths ahead:
* What other specializations (image processing techniques) can we add to the team?
* Can we design a light optimization layer that boosts the collaboration accuracy, e.g., by weighting the specialists based on their performance?
* Are there other ways to normalize and integrate the perspective of each TM?
* Can we find a way to fine-tune the TM specialists to enhance collaboration further?
* What is the best way to organize a library of composable pre-trained Tsetlin machines?
* How can we compose the most efficient team with a given size?
* Does this approach extend to speech?

## Paper

_Plug-and-Play Collaboration Between Specialized Tsetlin Machines_. Ole-Christoffer Granmo, 2023.

## Licence

Copyright (c) 2023 Ole-Christoffer Granmo

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
