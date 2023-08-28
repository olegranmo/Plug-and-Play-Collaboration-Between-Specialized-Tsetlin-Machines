# Plug-and-Play Collaboration Between Specialized Tsetlin Machines

I here propose an approach to teaming specialized TMs that collaborate. 
* We first investigate how well a TM's classification confidence~\cite{abeyrathna2020confidence} corresponds to its ability to classify images accurately. When the TM is confident, does it get high accuracy, and when it is uncertain, does it get low accuracy?
* We next provide empirical evidence that for CIFAR-10 and CIFAR-100, a single TM becomes a specialist rather than a generalist. That is, it specializes to get high accuracy on a subset of the data, with the subset being decided by how we booleanize the input. 
* After establishing that the TM specializes and that high confidence corresponds to high accuracy, we propose a novel approach for plug-and-play collaboration between specialized TMs. The approach is plug-and-play because independently pre-trained TMs can be connected at any time and in any combination, without any fine-tuning or further training. Their individual confidences are simply normalized and aggregated into a team decision.
* We finally evaluate the team performance on Fashion-MNIST, CIFAR-10, and CIFAR-100, reporting a percentage increase of $2$ points for Fashion-MNIST, 12 points for CIFAR-10, and 9 points for CIFAR-100. In conclusion, our team-based approach sets the new state-of-the-art performance for TMs across the three datasets.

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
