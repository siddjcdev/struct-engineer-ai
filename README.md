# The Resonance Assassin  
### Can AI Curb the Risk of Soft-Story Collapse?

## Abstract
The above project investigates how to reduce the risk of collapse of buildings with a soft-story using multiple different control methods, from fuzzy-logic to reinforcement learning. It was hypothesized that a reinforcement learning model with curriculum learning-powered TMDS would yield lower DCRs, lower RMS, and peak roof displacements, and degrade less under perturbations when compared against passive, fuzzy-logic powered, and reinforcement-learning powered TMDs with the same datasets. 
A 12-story building model with a soft-story at its 8th floor was simulated in MATLAB and tested using 8 earthquake datasets generated using real data from UCBerkley’s PEER Ground Motion Database, and the aforementioned four controllers were tested using metrics such as DCR, peak roof displacement, and efficiency under perturbations. 

The RL‑CL controller reduced peak displacement by up to **76%** relative to the passive TMD and **15%** relative to the fuzzy‑logic controller and was the most efficient under perturbations by a wide margin. However, it achieved the lowest DCR in only **1 of the 8** scenarios. These results show that while curriculum-trained RL controllers may be better at reducing peak displacement and be more efficient under perturbations and extreme scenarios than fuzzy-logic and passive TMDs, they need further training to truly prevent soft-story collapse by reducing DCR.

## 🚀 Keywords
Soft-story collapse, Tuned Mass Damper (TMD), Reinforcement Learning, Curriculum Learning, Fuzzy Logic, Earthquake Engineering, Structural Dynamics, MATLAB, PEER Ground Motion Database

