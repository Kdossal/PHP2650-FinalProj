# PHP2650 Final Project

## Sepsis Patient Treatment Using Reinforcement Learning 
For our final project, we have chosen to explore Reinforcement Learning by creating a Deep Q-Learning model that determines the optimal treatment policies for patients with sepsis. Our goal is to develop an agent that can learn optimal treatment policies over time by simulating a patient with sepsis in a controlled environment.

To create our patient simulation environment, we used real-world data and vitals to understand the effects of sepsis on patients. The simulation we created is only a toy model that takes in limited patient information and simulates only a small amount of the effects of sepsis. The simulation generates a random patient with sepsis, including age, sex, and stage of sepsis, and then uses these to generate beginning vitals for the patient. This information is then used as our state input for our Deep Q-Network.

The DQN then suggests a treatment option, taken from common treatments for sepsis in the real world, such as administering antipyretics, antibiotics, fluids, vasopressors, and oxygen therapy. Each treatment has an effect on the patient's vitals and is simulated within our patient environment. The primary aim of our agent is to stabilize each patient, rather than allow them to move through the stages of sepsis and eventually pass away. 

To train our agent, we simulated the treatment of 1250 episodes, where each episode follows one patient's treatment until they are either stabilized or pass away. We then tested our agent, comparing its performance to typical mortality and stabilization rates for sepsis in the real world.

On our GitHub page, we provide a brief introduction to reinforcement learning and deep Q-learning and illustrate the benefits of these methods by providing an application to patient treatment. We also walked through our simulation and explained the architecture of our model as well as the structure of the training process. Finally, we interpreted our results, described the limitations of our model, and explained how similar models could be extended in real-world contexts.


**Website link: https://kdossal.github.io/PHP2650-FinalProj/**


