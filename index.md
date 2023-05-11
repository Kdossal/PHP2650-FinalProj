<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/katex@0.10.2/dist/katex.min.css" integrity="sha384-yFRtMMDnQtDRO8rLpMIKrtPCD5jdktao2TV19YiZYWMDkUR5GQZR/NOVTdquEx1j" crossorigin="anonymous">
<script defer src="https://cdn.jsdelivr.net/npm/katex@0.10.2/dist/katex.min.js" integrity="sha384-9Nhn55MVVN0/4OFx7EE5kpFBPsEMZxKTCnA+4fqDmg12eCTqGi6+BB2LjY8brQxJ" crossorigin="anonymous"></script>
<script defer src="https://cdn.jsdelivr.net/npm/katex@0.10.2/dist/contrib/auto-render.min.js" integrity="sha384-kWPLUVMOks5AQFrykwIup5lo0m3iMkkHrD0uJ4H5cjeGihAutqP0yW0J6dpFiVkI" crossorigin="anonymous" onload="renderMathInElement(document.body);"></script>


# **What is Reinforcement Learning?**

We’ve previously discussed supervised learning, which is used for tasks where our data is labeled or has true values, and unsupervised learning, which is used to discover patterns or relationships in unlabeled data. The last major pillar of machine learning is Reinforcement Learning, which is used to solve problems for which we don’t have examples of gold-standard outcomes. Unlike supervised learning, where we train our model using comparisons to true outcome values, reinforcement learning involves the use of numerical rewards and punishments to adjust our model. 

For our final project, we created a reinforcement learning model that takes in medical information from simulated sepsis patients and makes a series of treatment decisions with the goal of stabilizing the patient so that they do not die of septic shock. Before discussing the specifics of our model implementation, let’s review some reinforcement learning basics in the context of the problem we want to solve with our project. 

The most basic type of reinforcement learning has 4 elements: 
**Agent:** The agent is the decision-making model that interacts with our environment. In our case, this is our Deep Q-Network that takes in information about an environment, called a state, and selects an action based on the information in that state. This agent will then over time learn from the feedback it receives in the form of rewards or punishments.
**Environment:** The environment in our model is the simulated patient, with a state that contains information concerning patient vitals (such as their blood pressure, temperature, or heart rate), as well as patient demographics when they enter the treatment facility. After each action the agent takes, the environment’s state changes accordingly. 
**Actions:** A set of actions that the agent can take. In our case, the model can take five possible actions: administer antipyretics, antibiotics, fluids, vasopressors, or oxygen therapy. The goal of reinforcement learning is to train the model such that the action it takes in response to an input state is the optimal action for that situation. For example, if the input state to our model was a patient with a high fever, our goal would be for the model to take the action: administer antipyretics. 
**Reward:** After an action transitions the environment from state $$s$$ to state $$s’$$, the model receives a reward (or punishment) depending on whether $$s’$$ is a desired outcome or not. In our model, the reward is set to 10 if an action leads to the patient being stabilized (stage 0), -10 if the patient is dead (as defined by vitals or the number of time steps the patient has spent with sepsis), and -1 if the patient is alive but not stabilized. These values were chosen to encourage the agent to quickly stabilize patients without overtreating them.

Here’s a visualization of how the four elements interact with each other:

| <figure> <img src="images/rl_visual.png" /> </figure> | 
|---| 
| *Figure 1* |

Now that we’ve provided a broad overview of the reinforcement learning paradigm, let’s discuss the specific type of reinforcement learning we use in our project: Deep Q-Learning.

# **What is Deep Q-Learning?**

Deep Q-Learning is a reinforcement learning algorithm that combines Deep Learning with Q-learning, an algorithm that learns to find optimal actions by estimating the maximum expected cumulative reward for each action in a given state, called the Q-values. The one setback with Q-learning is that estimating Q-values can become increasingly difficult as our task becomes more complex, this can be remedied by using a Deep Learning model to estimate our Q-values instead.

We decided to use DQL over other reinforcement learning algorithms as we want our agent to learn not just from what happened in the last iteration of the simulation, but from previous iterations as well. Other algorithms like transition probability-based models are appropriate when the environment we’re interacting with conforms to the Markov property, which can be represented by the following equation:

| <figure> <img src="images/transition_prob.png" width="700" height="53" />  </figure> | 
|:--:| 
| *Figure 2* |

The Markov property says that given the current state of the environment, there’s a fixed probability of transitioning to any other state in the state space in the next time step and that it doesn’t matter what happened before the current state. Think about this formula in terms of a chess game. If the pieces are in a certain configuration, there’s a fixed set of possible moves for the next step (even though this set is very large), and it doesn’t really matter what sequence of moves got the chess board in this configuration. 

However, patient health isn’t like a simple game of chess. We not only want patients to recover, we want them to recover as fast as possible, which means it’s important to have prior information on what worked most efficiently. For example, if one patient was treated in 12 time steps using just antipyretics, but another was treated in 4 steps using a more aggressive treatment approach, we’d want the model to recall that and use that information to update its approach. As such, a transition probability-based model would not be appropriate for our context. 

## **Training**

| <figure> <img src="images/DQN.png" width="700" height="562" /> </figure> | 
|:--:| 
| *Figure 3* |

Pictured above is the algorithm that Deep Q-Learning uses to update model weights and teach the agent how to perform in the environment. Before delving into the specifics of the algorithm, let’s visualize broadly what is happening: 

| <figure> <img src="images/deep_Q.png" /> </figure> | 
|:--:| 
| *Figure 4* |

We start by initializing two networks, our target and prediction networks, using identical model architectures and randomly generated weights. We use the target network as a helper to ensure stability and prevent divergence in our prediction network during training. We also initialize a memory that with each iteration of the algorithm, will be populated with the following information: current state, action taken, reward received, state transitioned to via the action, and whether or not an episode is finished. An episode refers to one round of iterations; in our case, an episode corresponds to the treatment of a patient from start to finish, which involves multiple iterations of the algorithm or epochs. 

| <figure> <img src="images/network_basic.png" /> </figure> | 
|:--:| 
| *Figure 5* |

As the above picture shows, in a feedforward neural network, we input the starting data (our input state), it is multiplied by weights $$W$$, then passed through an activation function to get an output from the first layer. The process repeats, with each layer taking in as input the output from the previous layer. Although we start with our weights being randomly sampled, our goal is to update these weights with each epoch so the output becomes as accurate as possible using backpropagation and gradient descent. Now let’s dive into how we update our weights for a Deep Q-learning model. 

Since the model is dependent on using prior information, we have to figure out how we’re going to use the memory that we created for our model and has been populating over the course of the runtime of the algorithm. Due to computation concerns, we must set a maximum size on the memory of the model, populating it with new memories and erasing the oldest. 

An intuitive assumption is that it’d be best for our agent to reference the entire content of the memory each time it has to make an evaluation using prior information. After all, humans tend to make wiser decisions the more experience they have to draw from. However, the issue is that at the beginning of the training process, when our model is exploring the state space, it is probably going to make a lot of bad decisions. Think about this: if you had a two-year-old choosing stocks for your retirement portfolio, their decisions would probably be as good as random at best. As such, rather than referencing the entire memory with each epoch, we choose a batch size of elements from the memory to sample for use in finding estimates of future rewards. Ideally, we choose our batch size so that it’s big enough to be representative of the memory and hopefully capture some episodes that were played fairly optimally, but small enough that the runtime of our algorithm training is reasonable. Once we have trained the agent for enough epochs such that the size of the memory is greater than the batch size, we can start sampling from the memory, and we do so for every following epoch during training. This process of randomly sampling past experiences from memory and training the model on these experiences is known as **experience replay**.

| <figure> <img src="images/loss.png" width="500" height="168" /> </figure> | 
|:--:| 
| *Figure 6* |

Experience replay iterates through the sampled memories and uses the formula pictured above to calculate our *loss* that we will use to update model weights, for our model we used **MSE** as our loss function. Let’s break down each of the key parameters. Firstly, $$r$$ refers to the rewards received after taking the chosen action. Next the decay/discount rate, $$\gamma$$, is chosen depending on the specific problem you’re trying to solve. Since gamma weights our predictive estimates of the reward using prior information, setting a lower value for gamma (close to 0) tells the model that we only care about maximizing short-term reward, while setting a higher value (close to 1) prioritizes maximizing future/long-term gains. More concretely, if you were setting up a Deep Q-Learning algorithm to choose how to allocate your money in investments for your retirement, you’d want to set a very high value of gamma – since you’re not touching that money until the “long term,” maximizing future gains is much more important. Next are our estimated Q values $$\hat{Q}$$ represents the maximum Q-value estimated by our Target model, and $$Q$$ represents the value estimated by our prediction model. 

Additionally in our model we also specify $$\alpha$$ and $$\epsilon$$. The learning rate $\alpha$ is set depending on how much the weights of the neural network are to be updated in response to our loss. $$\epsilon$$ refers to our exploration rate and allows us to balance between exploration and exploitation. When beginning training we start with an $$\epsilon$$ value close to one (exploration) that gradually decreases as we train (exploitation). At $$\epsilon = 1$$, we force the agent to take random actions in hopes that we will be able to explore our environment, while at $$\epsilon = 0$$ our agent is taking actions entirely based on prior information to exploit the environment. 

Now that we’ve gone over updating our model weights, why did we originally initialize two models? Going back to *Figure 6* the purpose of taking the difference between our target and predicted Q-values is to stabilize training, but how does it do this? During training our prediction model is updated after every epoch, however, we do not want to use the same approach for our target model. This is because during the long training process, the agent may make many poor decisions while exploring the state space. To limit how much our prediction model is exposed to poor decision-making, we specify a period or number of epochs after which we update the weights in the target model to match the current weights in our prediction model. This allows the prediction model to learn from its mistakes while not passing those mistakes onto the target. By using two models, we can ensure that our final model is accurate while also being exposed to representative samples of data.

This process of interacting with the environment, storing memories and training the model by replaying through previous experiences is how a Deep Q-Network learns and overtime the agent will hopefully find optimal policies, in our case how to treat patients. 
