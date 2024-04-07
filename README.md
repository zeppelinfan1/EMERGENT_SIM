ECO_EVO_SIM
Ecosystem & Evolution Simulation
A simulation where subjects are contained within a 2D environment and must survive by interacting with objects. The success of such interactions is dictated by genetics specific to each subject. A successful subject will pass on their genetics to the next generation.

Subjects

Each subject has a neural network brain with random weights. The input for the network is based on perceiving the surrounding environment i.e. nearby objects, other subjects, actions being performed nearby and the survivability result.

As the simulation starts, there is no history to base the simulation on. Subjects choose an action regardless based on the random weightings of the neural network brain. The success or failure of each action is dictated by the subject's genetics. The result of the action is implemented in the ecosystem i.e. an object is used/modified/removed or likewise to a subject.

Objects

All objects are designed according to the consumption creation axis. An object occupies a unique space on the axis. Genetic information is tailored to a specific space.




Building out original ECO EVO SIM idea in Python before translating to C++


To Do:

1) Revert back to using a dictionary to keep track of main data
   a) Need to find most efficient way
   b) Periodically store updates in DB (after each turn) but don't always rely on pulling from DB to refresh operational data

2) Start setting up system for NN brain to interact with the genes
   a) Need to build out environment first
   b) Have movement within the environment associated with the genes somehow
