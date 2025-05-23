Neural Network-Driven Emergent Simulation where subjects learn through interaction, perception, and observation, making decisions based on genetics, environmental stimuli, and adaptive reinforcement learning.

Key Concepts & Features
1. Environment & Terrain
The simulation takes place in a 2D grid-based environment with squares representing different terrain types.
Land squares are neutral, while Holes are lethal, causing subjects to die if they step on them.
Objects exist in the environment and have unique properties, but their usage is organically determined.

2. Subjects & Genetics
Each subject is assigned randomly generated binary genetics, which influence their decision-making.
Genes are mapped onto a scatter plot with two axes: Creation/Destruction and Energy Consumed/Produced.
The genetics system allows for organic function emergence, meaning actions (movement, object interactions, etc.) arise naturally rather than being predefined.

3. Neural Network Brain
Each subject is equipped with a neural network brain, initialized with random weights.
The network improves over time by observing previous decisions and their consequences (e.g., avoiding holes after seeing others die).
The Softmax output determines movement—subjects decide which adjacent square to move into or whether to stay still.

4. Perception & Decision-Making
Subjects perceive their surroundings within a specified range (including terrain, objects, and other subjects).
This perception data is stored in a database, allowing for historical learning across iterations.
The neural network is trained using both current perception and past experiences, enabling subjects to refine their decision-making process.

5. Simulation Loop
The main simulation loop consists of iterations where subjects:
Observe the environment (perception data).
Train using both current and historical data.
Act by choosing a movement decision from the neural network’s output.

6. Database & Historical Learning
A MySQL database stores perception data and past decisions.
Subjects learn from both their own mistakes and those of others in their perception range.
Every iteration, the database is updated with new experiences, which are used to train the neural network.



Next steps...

1) Incorporate genetics. Make it so that feature interaction is unique for each subject. Something that works for a subject will not work for another. And generate features organically so that every simulation is different.
