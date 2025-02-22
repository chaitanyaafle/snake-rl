# AI Snake Game with Deep Q-Learning

## Project Overview
This project implements an AI agent that learns to play the classic Snake game using Deep Q-Learning (DQN), a powerful reinforcement learning algorithm. The agent learns through experience, gradually improving its performance by maximizing the game score while avoiding collisions.

## Technical Implementation

### Deep Q-Learning Architecture
- **Neural Network Model**: Implements a deep neural network with:
  - Input layer (11 nodes): Represents the game state including danger detection and food location
  - Hidden layer (256 nodes): Processes complex patterns using ReLU activation
  - Output layer (3 nodes): Represents possible actions (straight, right, left)

### Key Reinforcement Learning Concepts
1. **State Space (11 dimensions)**:
   - Danger detection in three directions (straight, right, left)
   - Current movement direction (4 states)
   - Food location relative to snake's head (4 states)

2. **Action Space**:
   - Three possible actions: [1,0,0] (straight), [0,1,0] (right turn), [0,0,1] (left turn)

3. **Reward System**:
   - +10 points for eating food
   - -10 points for collision/game over
   - Encourages survival and food collection

4. **Learning Mechanisms**:
   - **Experience Replay**: Stores game experiences in memory for batch learning
   - **Epsilon Greedy Strategy**: Balances exploration and exploitation
   - **Q-Value Updates**: Uses Bellman equation with discount factor (gamma)

## Technical Components

### Files Structure
- `agent.py`: Implements the AI agent with DQN algorithm
- `model.py`: Contains the neural network architecture
- `snake_game.py`: Game environment implementation
- `helper.py`: Utilities for plotting and visualization

### Key Features
1. **Memory Management**:
   - Implements experience replay with a maximum memory of 100,000 states
   - Batch training with 1000 samples for efficient learning

2. **Training Optimization**:
   - Adam optimizer with learning rate of 0.001
   - MSE loss function for Q-value prediction
   - Automatic model saving when achieving new high scores

3. **State Processing**:
   - Real-time state evaluation
   - Efficient collision detection
   - Dynamic food placement

## Results and Performance
- The AI demonstrates significant learning progress over training episodes
- Successfully learns to:
  - Avoid collisions with walls and itself
  - Efficiently navigate towards food
  - Develop long-term survival strategies

## Technologies Used
- Python 3.x
- PyTorch (Neural Network and Training)
- Pygame (Game Environment)
- NumPy (Numerical Operations)
- Matplotlib (Performance Visualization)

## Skills Demonstrated
- Deep Reinforcement Learning Implementation
- Neural Network Architecture Design
- Python Object-Oriented Programming
- Game Development
- Algorithm Optimization
- Data Structure Management

## Future Improvements
- Implementation of Double DQN for more stable learning
- Prioritized experience replay
- Convolutional neural network for visual input processing
- Hyperparameter optimization

## Running the Project
1. Install requirements:
   ```bash
   pip install torch pygame numpy matplotlib
   ```
2. Run the training:
   ```bash
   python agent.py
   ```

## Learning Outcomes
This project demonstrates proficiency in:
- Implementing complex reinforcement learning algorithms
- Building and training neural networks
- Creating efficient game environments
- Managing state spaces and action policies
- Optimizing learning parameters
- Handling real-time decision-making systems

The successful implementation of this project shows strong understanding of both theoretical concepts and practical applications in reinforcement learning, neural networks, and game development.
