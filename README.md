# Traffic Signal Optimization Using Deep Q-Learning (DQN) for Vehicle Detection

This project leverages Deep Q-Learning (DQN) to optimize traffic signal timings based on real-time vehicle detection from video streams. It consists of three separate modules that work together:

- **main.py**: Processes the video feed and detects vehicles in real-time using background subtraction.
- **training.py**: Trains the Deep Q-Learning model to optimize vehicle detection parameters and traffic signal timings.
- **accuracy.py**: Evaluates the accuracy of the predictions and visualizes the improvements in optimal time allocation.

## Project Structure

### 1. main.py (Vehicle Detection & Traffic Signal Control)
- **Goal**: Detects vehicles from video footage and adjusts traffic signal timings dynamically using Q-Learning.
- **Key Features**:
  - Background subtraction for vehicle detection.
  - Vehicle counting at specific positions to adjust traffic signal timings.
  - Dynamic optimization of thresholds (e.g., vehicle size) based on DQN's actions.
  - Displays vehicle count and optimal signal timings for each lane.
- **Inputs**:
  - Video files (e.g., `video_01.mp4`, `video_02.mp4`).
- **Outputs**:
  - Real-time traffic signal color and optimal time adjustments based on the detected vehicle count.

### 2. training.py (Deep Q-Learning Model Training)
- **Goal**: Trains the Deep Q-Learning model to predict and optimize traffic signal timings.
- **Key Features**:
  - DQN setup with a neural network for training and prediction.
  - Experience replay using memory to train the agent.
  - Model training with backpropagation to adjust parameters for optimal traffic flow.
- **Inputs**:
  - Initial state (e.g., vehicle count) and actions (e.g., threshold adjustments).
- **Outputs**:
  - Trained DQN model stored periodically during training for checkpointing.

### 3. accuracy.py (Accuracy Evaluation & Visualization)
- **Goal**: Evaluates and visualizes the accuracy of the traffic signal optimization.
- **Key Features**:
  - Plots the optimal time allocation for each video.
  - Tracks the accuracy of predictions and improvements in time allocation based on the vehicle count.
  - Visualizes how DQN's actions impact the flow of traffic and overall performance.
- **Inputs**:
  - The optimal times and vehicle counts from previous modules.
- **Outputs**:
  - Plots showing the accuracy of prediction and improvements in time allocation over videos.

## Requirements

Before running the project, ensure you have the following dependencies installed:

- Python 3.x
- OpenCV: `pip install opencv-python`
- TensorFlow: `pip install tensorflow`
- NumPy: `pip install numpy`
- Matplotlib: `pip install matplotlib`

## How to Run

### Vehicle Detection & Signal Control (main.py):
This script processes each video file in sequence, detects vehicles, adjusts signal timings, and visualizes the results.
Run the script with:

```bash
python main.py

Accuracy Evaluation & Visualization (accuracy.py):

After running the detection script, use this script to evaluate and plot the performance improvements in time allocation and prediction accuracy. Run the script with:

python accuracy.py

Model Training (training.py):

Use this script to train the Deep Q-Learning model to optimize the traffic signal timings. The model will be saved periodically. Run the script with:

python training.py

File Descriptions
main.py

Contains the vehicle detection logic using OpenCV, processes video streams, counts vehicles, and adjusts signal timings using actions from the trained DQN agent.
training.py

Implements the DQN algorithm with a neural network to train on vehicle count data and optimize signal timings by adjusting thresholds.
accuracy.py

Visualizes the performance of the trained model, showing optimal time allocation and prediction accuracy for each video processed.
Future Improvements

    Additional Traffic Scenarios: Add different types of vehicles or lane configurations.
    Real-Time Deployment: Extend the project to work in a real-time traffic control system with live video feeds.
    Enhanced Reward System: Improve the reward function to better simulate real-world traffic flow behavior.

Acknowledgements

    OpenCV: For the computer vision functionality, such as vehicle detection and background subtraction.
    TensorFlow: For implementing the Deep Q-Learning algorithm.
    Matplotlib: For visualizing the accuracy of predictions and optimal time allocation.

By following this structure, the three scripts will allow you to detect vehicles, train the DQN model for optimizing traffic signal timings, and visualize the improvements in traffic management.
