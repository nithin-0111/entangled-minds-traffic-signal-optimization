# Intelligent Traffic Signal Optimization

## Overview

This project aims to improve traffic management by detecting vehicles, optimizing traffic signal timings using a Deep Q-Learning (DQN) model, and evaluating the system's performance. The solution includes three key scripts: `main.py` for vehicle detection and signal control, `accuracy.py` for evaluating performance, and `training.py` for training the DQN model.

## File Descriptions

- **main.py**: Contains the vehicle detection logic using OpenCV. Processes video streams, counts vehicles, and adjusts signal timings using actions from a trained DQN agent.
- **accuracy.py**: Visualizes the performance of the trained model. Shows optimal time allocation and prediction accuracy for each processed video.
- **training.py**: Implements the DQN algorithm with a neural network. Trains on vehicle count data and optimizes signal timings by adjusting thresholds. The model is periodically saved during training.

## Detailed Script Descriptions

### main.py

This script captures video frames, applies background subtraction for vehicle detection, and draws bounding boxes around detected vehicles. It counts vehicles crossing a set line and uses a DQN agent to optimize signal timings based on vehicle counts.

Key functionalities:
- Detects vehicles in video frames using OpenCV.
- Counts vehicles crossing a specified line.
- Uses a pre-trained DQN model to adjust signal timings.
- Visualizes vehicle counts and signal timings on video frames.

### accuracy.py

This script evaluates the performance of the traffic signal control system by plotting the optimal time allocation and prediction accuracy.

Key functionalities:
- Plots optimal signal timings over multiple videos.
- Displays prediction accuracy of the DQN model.
- Utilizes Matplotlib for visualization.

### training.py

This script trains a Deep Q-Learning model to optimize traffic signal timings. It processes video files, adjusts signal timings based on vehicle counts, and updates the DQN model.

Key functionalities:
- Implements DQN with a neural network.
- Trains the model on simulated traffic scenarios.
- Periodically saves the trained model.
- Adjusts signal timings based on learned policies.

## Future Improvements

- **Additional Traffic Scenarios**: Add different types of vehicles or lane configurations to enhance model robustness.
- **Real-Time Deployment**: Extend the project to work in a real-time traffic control system with live video feeds.
- **Enhanced Reward System**: Improve the reward function to better simulate real-world traffic flow behavior.

## Acknowledgements

- **OpenCV**: For the computer vision functionality, such as vehicle detection and background subtraction.
- **TensorFlow**: For implementing the Deep Q-Learning algorithm.
- **Matplotlib**: For visualizing the accuracy of predictions and optimal time allocation.

## How to Run

### Vehicle Detection & Signal Control (`main.py`)

Run the script with:

```bash
python main.py
```
### Accuracy Evaluation & Visualization (accuracy.py)

Run the script with:

```bash
python accuracy.py
```

### Model Training (training.py)

Run the script with:

```bash
python training.py
```
By following this structure, the three scripts will allow you to detect vehicles, train the DQN model for optimizing traffic signal timings, and visualize the improvements in traffic management.
