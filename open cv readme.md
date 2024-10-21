# Intelligent Traffic Signal Optimization using Deep Q-Learning and Reinforcement Learning

[![Python version](https://img.shields.io/badge/python-3.8-blue.svg)](https://www.python.org/downloads/release/python-370/)
[![OpenCV version](https://img.shields.io/badge/OpenCV-4.5.1-green.svg)](https://opencv.org/)
[![TensorFlow version](https://img.shields.io/badge/TensorFlow-2.6.0-yellow.svg)](https://www.tensorflow.org/)
[![NumPy version](https://img.shields.io/badge/NumPy-1.21.2-blue.svg)](https://numpy.org/)

## Introduction
This project implements an intelligent traffic signal optimization algorithm using Deep Q-Learning (DQN) and reinforcement learning techniques to dynamically adjust signal timings based on real-time vehicle counts from video feeds. The goal is to minimize vehicle wait times and improve overall traffic flow by adjusting the signal timings according to traffic density.

## Problem Statement
Traffic congestion is a growing problem in urban areas, leading to delays, fuel wastage, and higher pollution levels. Traditional traffic systems rely on fixed signal timings, which are not optimized for varying traffic conditions throughout the day. This project aims to address this issue by creating a dynamic, data-driven traffic management system.

## Project Overview
The system detects vehicles in video feeds using OpenCV, calculates the optimal signal timings based on the number of vehicles detected in each lane, and employs a reinforcement learning algorithm to adaptively adjust the signal timings in real time. By using Deep Q-Learning, the system improves its decision-making process to optimize traffic flow at intersections.

## Demo
- Live demo of the project: [Check it out here](https://joyful-cannoli-d6b1af.netlify.app/)
- **Password to access the website**: `My-Drop-Site`

## Technologies Used
- **Python**: Main programming language
- **OpenCV**: For vehicle detection and video processing
- **TensorFlow**: To implement Deep Q-Learning (DQN)
- **NumPy**: For efficient numerical operations
- **Matplotlib**: For visualizing traffic data and training performance
- **HTML/CSS/JavaScript**: Frontend for the demo interface

## Algorithm
1. **Vehicle Detection**: OpenCV processes real-time video feeds to detect and count vehicles in each lane.
2. **Reinforcement Learning (Deep Q-Learning)**: 
   - A DQN agent is trained to learn optimal traffic signal timings based on traffic flow patterns.
   - The agent selects actions (signal changes) that minimize overall vehicle wait times.
3. **Signal Timing Adjustment**: The system dynamically adjusts traffic signal timings based on the real-time vehicle counts.
4. **Simulation and Optimization**: The project includes simulations to test the algorithm in various traffic conditions and optimize signal timings based on traffic density, congestion, and time of day.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/abhishikkarmakar/entangled-minds-traffic-signal-optimization.git
   ```
2. Navigate to the project directory:
   ```bash
   cd open_cv
   ```

## Usage
1. Ensure that you have video files (`video_01.mp4`, `video_02.mp4`, etc.) in the project directory for processing.
2. Run the main script:
   ```bash
   python main.py
   ```
3. The program will process each video, detect vehicles in each lane, and calculate the optimal traffic signal timings. The results, including vehicle counts and adjusted signal timings, will be displayed in real time.

## Contributing
Contributions are welcome! If you'd like to contribute, please fork the repository and submit a pull request. 

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Notes
- Make sure that the appropriate video files are present in the directory before running the script.
- Include sample video files or instructions on how to create them for testing purposes if needed.

```
