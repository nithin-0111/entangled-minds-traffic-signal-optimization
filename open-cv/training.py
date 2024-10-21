import os
import numpy as np
import random
from collections import deque
import tensorflow as tf
import cv2  # Ensure you have OpenCV for video processing

# DQN parameters
STATE_SIZE = 4  # Example state size
ACTION_SIZE = 2  # Example action size
GAMMA = 0.95     
EPSILON = 1.0    
EPSILON_MIN = 0.1
EPSILON_DECAY = 0.995
LEARNING_RATE = 0.001
MEMORY_SIZE = 2000
BATCH_SIZE = 32

# Define the DQN model
class DQN:
    def __init__(self):
        self.model = self.create_model()

    def create_model(self):
        model = tf.keras.Sequential()
        model.add(tf.keras.layers.Dense(24, input_dim=STATE_SIZE, activation='relu'))
        model.add(tf.keras.layers.Dense(24, activation='relu'))
        model.add(tf.keras.layers.Dense(ACTION_SIZE, activation='linear'))

        if tf.__version__.startswith('2.'):
            model.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE))
        else:
            model.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(lr=LEARNING_RATE))

        return model

    def predict(self, state):
        return self.model.predict(state)

    def train(self, memory):
        if len(memory) < BATCH_SIZE:
            return
        minibatch = random.sample(memory, BATCH_SIZE)
        for state, action, reward, next_state in minibatch:
            target = reward
            if next_state is not None:
                target += GAMMA * np.amax(self.model.predict(next_state)[0])
            target_f = self.model.predict(state)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)

# Define the reward calculation function
def calculate_reward(counters, video_index, action):
    # Example reward logic
    if action == 0:  # Increase threshold
        return -1  # Penalty for taking this action
    elif action == 1:  # Decrease threshold
        return 1  # Reward for this action

    return counters[video_index] * 0.1  # Reward based on vehicle count

# Create a DQN agent
agent = DQN()
memory = deque(maxlen=MEMORY_SIZE)

# Directory for saving training data
training_dir = "videos"  # Change this to your training folder
os.makedirs(training_dir, exist_ok=True)

# List of video files
video_files = ['video_05.mp4', 'video_01.mp4', 'video_02.mp4', 'video_03.mp4']  # Example video files

# Initialize counters for each video
counters = [0] * len(video_files)

# Vehicle detection loop
for video_index, video in enumerate(video_files):
    cap = cv2.VideoCapture(video)
    
    while True:
        ret, frame1 = cap.read()
        if not ret:
            break
        
        # Your vehicle detection code goes here
        # (e.g., processing the frame, detecting vehicles, etc.)
        
        # Prepare state for DQN
        state = np.array([counters[video_index], 0, 0, 0]).reshape(1, STATE_SIZE)

        if random.random() <= EPSILON:
            action = random.randrange(ACTION_SIZE)
        else:
            action_values = agent.predict(state)
            action = np.argmax(action_values[0])

        # Calculate the reward
        reward = calculate_reward(counters, video_index, action)

        # Next state (you may need to adjust this logic)
        next_state = np.array([counters[video_index], 0, 0, 0]).reshape(1, STATE_SIZE)  # Update based on your state logic

        # Save experience to memory
        memory.append((state, action, reward, next_state))

        # Train the DQN
        agent.train(memory)

        # Adjust exploration rate
        if EPSILON > EPSILON_MIN:
            EPSILON *= EPSILON_DECAY

        # Save the model periodically
        if len(memory) % 1000 == 0:
            agent.model.save(os.path.join(training_dir, f"dqn_model_{video_index}.h5"))

    cap.release()

# Final model save after training
agent.model.save(os.path.join(training_dir, "final_dqn_model.h5"))

print("Training complete. Model saved.")
