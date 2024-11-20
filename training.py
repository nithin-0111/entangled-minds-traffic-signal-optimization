import os
import numpy as np
import random
from collections import deque
import tensorflow as tf
import cv2

STATE_SIZE = 4
ACTION_SIZE = 2
GAMMA = 0.95     
EPSILON = 1.0    
EPSILON_MIN = 0.1
EPSILON_DECAY = 0.995
LEARNING_RATE = 0.001
MEMORY_SIZE = 2000
BATCH_SIZE = 32

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

def calculate_reward(counters, video_index, action):
    if action == 0:
        return -1
    elif action == 1:
        return 1
    return counters[video_index] * 0.1

agent = DQN()
memory = deque(maxlen=MEMORY_SIZE)

training_dir = "videos"
os.makedirs(training_dir, exist_ok=True)

video_files = ['video_05.mp4', 'video_01.mp4', 'video_02.mp4', 'video_03.mp4']

counters = [0] * len(video_files)

for video_index, video in enumerate(video_files):
    cap = cv2.VideoCapture(video)
    
    while True:
        ret, frame1 = cap.read()
        if not ret:
            break
        
        state = np.array([counters[video_index], 0, 0, 0]).reshape(1, STATE_SIZE)

        if random.random() <= EPSILON:
            action = random.randrange(ACTION_SIZE)
        else:
            action_values = agent.predict(state)
            action = np.argmax(action_values[0])

        reward = calculate_reward(counters, video_index, action)

        next_state = np.array([counters[video_index], 0, 0, 0]).reshape(1, STATE_SIZE)

        memory.append((state, action, reward, next_state))

        agent.train(memory)

        if EPSILON > EPSILON_MIN:
            EPSILON *= EPSILON_DECAY

        if len(memory) % 1000 == 0:
            agent.model.save(os.path.join(training_dir, f"dqn_model_{video_index}.h5"))

    cap.release()

agent.model.save(os.path.join(training_dir, "final_dqn_model.h5"))

print("Training complete. Model saved.")
