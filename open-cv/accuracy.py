import cv2 
import numpy as np
import random
from collections import deque
import tensorflow as tf
import matplotlib.pyplot as plt

# DQN parameters
STATE_SIZE = 4  # Example state size
ACTION_SIZE = 2  # Example action size
GAMMA = 0.95     
EPSILON = 1.0    
EPSILON_MIN = 0.1
EPSILON_DECAY = 0.995
LEARNING_RATE = 0.001

# Define the DQN model
class DQN:
    def __init__(self):
        self.model = self.create_model()

    def create_model(self):
        model = tf.keras.Sequential()
        model.add(tf.keras.layers.Dense(24, input_dim=STATE_SIZE, activation='relu'))
        model.add(tf.keras.layers.Dense(24, activation='relu'))
        model.add(tf.keras.layers.Dense(ACTION_SIZE, activation='linear'))
        model.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(lr=LEARNING_RATE))
        return model

    def predict(self, state):
        return self.model.predict(state)

    def train(self, state, target):
        self.model.fit(state, target, epochs=1, verbose=0)

# Create a DQN agent
agent = DQN()
memory = deque(maxlen=2000)

# List of video files
video_files = ['video_01.mp4', 'video_02.mp4', 'video_03.mp4', 'video_04.mp4']
count_line_position = 550
min_width_rect = 80  
min_height_rect = 80  
algo = cv2.createBackgroundSubtractorMOG2()

# Initialize counters for each video
counters = [0, 0, 0, 0]
optimal_times = []
predictions_accuracy = []

def calculate_optimal_time(vehicle_count):
    L = (2 * 2) + 0  # Example: 2 seconds lost time per phase, 0 all-red time
    y = vehicle_count / 100  # Example: convert vehicle count to flow ratio
    if y >= 1:
        y = 0.99  # Prevent division by zero
    Co = (1.5 * L + 5) / (1 - y)
    return Co

# Vehicle detection loop
for video_index, video in enumerate(video_files):
    cap = cv2.VideoCapture(video)
    detect = []
    
    while True:
        ret, frame1 = cap.read()
        if not ret:
            break
        
        grey = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(grey, (3, 3), 5)
        img_sub = algo.apply(blur)
        dilat = cv2.dilate(img_sub, np.ones((5, 5)))
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        dilatada = cv2.morphologyEx(dilat, cv2.MORPH_CLOSE, kernel)
        dilatada = cv2.morphologyEx(dilatada, cv2.MORPH_CLOSE, kernel)
        counterShape, h = cv2.findContours(dilatada, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        cv2.line(frame1, (25, count_line_position), (1200, count_line_position), (255, 127, 0), 3)
        
        for (i, c) in enumerate(counterShape):
            (x, y, w, h) = cv2.boundingRect(c)
            validate_counter = (w >= min_width_rect) and (h >= min_height_rect)
            if not validate_counter:
                continue
            
            cv2.rectangle(frame1, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
            center = (x + w // 2, y + h // 2)
            detect.append(center)
            cv2.circle(frame1, center, 4, (0, 0, 255), -1)
            
            for (x, y) in detect:
                if y < (count_line_position + 6) and y > (count_line_position - 6):
                    counters[video_index] += 1
                detect.remove((x, y))
        
        # Prepare state for DQN
        state = np.array([counters[video_index], 0, 0, 0]).reshape(1, STATE_SIZE)
        if random.random() <= EPSILON:
            action = random.randrange(ACTION_SIZE)
        else:
            action_values = agent.predict(state)
            action = np.argmax(action_values[0])

        # Adjust detection parameters based on action
        if action == 0:  # Example action: Increase threshold
            min_width_rect += 5
        else:  # Decrease threshold
            min_width_rect = max(80, min_width_rect - 5)

        # Calculate optimal times and determine signal colors
        optimal_time = calculate_optimal_time(counters[video_index])
        optimal_times.append(optimal_time)

        # Track accuracy
        predictions_accuracy.append(optimal_time)

        max_count_index = np.argmax(counters)
        signal_color = ["green" if i == max_count_index else "red" for i in range(len(counters))]

        # Display signal colors and timings
        for i, (color, time) in enumerate(zip(signal_color, optimal_times)):
            cv2.putText(frame1, f"Lane {i+1}: {color} | Time: {time:.2f}s", 
                        (50, 100 + i * 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0) if color == "green" else (0, 0, 255), 2)

        # Print vehicle counts
        cv2.putText(frame1, f"Vehicle Count: {counters[video_index]}", 
                    (450, 70), cv2.FONT_HERSHEY_SIMPLEX, 2, (128, 128, 0), 5)
 
        cv2.imshow('Video original', frame1)
        
        if cv2.waitKey(1) == 13:
            break
    
    cap.release()

cv2.destroyAllWindows()

# Print final counts and optimal times
for i, (count, time) in enumerate(zip(counters, optimal_times), start=1):
    print(f"Final Vehicle Count for Video {i}: {count}, Optimal Time: {time:.2f}s")

# Plotting the accuracy of prediction and improvements in time allocation
plt.figure(figsize=(12, 6))

# Plot optimal times
plt.subplot(1, 2, 1)
plt.plot(optimal_times, label='Optimal Time', marker='o')
plt.title('Optimal Time Allocation Over Videos')
plt.xlabel('Video Index')
plt.ylabel('Time (seconds)')
plt.legend()

# Plot accuracy
plt.subplot(1, 2, 2)
plt.plot(predictions_accuracy, label='Prediction Accuracy', marker='o', color='orange')
plt.title('Prediction Accuracy Over Videos')
plt.xlabel('Video Index')
plt.ylabel('Accuracy')
plt.legend()

plt.tight_layout()
plt.show()
