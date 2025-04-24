import numpy as np
import random
from collections import deque

class ReplayBuffer:
    # Each time add() is called, transitions are accumulated into an N-step buffer and then written to the main buffer once N steps are reached.
    # When an episode ends (done=True), any remaining transitions in the N-step buffer (fewer than N) are flushed into the main buffer.
    # sample(batch_size) returns five NumPy arrays: states, actions, rewards, next_states, dones.
    def __init__(self, max_size=1_000_000, n_step=5, gamma=0.99):
        self.buffer = deque(maxlen=max_size)
        self.n_step = n_step
        self.gamma = gamma
        self.n_step_buffer = deque(maxlen=n_step)

    def add(self, state, action, reward, next_state, done):
        # First, append the latest transition to the N-step buffer
        self.n_step_buffer.append((state, action, reward, next_state, done))

        # If we've accumulated N steps, compute the N-step return and store it
        if len(self.n_step_buffer) >= self.n_step:
            R, next_s, d = 0.0, self.n_step_buffer[-1][3], self.n_step_buffer[-1][4]
            for idx, (_, _, r, _, done_flag) in enumerate(self.n_step_buffer):
                R += (self.gamma ** idx) * r
                if done_flag:
                    # Stop accumulating further rewards if episode ended within these N steps
                    break
            s0, a0 = self.n_step_buffer[0][0], self.n_step_buffer[0][1]
            self.buffer.append((s0, a0, R, next_s, d))

        # If the episode ends, flush any remaining transitions in the N-step buffer
        if done:
            while self.n_step_buffer:
                R, next_s, d = 0.0, self.n_step_buffer[-1][3], self.n_step_buffer[-1][4]
                for idx, (_, _, r, _, done_flag) in enumerate(self.n_step_buffer):
                    R += (self.gamma ** idx) * r
                    if done_flag:
                        break
                s0, a0 = self.n_step_buffer[0][0], self.n_step_buffer[0][1]
                self.buffer.append((s0, a0, R, next_s, d))
                self.n_step_buffer.popleft()

    def sample(self, batch_size):
        #Randomly sample batch_size transitions from the main buffer.
        #Returns five NumPy arrays: states, actions, rewards, next_states, dones.
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            np.array(states),
            np.array(actions),
            np.array(rewards, dtype=np.float32),
            np.array(next_states),
            np.array(dones, dtype=np.float32),
        )

    def __len__(self):
        return len(self.buffer)
