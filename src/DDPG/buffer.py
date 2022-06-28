#!usr/bin/env/python3

import numpy as np


class ReplayBuffer:

    def __init__(self, max_size, input_shape, n_actions):

        # If we exceed the memory size we overwrite our earliest memory with the new ones
        self.mem_size = max_size

        # Memory counter that start from 0
        self.mem_cntr = 0

        # Create State, New State, Action, Reward and Terminal Memories
        self.state_memory = np.zeros((self.mem_size, *input_shape))
        self.new_state_memory = np.zeros((self.mem_size, *input_shape))
        self.action_memory = np.zeros((self.mem_size, n_actions))
        self.reward_memory = np.zeros(self.mem_size)
        self.terminal_memory = np.zeros(self.mem_size, dtype=np.bool)

    def store_transition(self, state, action, reward, new_state, done):

        # Get the position of the first available memory
        index = self.mem_cntr % self.mem_size

        # Save Memories
        self.state_memory[index] = state
        self.new_state_memory[index] = new_state
        self.action_memory[index] = action
        self.reward_memory[index] = reward
        self.terminal_memory[index] = done

        # Increment Counter
        self.mem_cntr += 1

    def sample_buffer(self, batch_size):

        # How much memory is filled up
        max_mem = min(self.mem_cntr, self.mem_size)

        # Batch of Number Random Choice, replace=False -> prevent from double sampling two identical memory
        batch = np.random.choice(max_mem, batch_size, replace=False)

        # Dereference the Batch from the Memories Array
        states = self.state_memory[batch]
        new_states = self.new_state_memory[batch]
        actions = self.action_memory[batch]
        rewards = self.reward_memory[batch]
        dones = self.terminal_memory[batch]

        return states, actions, rewards, new_states, dones
