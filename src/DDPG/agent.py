#!usr/bin/env/python3

import tensorflow as tf
from keras.losses import MSE
from keras.optimizers import Adam

# Import Our Classes
from DDPG.buffer import ReplayBuffer
from DDPG.networks import ActorNetwork, CriticNetwork


class Agent:

    def __init__(self, input_dims, alpha=0.001, beta=0.002, env=None, gamma=0.99, n_actions=2,
                 max_size=1000000, tau=0.005, fc1=512, fc2=512, batch_size=64, noise=0.1, chkpt_dir='tmp/ddpg'):

        # Learning Rates for the ActorNetwork [alpha] and the CriticNetwork [beta]
        self.alpha = alpha
        self.beta = beta

        # Discount Factor for our Update Equation [gamma] and Value for our SoftUpdate [tau]
        self.gamma = gamma
        self.tau = tau

        # Create the ReplayBuffer
        self.memory = ReplayBuffer(max_size, input_dims, n_actions)
        
        # Min and Max Action
        self.max_action = env.action_space.high[0]
        self.min_action = env.action_space.low[0]

        # Other Variables
        self.n_actions = n_actions
        self.batch_size = batch_size
        self.noise = noise

        # Instantiate Networks
        self.actor = ActorNetwork(n_actions=n_actions, name='actor', fc1_dims=fc1, fc2_dims=fc2, chkpt_dir=chkpt_dir)
        self.critic = CriticNetwork(name='critic', fc1_dims=fc1, fc2_dims=fc2, chkpt_dir=chkpt_dir)
        self.target_actor = ActorNetwork(n_actions=n_actions, name='target_actor', fc1_dims=fc1, fc2_dims=fc2, chkpt_dir=chkpt_dir)
        self.target_critic = CriticNetwork(name='target_critic', fc1_dims=fc1, fc2_dims=fc2, chkpt_dir=chkpt_dir)

        # Compile Networks using the Adam Optimizer
        self.actor.compile(optimizer=Adam(learning_rate=alpha))
        self.critic.compile(optimizer=Adam(learning_rate=beta))
        self.target_actor.compile(optimizer=Adam(learning_rate=alpha))
        self.target_critic.compile(optimizer=Adam(learning_rate=beta))

        # Update Network Parameters (Hard Copy of Initial Weight of our Actor and Critic Networks to the Target Ones)
        self.update_network_parameters(tau=1)

    def update_network_parameters(self, tau=None):

        # Use Default if is None
        if tau is None: tau = self.tau

        actor_weights = []
        critic_weights = []
        actor_targets = self.target_actor.weights
        critic_targets = self.target_critic.weights
        
        # Iterate over the Actor Weights
        for i, actor_weight in enumerate(self.actor.weights):
            actor_weights.append(actor_weight*tau + actor_targets[i]*(1-tau))

        # Iterate over the Critic Weights
        for i, critic_weight in enumerate(self.critic.weights):
            critic_weights.append(critic_weight*tau +
                                  critic_targets[i]*(1-tau))

        # Update Actor and Critic Weights
        self.target_actor.set_weights(actor_weights)
        self.target_critic.set_weights(critic_weights)

    def remember(self, state, action, reward, new_state, done):

        # Save Transition Function
        self.memory.store_transition(state, action, reward, new_state, done)

    def save_models(self):

        print('\n.... saving models ....\n')
        self.actor.save_weights(self.actor.checkpoint_file)
        self.critic.save_weights(self.critic.checkpoint_file)
        self.target_actor.save_weights(self.target_actor.checkpoint_file)
        self.target_critic.save_weights(self.target_critic.checkpoint_file)

    def load_models(self):

        print('\n.... loading models ....\n')
        self.actor.load_weights(self.actor.checkpoint_file)
        self.critic.load_weights(self.critic.checkpoint_file)
        self.target_actor.load_weights(self.target_actor.checkpoint_file)
        self.target_critic.load_weights(self.target_critic.checkpoint_file)

    def choose_action(self, observation, evaluate=False):

        # Convert our State to a Tensor and add an Extra Dimension to our Observation to give it a Batch Dimension
        state = tf.convert_to_tensor([observation], dtype=tf.float32)
        actions = self.actor(state)
        
        # If Training add some Random Normal Noise
        if not evaluate: actions += tf.random.normal(shape=[self.n_actions], mean=0.0, stddev=self.noise)
        
        # Clip the Action inside the Enviroment Bounds
        actions = tf.clip_by_value(actions, self.min_action, self.max_action)
        
        # Return the Value of the Action (Zero-th Element of the Tensor)
        return actions[0]
    
    def learn(self):
        
        # Wait to Fill the Memory to at Least the Batch Size
        if self.memory.mem_cntr < self.batch_size: return
        
        # Sample Memory
        state, action, reward, new_state, done = self.memory.sample_buffer(self.batch_size)
        
        # Convert to Tensors
        states = tf.convert_to_tensor(state, dtype=tf.float32)
        new_states = tf.convert_to_tensor(new_state, dtype=tf.float32)
        actions = tf.convert_to_tensor(action, dtype=tf.float32)
        
        # Load Operation onto our Computational Graph for Calculation of Gradients -> Compute Critic Loss
        with tf.GradientTape() as tape:
            
            # Get the target Action from the Target Actor
            target_actions = self.target_actor(new_states)
            
            # Compute the Critic Value for the New States by the Target Critic Evaluation of those States and Actions (Squeezed along 1st Dimension)
            new_critic_value = tf.squeeze(self.target_critic(new_states, target_actions), 1)
            
            # Compute Critic Value
            critic_value = tf.squeeze(self.critic(states, actions), 1)
            
            # Compute the Target (if episode is over, done = 1 -> * 0)
            target = reward + self.gamma * new_critic_value * (1 - done)
            
            # Compute Loss and Gradient
            critic_loss = MSE(target, critic_value)
            critic_network_gradient = tape.gradient(critic_loss, self.critic.trainable_variables)
            
            # Apply Gradients
            self.critic.optimizer.apply_gradients(zip(critic_network_gradient, self.critic.trainable_variables))
        
        # Compute Actor Loss
        with tf.GradientTape() as tape:
            
            # Get Actions based on its Current set of Weights
            new_policy_actions = self.actor(states)
            
            # Actor Loss (Gradient Ascent)
            actor_loss = -self.critic(states, new_policy_actions)
            actor_loss = tf.math.reduce_mean(actor_loss)
            
            # Compute Actor Gradient
            actor_network_gradient = tape.gradient(actor_loss, self.actor.trainable_variables)
        
            # Apply Gradients
            self.actor.optimizer.apply_gradients(zip(actor_network_gradient, self.actor.trainable_variables))
            
        
        # Update Parameters
        self.update_network_parameters()