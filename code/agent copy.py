import torch

class Agent:
    def __init__(self, n_actions, network_policy, optimizer, gamma=0.99):
        self.gamma=gamma
        self.n_actions=n_actions
        self.action = None
        self.action_spaces = [i for i in range(self.n_actions)]

        # self.actor_critic = ActorCriticNetwork(n_actions=n_actions)
        self.network_policy = network_policy
        self.network_policy.compile(optimizer=optimizer)


    def select_action(self, state):
        state = torch.tensor(state, dtype=torch.float32)
        probs = self.network_policy(state)
        #TODO: Add epsilon-greedy algo 
        action_dist = torch.distributions.Categorical(probs=probs) 
        self.action = action_dist

        state = tf.convert_to_tensor([state])
        _, probs = self.actor_critic(state)

        action_prob = tfp.distributions.Categorical(probs=probs)
        action = action_prob.sample()
        self.action = action 
        return action.numpy()[0]

    def save_models(self):
        print('... saving models ...')
        self.actor_critic.save_weights(self.actor_critic.checkpoint_file)
    
    def load_models(self):
        print('... loading models ...')
        self.actor_critic.load_weights(self.actor_critic.checkpoint_file)
    
    def learn(self, state, reward, next_state, done):
        states = tf.convert_to_tensor([state], dtype=tf.float32)
        rewards = tf.convert_to_tensor([reward], dtype=tf.float32)
        next_states = tf.convert_to_tensor([next_state], dtype=tf.float32)

        with tf.GradientTape(persistent=True) as tape:
            state_value, probs = self.actor_critic(states)
            next_state_value, _ = self.actor_critic(next_states)
            state_value = tf.squeeze(state_value)
            next_state_value = tf.squeeze(next_state_value)

            action_probs = tfp.distributions.Categorical(probs=probs)
            log_prob = action_probs.log_prob(self.action)

            delta = rewards + self.gamma*next_state_value(1-int(done)) - state_value
            actor_loss = -log_prob * delta
            critic_loss = delta**2

            total_loss = actor_loss = critic_loss
        
        gradient = tape.gradient(total_loss, self.actor_critic.trainable_variables)
        self.actor_critic.optimizer.apply_gradients(zip(
            gradient, self.actor_critic.trainable_variables))
        
        

