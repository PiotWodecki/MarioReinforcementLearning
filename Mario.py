class Mario:
    def __init__(self, state_dim, action_dim):
        pass

    def act(self, state):
        """Given a state, choose an epsilon-greedy action
        """
        pass

    def remember(self, experience):
        """Add the observation to memory
        """
        pass

    def learn(self):
        """Update online action value (Q) function with a batch of experiences
        """
        pass