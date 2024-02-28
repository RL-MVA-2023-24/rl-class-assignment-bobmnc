from gymnasium.wrappers import TimeLimit
from env_hiv import HIVPatient
import joblib
import numpy as np

env = TimeLimit(
    env=HIVPatient(domain_randomization=False), max_episode_steps=200
)  # The time wrapper limits the number of steps in an episode at 200.
# Now is the floor is yours to implement the agent and train it.


# You have to implement your own agent.
# Don't modify the methods names and signatures, but you can add methods.
# ENJOY!
class ProjectAgent:
    def __init__(self):
       
        
    def act(self, observation, use_random=False):
        Q2 = np.zeros(self.nb_actions)
        for a2 in range(self.nb_actions):
            A2 = a2
            S2A2 = np.append(observation,A2)
            Q2[a2] = self.rf.predict(S2A2.reshape(1, -1))
        return np.argmax(Q2)

    def save(self, path):
        pass

    def load(self):
        self.rf = joblib.load('rf.joblib')
        self.nb_actions = 4
