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
        self.nb_actions = 4
       
        
    def act(self, observation, use_random=False):
        Qsa = []
        for a in range(self.nb_actions):
            sa = np.append(observation,a).reshape(1, -1)
            Qsa.append(self.rf.predict(sa))
        return np.argmax(Qsa)

    def save(self, path):
        pass

    def load(self):
        self.rf = joblib.load('src/rf1000_iter.joblib')
