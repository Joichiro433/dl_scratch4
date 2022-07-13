from typing import List, Dict, Tuple
from collections import defaultdict

import numpy as np
from dl_scratch4.common.gridworld import GridWorld, Coord
# from common.utils import greedy_probs


def greedy_probs(Q: Dict[Tuple[Coord, int], float], state: Coord, epsilon: float = 0, action_size: int = 4) -> Dict[int, float]:
    qs: List[float] = [Q[(state, action)] for action in range(action_size)]
    max_action: float = np.argmax(qs)

    base_prob: float = epsilon / action_size
    action_probs: Dict[int, float] = {action: base_prob for action in range(action_size)}  #{0: ε/4, 1: ε/4, 2: ε/4, 3: ε/4}
    action_probs[max_action] += (1 - epsilon)
    return action_probs


class McAgent:
    def __init__(self):
        self.gamma: float = 0.9
        self.epsilon: float = 0.1
        self.alpha: float = 0.1
        self.action_size: int = 4

        random_actions: Dict[int, float] = {0: 0.25, 1: 0.25, 2: 0.25, 3: 0.25}
        self.pi: Dict[Coord, Dict[int, float]] = defaultdict(lambda: random_actions)
        self.Q: Dict[Tuple[Coord, int], float] = defaultdict(lambda: 0)
        self.memory: List[Tuple[Coord, int, float]] = []

    def get_action(self, state: Coord) -> int:
        action_probs: Dict[int, float] = self.pi[state]
        actions: List[int] = list(action_probs.keys())
        probs: List[float] = list(action_probs.values())
        return np.random.choice(actions, p=probs)

    def add(self, state: Coord, action: int, reward: float) -> None:
        data: Tuple[Coord, int, float] = (state, action, reward)
        self.memory.append(data)

    def reset(self) -> None:
        self.memory.clear()

    def update(self) -> None:
        G: float = 0
        for data in reversed(self.memory):
            state, action, reward = data
            G = self.gamma * G + reward
            key: Tuple[Coord, int] = (state, action)
            self.Q[key] += (G - self.Q[key]) * self.alpha
            self.pi[state] = greedy_probs(self.Q, state, self.epsilon)


env = GridWorld()
agent = McAgent()

episodes = 10000
for episode in range(episodes):
    state = env.reset()
    agent.reset()

    while True:
        action = agent.get_action(state)
        next_state, reward, done = env.step(action)

        agent.add(state, action, reward)
        if done:
            agent.update()
            break

        state = next_state

env.render_q(agent.Q)
