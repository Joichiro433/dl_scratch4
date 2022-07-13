from typing import List, Dict, Tuple, Deque
from collections import defaultdict, deque

import numpy as np
from dl_scratch4.common.gridworld import GridWorld, Coord
from dl_scratch4.common.utils import greedy_probs


class SarsaAgent:
    def __init__(self) -> None:
        self.gamma: float = 0.9
        self.alpha: float = 0.8
        self.epsilon: float = 0.1
        self.action_size: int = 4

        random_actions: Dict[int, float] = {0: 0.25, 1: 0.25, 2: 0.25, 3: 0.25}
        self.pi: Dict[Coord, Dict[int, float]] = defaultdict(lambda: random_actions)
        self.Q: Dict[Tuple[Coord, int], float] = defaultdict(lambda: 0)
        self.memory: Deque[Tuple[Coord, int, float, bool]] = deque(maxlen=2)

    def get_action(self, state) -> int:
        action_probs: Dict[int, float] = self.pi[state]
        actions: List[int] = list(action_probs.keys())
        probs: List[float] = list(action_probs.values())
        return np.random.choice(actions, p=probs)

    def reset(self) -> None:
        self.memory.clear()

    def update(self, state: Coord, action: int, reward: float, done: bool) -> None:
        self.memory.append((state, action, reward, done))
        if len(self.memory) < 2:
            return

        state, action, reward, done = self.memory[0]
        next_state, next_action, _, _ = self.memory[1]
        next_q: float = 0 if done else self.Q[next_state, next_action]

        target: float = reward + self.gamma * next_q
        self.Q[state, action] += (target - self.Q[state, action]) * self.alpha
        self.pi[state] = greedy_probs(self.Q, state, self.epsilon)


env = GridWorld()
agent = SarsaAgent()

episodes = 10000
for episode in range(episodes):
    state = env.reset()
    agent.reset()

    while True:
        action = agent.get_action(state)
        next_state, reward, done = env.step(action)

        agent.update(state, action, reward, done)

        if done:
            agent.update(next_state, None, None, None)
            break
        state = next_state

env.render_q(agent.Q)