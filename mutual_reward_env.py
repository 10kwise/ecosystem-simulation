"""
Mutual Reward Environment
--------------------------
Two agents A and B each have a shifting internal target (1-5).
Each timestep both pick a number (1-5).

Reward structure:
  - A is rewarded based on how close B's pick is to A's target
  - B is rewarded based on how close A's pick is to B's target

This forces each agent to:
  1. Pick the other agent's target (to earn its own reward)
  2. Signal its own target through its behavior (so the other agent can reward it)

Observations: last K actions of both agents (no direct target info)
Target shift: every SHIFT_INTERVAL steps, each agent's target is resampled

Measurements logged:
  - Cumulative reward per agent
  - Mutual information between action sequences
  - Actual action values over time (to detect manipulation vs cooperation)
"""

import numpy as np
from gymnasium import spaces
from pettingzoo import ParallelEnv
from collections import deque
from sklearn.metrics import mutual_info_score
import matplotlib.pyplot as plt
import pickle   # used for saving and loading agent brains


# ─────────────────────────────────────────────
#  Environment
# ─────────────────────────────────────────────

class MutualRewardEnv(ParallelEnv):
    metadata = {"name": "mutual_reward_v0"}

    def __init__(
        self,
        history_len: int = 4,        # how many past timesteps each agent observes
                                      # KEEP THIS SMALL (3-4) — see explanation below
        shift_interval: int = 400,   # how often targets shift (was 75 — too fast)
        n_actions: int = 5,
        graded_reward: bool = True,
        max_steps: int = 5000,
    ):
        super().__init__()

        self.history_len     = history_len
        self.shift_interval  = shift_interval
        self.n_actions       = n_actions
        self.graded_reward   = graded_reward
        self.max_steps       = max_steps

        self.possible_agents = ["A", "B"]
        self.agents          = self.possible_agents[:]

        # observation: [own last K actions, partner last K actions, own current target]
        obs_size = history_len * 2 + 1
        self.observation_spaces = {
            agent: spaces.Box(low=0, high=n_actions, shape=(obs_size,), dtype=np.float32)
            for agent in self.possible_agents
        }
        self.action_spaces = {
            agent: spaces.Discrete(n_actions)
            for agent in self.possible_agents
        }

    def _sample_target(self):
        return np.random.randint(1, self.n_actions + 1)

    def _compute_reward(self, pick, target):
        if self.graded_reward:
            dist = abs(pick - target) / (self.n_actions - 1)
            return 1.0 - dist
        else:
            return 1.0 if pick == target else 0.0

    def _get_obs(self, agent):
        partner      = "B" if agent == "A" else "A"
        own_hist     = list(self.history[agent])
        partner_hist = list(self.history[partner])
        target       = [self.targets[agent]]
        return np.array(own_hist + partner_hist + target, dtype=np.float32)

    def reset(self, seed=None, options=None):
        if seed is not None:
            np.random.seed(seed)

        self.agents    = self.possible_agents[:]
        self.timestep  = 0
        self.targets   = {agent: self._sample_target() for agent in self.agents}
        self.history   = {
            agent: deque([0] * self.history_len, maxlen=self.history_len)
            for agent in self.agents
        }

        # logging
        self.action_log  = {"A": [], "B": []}
        self.reward_log  = {"A": [], "B": []}
        self.target_log  = {"A": [], "B": []}
        self.shift_steps = []

        observations = {agent: self._get_obs(agent) for agent in self.agents}
        infos        = {agent: {} for agent in self.agents}
        return observations, infos

    def step(self, actions):
        self.timestep += 1

        a_pick = actions["A"] + 1
        b_pick = actions["B"] + 1

        r_a = self._compute_reward(b_pick, self.targets["A"])
        r_b = self._compute_reward(a_pick, self.targets["B"])

        rewards = {"A": r_a, "B": r_b}

        self.action_log["A"].append(a_pick)
        self.action_log["B"].append(b_pick)
        self.reward_log["A"].append(r_a)
        self.reward_log["B"].append(r_b)
        self.target_log["A"].append(self.targets["A"])
        self.target_log["B"].append(self.targets["B"])

        self.history["A"].append(a_pick)
        self.history["B"].append(b_pick)

        if self.timestep % self.shift_interval == 0:
            self.targets["A"] = self._sample_target()
            self.targets["B"] = self._sample_target()
            self.shift_steps.append(self.timestep)

        terminated = {agent: False            for agent in self.agents}
        truncated  = {agent: self.timestep >= self.max_steps for agent in self.agents}

        if all(truncated.values()):
            self.agents = []

        observations = {agent: self._get_obs(agent) for agent in self.possible_agents}
        infos        = {agent: {} for agent in self.possible_agents}
        return observations, rewards, terminated, truncated, infos


# ─────────────────────────────────────────────
#  Mutual Information Helper
# ─────────────────────────────────────────────

def rolling_mutual_information(seq_a, seq_b, window=200):
    """
    MI measures statistical dependency between the two action sequences.
    If MI is near zero: agents are acting independently — no coordination.
    If MI is rising: agents' choices are becoming predictable from each other
                     — a signaling convention is forming.
    """
    mi_values = []
    for i in range(window, len(seq_a) + 1):
        a_window = seq_a[i - window: i]
        b_window = seq_b[i - window: i]
        mi = mutual_info_score(a_window, b_window)
        mi_values.append(mi)
    return mi_values


# ─────────────────────────────────────────────
#  Q-Learning Agent
# ─────────────────────────────────────────────
#
# Think of the Q-table as a notebook the agent builds while playing.
# Each row = a situation the agent has been in (its current observation).
# Each column = an action it could take (pick 1, 2, 3, 4, or 5).
# Each cell = the agent's running estimate of how much future reward
#             it will get if it takes that action in that situation.
#
# When acting: look up current situation, pick the column with highest value.
# When learning: after getting reward, update the relevant cell.
#
# WHY SMALL HISTORY_LEN:
# The notebook has one row per UNIQUE SITUATION the agent encounters.
# If history_len=8, a situation is described by 17 numbers (8 own + 8 partner + 1 target),
# each from 0-5. That gives 6^17 = ~16 trillion possible rows.
# The agent will only see a tiny fraction of those in 5000 steps.
# Most rows stay blank, so most of the time the agent is acting on no information.
# With history_len=4, the notebook has only 6^9 = ~10 million rows — much more manageable.

class QLearningAgent:
    def __init__(self, n_actions, epsilon=0.3, alpha=0.1, gamma=0.95):
        self.n_actions = n_actions
        self.epsilon   = epsilon   # probability of picking randomly (exploration)
        self.alpha     = alpha     # how fast to update the notebook
        self.gamma     = gamma     # how much to value future vs immediate reward
        self.q_table   = {}        # the notebook

    def _state_key(self, obs):
        return tuple(obs.astype(int))

    def get_q(self, obs):
        key = self._state_key(obs)
        if key not in self.q_table:
            self.q_table[key] = np.zeros(self.n_actions)
        return self.q_table[key]

    def act(self, obs, frozen=False):
        """
        frozen=True: agent uses notebook but never explores randomly.
        Use this when testing a saved brain without allowing further learning.
        """
        if not frozen and np.random.rand() < self.epsilon:
            return np.random.randint(self.n_actions)
        return int(np.argmax(self.get_q(obs)))

    def update(self, obs, action, reward, next_obs):
        q      = self.get_q(obs)
        q_next = self.get_q(next_obs)
        q[action] += self.alpha * (reward + self.gamma * np.max(q_next) - q[action])

    def decay_epsilon(self, rate=0.9995, min_eps=0.05):
        self.epsilon = max(self.epsilon * rate, min_eps)

    # ── Save and load brain ───────────────────────────────────────────

    def save(self, path):
        """
        Save everything the agent has learned to a file.
        This is the entire brain — the notebook plus current settings.
        """
        data = {
            "q_table":   self.q_table,
            "epsilon":   self.epsilon,
            "alpha":     self.alpha,
            "gamma":     self.gamma,
            "n_actions": self.n_actions,
        }
        with open(path, "wb") as f:
            pickle.dump(data, f)
        print(f"  Brain saved -> {path}")

    @classmethod
    def load(cls, path):
        """
        Load a saved brain from file. Returns a fully ready agent.
        The loaded agent picks up exactly where the saved one left off.
        """
        with open(path, "rb") as f:
            data = pickle.load(f)
        agent           = cls(n_actions=data["n_actions"])
        agent.q_table   = data["q_table"]
        agent.epsilon   = data["epsilon"]
        agent.alpha     = data["alpha"]
        agent.gamma     = data["gamma"]
        print(f"  Brain loaded <- {path}")
        return agent


# ─────────────────────────────────────────────
#  Training Loop
# ─────────────────────────────────────────────

def train(episodes=10, max_steps=5000, history_len=4, shift_interval=400,
          save_brains=True, fixed_seed=42):
    """
    fixed_seed: use the same random seed every episode so the environment
                is consistent and the Q-table can build useful knowledge
                across episodes rather than being confused by different
                random sequences.
    save_brains: save each agent's brain after the final episode.
    """
    env = MutualRewardEnv(
        history_len=history_len,
        shift_interval=shift_interval,
        n_actions=5,
        graded_reward=True,
        max_steps=max_steps,
    )

    agents = {
        "A": QLearningAgent(n_actions=5),
        "B": QLearningAgent(n_actions=5),
    }

    all_rewards = {"A": [], "B": []}
    all_mi      = []
    all_shifts  = []

    for ep in range(episodes):
        # fixed_seed keeps environment consistent across episodes
        obs, _   = env.reset(seed=fixed_seed)
        prev_obs = {agent: obs[agent] for agent in env.possible_agents}
        ep_rewards = {"A": 0.0, "B": 0.0}

        while env.agents:
            actions = {
                agent: agents[agent].act(obs[agent])
                for agent in env.agents
            }

            next_obs, rewards, terminated, truncated, _ = env.step(actions)

            for agent in env.possible_agents:
                if agent in rewards:
                    agents[agent].update(
                        prev_obs[agent],
                        actions[agent],
                        rewards[agent],
                        next_obs[agent],
                    )
                    ep_rewards[agent] += rewards[agent]

            prev_obs = {agent: next_obs[agent] for agent in env.possible_agents}
            obs      = next_obs

            for agent in env.possible_agents:
                agents[agent].decay_epsilon()

        for agent in env.possible_agents:
            all_rewards[agent].append(ep_rewards[agent])

        mi = rolling_mutual_information(env.action_log["A"], env.action_log["B"], window=200)
        all_mi.append(mi)
        all_shifts.append(env.shift_steps)

        from collections import Counter
        freq_A = Counter(env.action_log["A"])
        freq_B = Counter(env.action_log["B"])
        total_steps = len(env.action_log["A"])
        freq_str_A = " ".join(f"{n}:{freq_A.get(n,0)/total_steps*100:4.1f}%" for n in range(1,6))
        freq_str_B = " ".join(f"{n}:{freq_B.get(n,0)/total_steps*100:4.1f}%" for n in range(1,6))

        print(
            f"Episode {ep+1:>2}/{episodes} | "
            f"Reward A: {ep_rewards['A']:>7.1f} | "
            f"Reward B: {ep_rewards['B']:>7.1f} | "
            f"Final MI: {mi[-1]:.4f} | "
            f"Shifts: {len(env.shift_steps)}"
        )
        print(f"  A picks -> {freq_str_A}")
        print(f"  B picks -> {freq_str_B}")

    if save_brains:
        agents["A"].save("outputs/brain_A.pkl")
        agents["B"].save("outputs/brain_B.pkl")

    plot_results(env, all_rewards, all_mi, all_shifts)
    return env, agents


# ─────────────────────────────────────────────
#  Frozen Test
# ─────────────────────────────────────────────

def frozen_test(brain_path_A, brain_path_B, max_steps=2000, fixed_seed=99):
    """
    Load saved brains, freeze both agents (no learning, no random exploration),
    and observe their pure coordination behavior.

    This isolates what the agents have actually learned.
    A frozen agent acts entirely on its notebook — no guessing.

    You can also run asymmetric tests:
      - Freeze A, give B a fresh random brain -> tests if A is a good signaler
        (a fresh B should learn to read A quickly if A signals clearly)
      - Freeze B, give A a fresh random brain -> same test in reverse
    """
    print("\n── Frozen Test ──────────────────────────────")
    env = MutualRewardEnv(max_steps=max_steps, history_len=4, shift_interval=400)

    agent_A = QLearningAgent.load(brain_path_A)
    agent_B = QLearningAgent.load(brain_path_B)
    loaded  = {"A": agent_A, "B": agent_B}

    obs, _     = env.reset(seed=fixed_seed)
    total_r    = {"A": 0.0, "B": 0.0}

    while env.agents:
        actions = {
            agent: loaded[agent].act(obs[agent], frozen=True)
            for agent in env.agents
        }
        obs, rewards, _, _, _ = env.step(actions)
        for agent in env.possible_agents:
            if agent in rewards:
                total_r[agent] += rewards[agent]

    print(f"Frozen test reward — A: {total_r['A']:.1f} | B: {total_r['B']:.1f}")

    mi = rolling_mutual_information(env.action_log["A"], env.action_log["B"], window=200)
    print(f"Frozen test final MI: {mi[-1]:.4f}")

    plot_frozen(env, mi)
    return env


# ─────────────────────────────────────────────
#  Plotting
# ─────────────────────────────────────────────

def plot_results(env, all_rewards, all_mi, all_shifts):
    fig, axes = plt.subplots(4, 1, figsize=(13, 16))
    fig.suptitle("Mutual Reward Environment — Training Results", fontsize=14)

    # ── Plot 1: Cumulative reward per episode ──
    ax = axes[0]
    ax.plot(all_rewards["A"], label="Agent A", marker="o")
    ax.plot(all_rewards["B"], label="Agent B", marker="s")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Cumulative Reward")
    ax.set_title("Cumulative Reward per Episode  (rising = learning to coordinate)")
    ax.legend(); ax.grid(True, alpha=0.3)

    # ── Plot 2: Rolling MI — last episode ──
    ax = axes[1]
    last_mi = all_mi[-1]
    x_mi = list(range(200, 200 + len(last_mi)))
    ax.plot(x_mi, last_mi, color="purple", label="Rolling MI (window=200)")
    for s in all_shifts[-1]:
        ax.axvline(x=s, color="red", linestyle="--", alpha=0.5,
                   label="Target Shift" if s == all_shifts[-1][0] else "")
    ax.set_xlabel("Timestep")
    ax.set_ylabel("Mutual Information (nats)")
    ax.set_title("MI — Last Episode  (rising = hidden structure in picks)")
    ax.legend(); ax.grid(True, alpha=0.3)

    # ── Plot 3: Smoothed per-step reward — last episode ──
    ax = axes[2]
    window = 80
    r_a = np.convolve(env.reward_log["A"], np.ones(window)/window, mode="valid")
    r_b = np.convolve(env.reward_log["B"], np.ones(window)/window, mode="valid")
    ax.plot(r_a, label="Agent A (smoothed)", alpha=0.8)
    ax.plot(r_b, label="Agent B (smoothed)", alpha=0.8)
    for s in env.shift_steps:
        ax.axvline(x=s, color="red", linestyle="--", alpha=0.5,
                   label="Target Shift" if s == env.shift_steps[0] else "")
    ax.set_xlabel("Timestep")
    ax.set_ylabel("Reward (smoothed)")
    ax.set_title("Per-Step Reward — Last Episode  (dips at shifts show recovery speed)")
    ax.legend(); ax.grid(True, alpha=0.3)

    # ── Plot 4: Actual action values — last episode (KEY new plot) ──
    # This tells you WHAT the agents are actually picking over time.
    # Patterns to look for:
    #   - One agent stuck on one number for long = possible manipulation/collapse
    #   - Actions tracking the target_log values = genuine coordination
    #   - Random-looking but MI is high = hidden convention (the interesting case)
    ax = axes[3]
    steps = list(range(len(env.action_log["A"])))
    ax.plot(steps, env.action_log["A"],  alpha=0.4, label="A picks",    color="blue",   linewidth=0.8)
    ax.plot(steps, env.action_log["B"],  alpha=0.4, label="B picks",    color="orange", linewidth=0.8)
    ax.plot(steps, env.target_log["A"],  alpha=0.8, label="A's target", color="blue",   linewidth=1.5, linestyle="--")
    ax.plot(steps, env.target_log["B"],  alpha=0.8, label="B's target", color="orange", linewidth=1.5, linestyle="--")
    for s in env.shift_steps:
        ax.axvline(x=s, color="red", linestyle="--", alpha=0.4,
                   label="Target Shift" if s == env.shift_steps[0] else "")
    ax.set_xlabel("Timestep")
    ax.set_ylabel("Number (1-5)")
    ax.set_title("Actual Picks vs Targets  (solid=picks, dashed=targets)")
    ax.legend(ncol=3); ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("outputs/mutual_reward_results.png", dpi=150, bbox_inches="tight")
    print("\nPlot saved -> mutual_reward_results.png")
    plt.close()


def plot_frozen(env, mi):
    fig, axes = plt.subplots(2, 1, figsize=(13, 8))
    fig.suptitle("Frozen Agent Test — Pure Coordination Behavior", fontsize=13)

    ax = axes[0]
    steps = list(range(len(env.action_log["A"])))
    ax.plot(steps, env.action_log["A"],  alpha=0.4, label="A picks",    color="blue",   linewidth=0.8)
    ax.plot(steps, env.action_log["B"],  alpha=0.4, label="B picks",    color="orange", linewidth=0.8)
    ax.plot(steps, env.target_log["A"],  alpha=0.9, label="A's target", color="blue",   linewidth=1.5, linestyle="--")
    ax.plot(steps, env.target_log["B"],  alpha=0.9, label="B's target", color="orange", linewidth=1.5, linestyle="--")
    for s in env.shift_steps:
        ax.axvline(x=s, color="red", linestyle="--", alpha=0.4,
                   label="Target Shift" if s == env.shift_steps[0] else "")
    ax.set_title("Picks vs Targets (no exploration, no learning)")
    ax.legend(ncol=3); ax.grid(True, alpha=0.3)

    ax = axes[1]
    x_mi = list(range(200, 200 + len(mi)))
    ax.plot(x_mi, mi, color="purple", label="Rolling MI (window=200)")
    for s in env.shift_steps:
        ax.axvline(x=s, color="red", linestyle="--", alpha=0.4)
    ax.set_title("MI During Frozen Test")
    ax.set_ylabel("Mutual Information (nats)")
    ax.legend(); ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("outputs/frozen_test_results.png", dpi=150, bbox_inches="tight")
    print("Frozen test plot saved -> frozen_test_results.png")
    plt.close()


# ─────────────────────────────────────────────
#  Run
# ─────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("MUTUAL REWARD EXPERIMENT")
    print("=" * 60)
    print()

    # ── Phase 1: Train ───────────────────────────────────────────────
    print("Phase 1: Training...\n")
    env, agents = train(
        episodes=12,
        max_steps=5000,
        history_len=4,       # small — keeps the notebook manageable
        shift_interval=400,  # slow — gives agents time to coordinate before disruption
        save_brains=True,
        fixed_seed=42,
    )

    # ── Phase 2: Frozen test ─────────────────────────────────────────
    print("\nPhase 2: Frozen test (no learning, no exploration)...")
    frozen_test(
        brain_path_A="outputs/brain_A.pkl",
        brain_path_B="outputs/brain_B.pkl",
        max_steps=2000,
        fixed_seed=99,
    )

    print("\nDone.")
    print("Files: mutual_reward_results.png, frozen_test_results.png,")
    print("       brain_A.pkl, brain_B.pkl")
