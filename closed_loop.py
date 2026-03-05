"""
Truly Closed Loop Experiment
==============================
This is the philosophically cleaner version of the mutual reward experiment.

The setup:
  - Each timestep, A picks a reward level to GIVE B  (0, 1, 2, 3, or 4)
  - Each timestep, B picks a reward level to GIVE A  (0, 1, 2, 3, or 4)
  - A's reward this step  = whatever B chose to give A
  - B's reward this step  = whatever A chose to give B
  - Each agent observes only the last 4 rewards the other gave them

There is NO external target. NO external success condition.
The only thing shaping each agent's behavior is the other agent's choices.

The question:
  Starting from two agents picking completely randomly, does any
  non-random pattern emerge in what they give each other?

  If yes: a closed loop with no external signal can develop internal structure.
  If no:  mutual reward alone is insufficient to bootstrap coordination.

What to look for in the output:
  - MI rising above zero  ->  their choices became statistically coupled
  - Conditional frequency tables  ->  "when A gives X, B tends to give Y"
  - Generosity drift  ->  do average reward levels rise, fall, or stabilize?
  - Reciprocity  ->  does getting a high reward cause giving a high reward?
"""

import numpy as np
from collections import deque, Counter, defaultdict
from gymnasium import spaces
from pettingzoo import ParallelEnv
from sklearn.metrics import mutual_info_score
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os

os.makedirs("cl_results", exist_ok=True)


# ─────────────────────────────────────────────
#  Environment
# ─────────────────────────────────────────────

REWARD_LEVELS = 5     # agents can give 0, 1, 2, 3, or 4
HISTORY_LEN   = 4     # how many past rewards each agent remembers

class ClosedLoopEnv(ParallelEnv):
    """
    Truly closed loop.

    Action space:  discrete, 0-4  (the reward you choose to give the other agent)
    Observation:   the last HISTORY_LEN rewards the OTHER agent gave YOU
                   — nothing else. No targets. No external signal.
    Reward:        literally whatever the other agent chose to give you this step.
    """
    metadata = {"name": "closed_loop_v0"}

    def __init__(self, max_steps=5000):
        super().__init__()
        self.max_steps       = max_steps
        self.possible_agents = ["A", "B"]
        self.agents          = self.possible_agents[:]

        # observation: last HISTORY_LEN rewards received from the other agent
        self.observation_spaces = {
            ag: spaces.Box(low=0, high=REWARD_LEVELS-1,
                           shape=(HISTORY_LEN,), dtype=np.float32)
            for ag in self.possible_agents
        }
        # action: what reward level to give the other agent
        self.action_spaces = {
            ag: spaces.Discrete(REWARD_LEVELS)
            for ag in self.possible_agents
        }

    def reset(self, seed=None, options=None):
        if seed is not None:
            np.random.seed(seed)
        self.agents    = self.possible_agents[:]
        self.timestep  = 0

        # memory of rewards received — starts at 0 (neutral)
        self.received = {
            ag: deque([0] * HISTORY_LEN, maxlen=HISTORY_LEN)
            for ag in self.possible_agents
        }

        # logs for analysis
        self.gave_log     = {"A": [], "B": []}   # what each agent gave
        self.received_log = {"A": [], "B": []}   # what each agent received

        obs   = {ag: np.array(list(self.received[ag]), dtype=np.float32)
                 for ag in self.agents}
        infos = {ag: {} for ag in self.agents}
        return obs, infos

    def step(self, actions):
        self.timestep += 1

        give_A = actions["A"]   # reward A chose to give B
        give_B = actions["B"]   # reward B chose to give A

        # rewards are what the OTHER agent decided to give
        rewards = {"A": float(give_B), "B": float(give_A)}

        # log
        self.gave_log["A"].append(give_A)
        self.gave_log["B"].append(give_B)
        self.received_log["A"].append(give_B)
        self.received_log["B"].append(give_A)

        # update memory of received rewards
        self.received["A"].append(give_B)
        self.received["B"].append(give_A)

        terminated = {ag: False for ag in self.agents}
        truncated  = {ag: self.timestep >= self.max_steps for ag in self.agents}
        if all(truncated.values()):
            self.agents = []

        obs = {
            ag: np.array(list(self.received[ag]), dtype=np.float32)
            for ag in self.possible_agents
        }
        infos = {ag: {} for ag in self.possible_agents}
        return obs, rewards, terminated, truncated, infos


# ─────────────────────────────────────────────
#  Q-Learning Agent
# ─────────────────────────────────────────────

class QLearningAgent:
    """
    The notebook agent. State = last 4 rewards received.
    Action = reward level to give (0-4).

    With no external target, the agent is purely trying to maximize
    what it receives — which means influencing what the other agent gives.
    The only tool it has is its own giving behavior.
    """
    def __init__(self, n_actions=REWARD_LEVELS, epsilon=0.5,
                 alpha=0.15, gamma=0.9):
        self.n_actions = n_actions
        self.epsilon   = epsilon
        self.alpha     = alpha
        self.gamma     = gamma
        self.q_table   = {}

    def _key(self, obs):
        return tuple(obs.astype(int))

    def get_q(self, obs):
        k = self._key(obs)
        if k not in self.q_table:
            self.q_table[k] = np.zeros(self.n_actions)
        return self.q_table[k]

    def act(self, obs):
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.n_actions)
        return int(np.argmax(self.get_q(obs)))

    def update(self, obs, action, reward, next_obs):
        q      = self.get_q(obs)
        q_next = self.get_q(next_obs)
        q[action] += self.alpha * (reward + self.gamma * np.max(q_next) - q[action])

    def decay_epsilon(self, rate=0.9998, min_eps=0.05):
        self.epsilon = max(self.epsilon * rate, min_eps)


# ─────────────────────────────────────────────
#  Pattern Analysis
# ─────────────────────────────────────────────

def analyze_patterns(env):
    """
    Look for structure in the giving/receiving logs.
    Returns a dict of findings and writes a text report.
    """
    gave_A = np.array(env.gave_log["A"])
    gave_B = np.array(env.gave_log["B"])

    lines = []
    lines.append("=" * 60)
    lines.append("CLOSED LOOP PATTERN REPORT")
    lines.append("=" * 60)

    # ── 1. Basic giving distribution ────────────────────────────
    lines.append("\n[1] How often did each agent give each reward level?")
    lines.append(f"    (random baseline would be 20% each)")
    lines.append(f"\n    {'Level':<8} {'A gave':>10} {'B gave':>10}")
    lines.append(f"    {'-'*30}")
    freq_A = Counter(gave_A)
    freq_B = Counter(gave_B)
    for lvl in range(REWARD_LEVELS):
        pA = freq_A.get(lvl, 0) / len(gave_A) * 100
        pB = freq_B.get(lvl, 0) / len(gave_B) * 100
        lines.append(f"    {lvl:<8} {pA:>9.1f}%  {pB:>9.1f}%")

    avg_A = np.mean(gave_A)
    avg_B = np.mean(gave_B)
    lines.append(f"\n    Avg reward A gave: {avg_A:.3f}  (max possible: {REWARD_LEVELS-1})")
    lines.append(f"    Avg reward B gave: {avg_B:.3f}  (random expected: {(REWARD_LEVELS-1)/2:.1f})")

    # ── 2. Reciprocity: when I receive high, do I give high back? ─
    lines.append("\n[2] Reciprocity — when A receives reward X, what does A give next step?")
    lines.append("    (positive reciprocity = getting more makes you give more)")
    lines.append(f"\n    {'A received':<14} {'A gave next (avg)':>20}  {'count':>8}")
    lines.append(f"    {'-'*44}")
    for received_val in range(REWARD_LEVELS):
        mask = gave_B[:-1] == received_val   # B gave A this value at step t
        if mask.sum() > 0:
            next_gave = gave_A[1:][mask]     # what A gave at step t+1
            lines.append(f"    {received_val:<14} {np.mean(next_gave):>20.3f}  {mask.sum():>8}")

    lines.append(f"\n    {'B received':<14} {'B gave next (avg)':>20}  {'count':>8}")
    lines.append(f"    {'-'*44}")
    for received_val in range(REWARD_LEVELS):
        mask = gave_A[:-1] == received_val
        if mask.sum() > 0:
            next_gave = gave_B[1:][mask]
            lines.append(f"    {received_val:<14} {np.mean(next_gave):>20.3f}  {mask.sum():>8}")

    # ── 3. Conditional giving: when A gives X, what does B give? ─
    lines.append("\n[3] Conditional table — when A gives X this step, what does B give NEXT step?")
    lines.append("    (this detects if B is responding to A's generosity or lack of it)")
    lines.append(f"\n    {'A gave (t)':<14} {'B gave (t+1) avg':>20}  {'count':>8}")
    lines.append(f"    {'-'*44}")
    for a_val in range(REWARD_LEVELS):
        mask = gave_A[:-1] == a_val
        if mask.sum() > 0:
            b_next = gave_B[1:][mask]
            lines.append(f"    {a_val:<14} {np.mean(b_next):>20.3f}  {mask.sum():>8}")

    lines.append(f"\n    {'B gave (t)':<14} {'A gave (t+1) avg':>20}  {'count':>8}")
    lines.append(f"    {'-'*44}")
    for b_val in range(REWARD_LEVELS):
        mask = gave_B[:-1] == b_val
        if mask.sum() > 0:
            a_next = gave_A[1:][mask]
            lines.append(f"    {b_val:<14} {np.mean(a_next):>20.3f}  {mask.sum():>8}")

    # ── 4. MI between giving sequences ──────────────────────────
    mi_same  = mutual_info_score(gave_A, gave_B)
    mi_lag_a = mutual_info_score(gave_A[:-1], gave_B[1:])
    mi_lag_b = mutual_info_score(gave_B[:-1], gave_A[1:])
    random_mi = mutual_info_score(
        np.random.randint(0, REWARD_LEVELS, 5000),
        np.random.randint(0, REWARD_LEVELS, 5000)
    )

    lines.append("\n[4] Mutual Information (MI)")
    lines.append(f"    Random baseline MI:           {random_mi:.4f}")
    lines.append(f"    MI(A gave, B gave same step): {mi_same:.4f}")
    lines.append(f"    MI(A gave t, B gave t+1):     {mi_lag_a:.4f}  <- does A's give predict B's next give?")
    lines.append(f"    MI(B gave t, A gave t+1):     {mi_lag_b:.4f}  <- does B's give predict A's next give?")

    if mi_lag_a > random_mi * 2 or mi_lag_b > random_mi * 2:
        lines.append("\n    FINDING: Statistical dependency detected across time.")
        lines.append("    The giving sequences are coupled — not independent random noise.")
    else:
        lines.append("\n    FINDING: No significant temporal dependency detected.")
        lines.append("    Giving sequences appear statistically independent.")

    # ── 5. Generosity over time ──────────────────────────────────
    chunk = 500
    chunks_A = [np.mean(gave_A[i:i+chunk]) for i in range(0, len(gave_A)-chunk, chunk)]
    chunks_B = [np.mean(gave_B[i:i+chunk]) for i in range(0, len(gave_B)-chunk, chunk)]
    lines.append("\n[5] Average generosity per 500-step chunk (did it drift?)")
    lines.append(f"    {'Chunk':<8} {'A avg give':>12} {'B avg give':>12}")
    lines.append(f"    {'-'*34}")
    for i, (ca, cb) in enumerate(zip(chunks_A, chunks_B)):
        lines.append(f"    {i+1:<8} {ca:>12.3f} {cb:>12.3f}")

    lines.append("\n" + "=" * 60)

    report = "\n".join(lines)
    print(report)
    path = "outputs/closed_loop_patterns.txt"
    with open(path, "w") as f:
        f.write(report)
    print(f"\nText report saved -> {path}")

    return {
        "gave_A": gave_A, "gave_B": gave_B,
        "avg_A": avg_A, "avg_B": avg_B,
        "mi_same": mi_same, "mi_lag_a": mi_lag_a, "mi_lag_b": mi_lag_b,
        "chunks_A": chunks_A, "chunks_B": chunks_B,
    }


# ─────────────────────────────────────────────
#  Visualizations
# ─────────────────────────────────────────────

def rolling_mi(seq_a, seq_b, window=300):
    out = []
    for i in range(window, len(seq_a)+1):
        out.append(mutual_info_score(seq_a[i-window:i], seq_b[i-window:i]))
    return out


def visualize(env, stats, reward_log_A, reward_log_B):
    gave_A   = stats["gave_A"]
    gave_B   = stats["gave_B"]
    steps    = np.arange(len(gave_A))
    n_levels = REWARD_LEVELS

    fig = plt.figure(figsize=(15, 22))
    gs  = gridspec.GridSpec(5, 2, figure=fig, hspace=0.45, wspace=0.35)
    fig.suptitle("Closed Loop Experiment — Reward Giving Analysis", fontsize=14, fontweight="bold")

    # ── Plot 1: Smoothed giving over time ───────────────────────
    ax = fig.add_subplot(gs[0, :])
    w = 150
    smooth_A = np.convolve(gave_A, np.ones(w)/w, mode="valid")
    smooth_B = np.convolve(gave_B, np.ones(w)/w, mode="valid")
    ax.plot(smooth_A, color="blue",   label="A gives (smoothed)", alpha=0.8)
    ax.plot(smooth_B, color="orange", label="B gives (smoothed)", alpha=0.8)
    ax.axhline(y=(n_levels-1)/2, color="gray", linestyle="--", alpha=0.6, label="Random expected")
    ax.set_ylabel("Avg reward given")
    ax.set_xlabel("Timestep")
    ax.set_title("Generosity over time  |  Does giving drift up, down, or stay random?")
    ax.legend(); ax.grid(True, alpha=0.2)

    # ── Plot 2: Rolling MI ───────────────────────────────────────
    ax = fig.add_subplot(gs[1, :])
    mi_vals = rolling_mi(list(gave_A), list(gave_B), window=300)
    ax.plot(range(300, 300+len(mi_vals)), mi_vals, color="purple", label="Rolling MI (window=300)")
    random_mi = mutual_info_score(
        np.random.randint(0, n_levels, 5000), np.random.randint(0, n_levels, 5000)
    )
    ax.axhline(y=random_mi, color="gray", linestyle="--", alpha=0.6, label=f"Random baseline ({random_mi:.4f})")
    ax.set_ylabel("Mutual Information (nats)")
    ax.set_xlabel("Timestep")
    ax.set_title("Rolling MI between giving sequences  |  Rising = hidden coupling")
    ax.legend(); ax.grid(True, alpha=0.2)

    # ── Plot 3: Action distribution — A ─────────────────────────
    ax = fig.add_subplot(gs[2, 0])
    freq_A = [np.sum(gave_A == v)/len(gave_A)*100 for v in range(n_levels)]
    bars = ax.bar(range(n_levels), freq_A, color="blue", alpha=0.7)
    ax.axhline(y=100/n_levels, color="gray", linestyle="--", alpha=0.6, label="Random (20%)")
    ax.set_xlabel("Reward given"); ax.set_ylabel("% of timesteps")
    ax.set_title("A's giving distribution\n(flat = random, spike = preference)")
    ax.set_xticks(range(n_levels))
    for bar, val in zip(bars, freq_A):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.3,
                f"{val:.1f}%", ha="center", fontsize=8)
    ax.legend(); ax.grid(True, alpha=0.2, axis="y")

    # ── Plot 4: Action distribution — B ─────────────────────────
    ax = fig.add_subplot(gs[2, 1])
    freq_B = [np.sum(gave_B == v)/len(gave_B)*100 for v in range(n_levels)]
    bars = ax.bar(range(n_levels), freq_B, color="orange", alpha=0.7)
    ax.axhline(y=100/n_levels, color="gray", linestyle="--", alpha=0.6, label="Random (20%)")
    ax.set_xlabel("Reward given"); ax.set_ylabel("% of timesteps")
    ax.set_title("B's giving distribution\n(flat = random, spike = preference)")
    ax.set_xticks(range(n_levels))
    for bar, val in zip(bars, freq_B):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.3,
                f"{val:.1f}%", ha="center", fontsize=8)
    ax.legend(); ax.grid(True, alpha=0.2, axis="y")

    # ── Plot 5: Reciprocity heatmap — what B gives after A gives X ─
    # Each row = what A gave at time t
    # Color = average of what B gave at time t+1
    # Hot color = B tends to give a lot after A gave that amount
    ax = fig.add_subplot(gs[3, 0])
    matrix_B_after_A = np.zeros((n_levels, n_levels))
    for a_val in range(n_levels):
        mask = gave_A[:-1] == a_val
        if mask.sum() > 0:
            b_next = gave_B[1:][mask]
            for b_val in range(n_levels):
                matrix_B_after_A[a_val, b_val] = np.sum(b_next == b_val) / len(b_next)
    im = ax.imshow(matrix_B_after_A, cmap="Blues", aspect="auto", vmin=0, vmax=0.5)
    ax.set_xlabel("B gave at t+1")
    ax.set_ylabel("A gave at t")
    ax.set_title("When A gives X, what does B give next?\n(bright = common combination)")
    ax.set_xticks(range(n_levels)); ax.set_yticks(range(n_levels))
    plt.colorbar(im, ax=ax, label="Proportion")
    for i in range(n_levels):
        for j in range(n_levels):
            ax.text(j, i, f"{matrix_B_after_A[i,j]:.2f}",
                    ha="center", va="center", fontsize=7,
                    color="white" if matrix_B_after_A[i,j] > 0.3 else "black")

    # ── Plot 6: Reciprocity heatmap — what A gives after B gives X ─
    ax = fig.add_subplot(gs[3, 1])
    matrix_A_after_B = np.zeros((n_levels, n_levels))
    for b_val in range(n_levels):
        mask = gave_B[:-1] == b_val
        if mask.sum() > 0:
            a_next = gave_A[1:][mask]
            for a_val in range(n_levels):
                matrix_A_after_B[b_val, a_val] = np.sum(a_next == a_val) / len(a_next)
    im = ax.imshow(matrix_A_after_B, cmap="Oranges", aspect="auto", vmin=0, vmax=0.5)
    ax.set_xlabel("A gave at t+1")
    ax.set_ylabel("B gave at t")
    ax.set_title("When B gives X, what does A give next?\n(bright = common combination)")
    ax.set_xticks(range(n_levels)); ax.set_yticks(range(n_levels))
    plt.colorbar(im, ax=ax, label="Proportion")
    for i in range(n_levels):
        for j in range(n_levels):
            ax.text(j, i, f"{matrix_A_after_B[i,j]:.2f}",
                    ha="center", va="center", fontsize=7,
                    color="white" if matrix_A_after_B[i,j] > 0.3 else "black")

    # ── Plot 7: Cumulative reward over time ─────────────────────
    ax = fig.add_subplot(gs[4, :])
    cum_A = np.cumsum(reward_log_A)
    cum_B = np.cumsum(reward_log_B)
    ax.plot(cum_A, color="blue",   label="A's cumulative reward", alpha=0.8)
    ax.plot(cum_B, color="orange", label="B's cumulative reward", alpha=0.8)
    # random baseline: each step expected = (0+1+2+3+4)/5 = 2.0
    random_cum = np.arange(len(reward_log_A)) * ((n_levels - 1) / 2)
    ax.plot(random_cum, color="gray", linestyle="--", alpha=0.5, label="Random expected")
    ax.set_xlabel("Timestep")
    ax.set_ylabel("Cumulative reward")
    ax.set_title("Cumulative reward  |  Above gray = earning above random")
    ax.legend(); ax.grid(True, alpha=0.2)

    plt.savefig("outputs/closed_loop_results.png", dpi=150, bbox_inches="tight")
    print("Visualization saved -> outputs/closed_loop_results.png")
    plt.close()


# ─────────────────────────────────────────────
#  Training Loop
# ─────────────────────────────────────────────

def train(episodes=15, max_steps=5000, seed=42):
    env = ClosedLoopEnv(max_steps=max_steps)
    agents = {
        "A": QLearningAgent(),
        "B": QLearningAgent(),
    }

    all_rewards = {"A": [], "B": []}
    print("=" * 60)
    print("CLOSED LOOP EXPERIMENT")
    print("No external targets. No external reward signal.")
    print("Each agent's reward = whatever the other chooses to give.")
    print("=" * 60)
    print()

    for ep in range(episodes):
        obs, _     = env.reset(seed=ep)
        prev_obs   = {ag: obs[ag].copy() for ag in env.possible_agents}
        ep_rewards = {"A": 0.0, "B": 0.0}

        while env.agents:
            actions  = {ag: agents[ag].act(obs[ag]) for ag in env.agents}
            next_obs, rewards, _, _, _ = env.step(actions)

            for ag in env.possible_agents:
                if ag in rewards:
                    agents[ag].update(prev_obs[ag], actions[ag],
                                      rewards[ag], next_obs[ag])
                    ep_rewards[ag] += rewards[ag]

            prev_obs = {ag: next_obs[ag].copy() for ag in env.possible_agents}
            obs      = next_obs

            for ag in env.possible_agents:
                agents[ag].decay_epsilon()

        for ag in env.possible_agents:
            all_rewards[ag].append(ep_rewards[ag])

        freq_A = Counter(env.gave_log["A"])
        freq_B = Counter(env.gave_log["B"])
        total  = len(env.gave_log["A"])
        fA = " ".join(f"{v}:{freq_A.get(v,0)/total*100:4.1f}%" for v in range(REWARD_LEVELS))
        fB = " ".join(f"{v}:{freq_B.get(v,0)/total*100:4.1f}%" for v in range(REWARD_LEVELS))

        mi = mutual_info_score(env.gave_log["A"], env.gave_log["B"])
        print(f"Episode {ep+1:>2}/{episodes} | "
              f"Reward A: {ep_rewards['A']:>7.1f} | "
              f"Reward B: {ep_rewards['B']:>7.1f} | "
              f"MI: {mi:.4f}")
        print(f"  A gave -> {fA}")
        print(f"  B gave -> {fB}")

    # use the last episode's environment for analysis
    stats = analyze_patterns(env)
    visualize(env, stats, env.received_log["A"], env.received_log["B"])

    return env, agents


# ─────────────────────────────────────────────
#  Run
# ─────────────────────────────────────────────

if __name__ == "__main__":
    train(episodes=10, max_steps=50000, seed=42)
    print("\nDone.")
    print("Check outputs/closed_loop_results.png and outputs/closed_loop_patterns.txt")
