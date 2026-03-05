"""
Brain Analyzer
--------------
Load saved agent brains, run them frozen (no learning, no exploration),
and produce clean readable visualizations with pattern detection.

Usage:
    python analyze_brains.py

Expects brain_A.pkl and brain_B.pkl to exist in outputs/
These are saved automatically after training in mutual_reward_env.py
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pickle
from collections import deque, Counter
from gymnasium import spaces
from pettingzoo import ParallelEnv


# ─────────────────────────────────────────────
#  Paste the environment here so this file
#  can run standalone without importing the
#  training file
# ─────────────────────────────────────────────

class MutualRewardEnv(ParallelEnv):
    metadata = {"name": "mutual_reward_v0"}

    def __init__(self, history_len=4, shift_interval=400,
                 n_actions=5, graded_reward=True, max_steps=3000):
        super().__init__()
        self.history_len    = history_len
        self.shift_interval = shift_interval
        self.n_actions      = n_actions
        self.graded_reward  = graded_reward
        self.max_steps      = max_steps
        self.possible_agents = ["A", "B"]
        self.agents          = self.possible_agents[:]
        obs_size = history_len * 2 + 1
        self.observation_spaces = {
            a: spaces.Box(low=0, high=n_actions, shape=(obs_size,), dtype=np.float32)
            for a in self.possible_agents
        }
        self.action_spaces = {
            a: spaces.Discrete(n_actions) for a in self.possible_agents
        }

    def _sample_target(self):
        return np.random.randint(1, self.n_actions + 1)

    def _compute_reward(self, pick, target):
        if self.graded_reward:
            return 1.0 - abs(pick - target) / (self.n_actions - 1)
        return 1.0 if pick == target else 0.0

    def _get_obs(self, agent):
        partner = "B" if agent == "A" else "A"
        return np.array(
            list(self.history[agent]) +
            list(self.history[partner]) +
            [self.targets[agent]],
            dtype=np.float32
        )

    def reset(self, seed=None, options=None):
        if seed is not None:
            np.random.seed(seed)
        self.agents     = self.possible_agents[:]
        self.timestep   = 0
        self.targets    = {a: self._sample_target() for a in self.agents}
        self.history    = {
            a: deque([0] * self.history_len, maxlen=self.history_len)
            for a in self.agents
        }
        self.action_log = {"A": [], "B": []}
        self.reward_log = {"A": [], "B": []}
        self.target_log = {"A": [], "B": []}
        self.shift_steps = []
        obs   = {a: self._get_obs(a) for a in self.agents}
        infos = {a: {} for a in self.agents}
        return obs, infos

    def step(self, actions):
        self.timestep += 1
        a_pick = actions["A"] + 1
        b_pick = actions["B"] + 1
        r_a = self._compute_reward(b_pick, self.targets["A"])
        r_b = self._compute_reward(a_pick, self.targets["B"])
        rewards = {"A": r_a, "B": r_b}
        for ag, pk in [("A", a_pick), ("B", b_pick)]:
            self.action_log[ag].append(pk)
            self.reward_log[ag].append(rewards[ag])
            self.target_log[ag].append(self.targets[ag])
            self.history[ag].append(pk)
        if self.timestep % self.shift_interval == 0:
            self.targets["A"] = self._sample_target()
            self.targets["B"] = self._sample_target()
            self.shift_steps.append(self.timestep)
        terminated = {a: False for a in self.agents}
        truncated  = {a: self.timestep >= self.max_steps for a in self.agents}
        if all(truncated.values()):
            self.agents = []
        obs   = {a: self._get_obs(a) for a in self.possible_agents}
        infos = {a: {} for a in self.possible_agents}
        return obs, rewards, terminated, truncated, infos


# ─────────────────────────────────────────────
#  Agent (minimal — just act, no training)
# ─────────────────────────────────────────────

class QLearningAgent:
    def __init__(self, n_actions):
        self.n_actions = n_actions
        self.q_table   = {}

    def _key(self, obs):
        return tuple(obs.astype(int))

    def act(self, obs, frozen=False):
            key = self._key(obs)
            if key not in self.q_table:
                # if frozen and we've never seen this state, just pick 0
                return 0 if frozen else np.random.randint(self.n_actions)
            if not frozen and np.random.rand() < 0.05:
                return np.random.randint(self.n_actions)
            return int(np.argmax(self.q_table[key]))
    def update(self, obs, action, reward, next_obs):
            key      = self._key(obs)
            next_key = self._key(next_obs)
            if key not in self.q_table:
                self.q_table[key] = np.zeros(self.n_actions)
            if next_key not in self.q_table:
                self.q_table[next_key] = np.zeros(self.n_actions)
            q      = self.q_table[key]
            q_next = self.q_table[next_key]
            q[action] += 0.1 * (reward + 0.95 * np.max(q_next) - q[action])

    def decay_epsilon(self, rate=0.999, min_eps=0.05):
        self.epsilon = max(getattr(self, 'epsilon', 1.0) * rate, min_eps)

    @classmethod
    def load(cls, path):
        with open(path, "rb") as f:
            data = pickle.load(f)
        agent = cls(n_actions=data["n_actions"])
        agent.q_table = data["q_table"]
        print(f"  Loaded {path}  ({len(agent.q_table):,} notebook entries)")
        return agent


# ─────────────────────────────────────────────
#  Run a frozen episode and collect data
# ─────────────────────────────────────────────

def run_frozen(path_A, path_B, max_steps=3000, shift_interval=400, seed=77):
    print("Running frozen episode...")
    env     = MutualRewardEnv(max_steps=max_steps, shift_interval=shift_interval)
    agent_A = QLearningAgent.load(path_A)
    agent_B = QLearningAgent.load(path_B)

    obs, _ = env.reset(seed=seed)
    while env.agents:
        actions = {
            "A": agent_A.act(obs["A"]),
            "B": agent_B.act(obs["B"]),
        }
        obs, _, _, _, _ = env.step(actions)

    return env


# ─────────────────────────────────────────────
#  Pattern detection
# ─────────────────────────────────────────────

def detect_patterns(env):
    """
    Analyse what actually happened and print plain-English findings.
    """
    picks_A  = np.array(env.action_log["A"])
    picks_B  = np.array(env.action_log["B"])
    target_A = np.array(env.target_log["A"])
    target_B = np.array(env.target_log["B"])

    print("\n" + "=" * 55)
    print("PATTERN DETECTION REPORT")
    print("=" * 55)

    # ── 1. Did either agent collapse to one number? ──────────────
    dist_A = Counter(picks_A)
    dist_B = Counter(picks_B)
    most_common_A, count_A = dist_A.most_common(1)[0]
    most_common_B, count_B = dist_B.most_common(1)[0]
    pct_A = count_A / len(picks_A) * 100
    pct_B = count_B / len(picks_B) * 100

    print(f"\n[1] Action diversity")
    print(f"  A picked {most_common_A} in {pct_A:.1f}% of steps  (collapse threshold ~70%)")
    print(f"  B picked {most_common_B} in {pct_B:.1f}% of steps")
    if pct_A > 70:
        print(f"  ⚠  A has COLLAPSED — always picks {most_common_A}. "
              f"Not signaling, not reading. Just defaulting.")
    if pct_B > 70:
        print(f"  ⚠  B has COLLAPSED — always picks {most_common_B}.")
    if pct_A <= 70 and pct_B <= 70:
        print("  ✓  Both agents are varying their picks — no collapse detected.")

    # ── 2. Is A's pick tracking B's target? ─────────────────────
    # A earns reward when B picks A's target.
    # B earns reward when A picks B's target.
    # So for B to get high reward, A must pick close to B's target.
    a_tracks_b_target = np.mean(np.abs(picks_A - target_B))
    b_tracks_a_target = np.mean(np.abs(picks_B - target_A))
    random_baseline   = np.mean([abs(np.random.randint(1, 6) - np.random.randint(1, 6))
                                  for _ in range(10000)])

    print(f"\n[2] Tracking accuracy  (lower = better, random baseline = {random_baseline:.2f})")
    print(f"  A's picks vs B's target:  avg distance {a_tracks_b_target:.2f}")
    print(f"  B's picks vs A's target:  avg distance {b_tracks_a_target:.2f}")
    if a_tracks_b_target < random_baseline * 0.85:
        print("  ✓  A is meaningfully tracking B's target — A is reading B.")
    else:
        print("  ✗  A is not tracking B's target better than random.")
    if b_tracks_a_target < random_baseline * 0.85:
        print("  ✓  B is meaningfully tracking A's target — B is reading A.")
    else:
        print("  ✗  B is not tracking A's target better than random.")

    # ── 3. Does A's pick PREDICT B's next pick? ─────────────────
    # This tests whether A is signaling: does knowing A's pick this step
    # tell you anything about what B will do next step?
    from sklearn.metrics import mutual_info_score
    mi_a_predicts_b = mutual_info_score(picks_A[:-1], picks_B[1:])
    mi_b_predicts_a = mutual_info_score(picks_B[:-1], picks_A[1:])
    mi_random = mutual_info_score(
        np.random.randint(1, 6, 5000),
        np.random.randint(1, 6, 5000)
    )

    print(f"\n[3] Signal detection  (does one agent's pick predict the other's next pick?)")
    print(f"  MI(A_t -> B_t+1): {mi_a_predicts_b:.4f}  (random baseline ≈ {mi_random:.4f})")
    print(f"  MI(B_t -> A_t+1): {mi_b_predicts_a:.4f}")
    if mi_a_predicts_b > mi_random * 2:
        print("  ✓  A's actions predict B's future behavior — A is signaling something B reads.")
    else:
        print("  ✗  A's picks do not predict B's future picks.")
    if mi_b_predicts_a > mi_random * 2:
        print("  ✓  B's actions predict A's future behavior — B is signaling something A reads.")
    else:
        print("  ✗  B's picks do not predict A's future picks.")

    # ── 4. Per-segment analysis across target windows ───────────
    print(f"\n[4] Per-window tracking  (one row per target period)")
    shifts = [0] + env.shift_steps + [len(picks_A)]
    print(f"  {'Window':<10} {'A target':<10} {'B target':<10} "
          f"{'A picks (mode)':<16} {'B picks (mode)':<16} "
          f"{'A→B dist':<10} {'B→A dist':<10}")
    print("  " + "-" * 82)
    for i in range(len(shifts) - 1):
        start, end = shifts[i], shifts[i+1]
        if end - start < 5:
            continue
        seg_A  = picks_A[start:end]
        seg_B  = picks_B[start:end]
        seg_tA = target_A[start:end]
        seg_tB = target_B[start:end]
        mode_A = Counter(seg_A).most_common(1)[0][0]
        mode_B = Counter(seg_B).most_common(1)[0][0]
        d_AB   = np.mean(np.abs(seg_A - seg_tB))
        d_BA   = np.mean(np.abs(seg_B - seg_tA))
        print(f"  {start:<10} {seg_tA[0]:<10} {seg_tB[0]:<10} "
              f"{mode_A} ({Counter(seg_A).most_common(1)[0][1]}x)       "
              f"{mode_B} ({Counter(seg_B).most_common(1)[0][1]}x)       "
              f"{d_AB:<10.2f} {d_BA:<10.2f}")

    print("=" * 55)
    return {
        "collapse_A": pct_A > 70,
        "collapse_B": pct_B > 70,
        "a_tracks_b": a_tracks_b_target,
        "b_tracks_a": b_tracks_a_target,
        "random_baseline": random_baseline,
        "mi_a_to_b": mi_a_predicts_b,
        "mi_b_to_a": mi_b_predicts_a,
    }


# ─────────────────────────────────────────────
#  Clean Visualization
# ─────────────────────────────────────────────

def visualize(env, stats, output_path="outputs/brain_analysis.png"):
    picks_A  = np.array(env.action_log["A"])
    picks_B  = np.array(env.action_log["B"])
    target_A = np.array(env.target_log["A"])
    target_B = np.array(env.target_log["B"])
    steps    = np.arange(len(picks_A))

    fig, axes = plt.subplots(5, 1, figsize=(14, 18))
    fig.suptitle("Brain Analysis — Frozen Agents", fontsize=14, fontweight="bold")

    shift_color = "red"

    def draw_shifts(ax):
        for s in env.shift_steps:
            ax.axvline(x=s, color=shift_color, linestyle="--", alpha=0.4, linewidth=1)

    # ── Plot 1: What A picked vs B's target ─────────────────────
    # This tells you how well A is reading B's signal.
    ax = axes[0]
    ax.step(steps, target_B, where="post", color="orange", linewidth=2,
            label="B's target  (what A SHOULD pick)", alpha=0.9)
    ax.scatter(steps[::5], picks_A[::5], color="blue", s=4, alpha=0.5,
               label="A's actual pick")
    draw_shifts(ax)
    ax.set_ylabel("Number (1-5)")
    ax.set_title("A's picks vs B's target  |  How well is A reading B?")
    ax.set_ylim(0.5, 5.5); ax.set_yticks([1, 2, 3, 4, 5])
    ax.legend(loc="upper right"); ax.grid(True, alpha=0.2)

    # ── Plot 2: What B picked vs A's target ─────────────────────
    ax = axes[1]
    ax.step(steps, target_A, where="post", color="blue", linewidth=2,
            label="A's target  (what B SHOULD pick)", alpha=0.9)
    ax.scatter(steps[::5], picks_B[::5], color="orange", s=4, alpha=0.5,
               label="B's actual pick")
    draw_shifts(ax)
    ax.set_ylabel("Number (1-5)")
    ax.set_title("B's picks vs A's target  |  How well is B reading A?")
    ax.set_ylim(0.5, 5.5); ax.set_yticks([1, 2, 3, 4, 5])
    ax.legend(loc="upper right"); ax.grid(True, alpha=0.2)

    # ── Plot 3: Action distribution histograms ───────────────────
    ax = axes[2]
    x = np.arange(1, 6)
    dist_A = [np.sum(picks_A == v) / len(picks_A) * 100 for v in x]
    dist_B = [np.sum(picks_B == v) / len(picks_B) * 100 for v in x]
    width  = 0.35
    bars_a = ax.bar(x - width/2, dist_A, width, label="Agent A", color="blue",  alpha=0.7)
    bars_b = ax.bar(x + width/2, dist_B, width, label="Agent B", color="orange", alpha=0.7)
    ax.axhline(y=20, color="gray", linestyle="--", alpha=0.6, label="Random (20% each)")
    ax.set_xlabel("Action (number picked)")
    ax.set_ylabel("% of timesteps")
    ax.set_title("Action Distribution  |  A flat bar = collapse to one number")
    ax.set_xticks(x)
    ax.legend(); ax.grid(True, alpha=0.2, axis="y")
    for bar in bars_a:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f"{bar.get_height():.0f}%", ha="center", va="bottom", fontsize=8, color="blue")
    for bar in bars_b:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f"{bar.get_height():.0f}%", ha="center", va="bottom", fontsize=8, color="darkorange")

    # ── Plot 4: Smoothed reward trajectories ────────────────────
    ax = axes[3]
    w  = 60
    r_a = np.convolve(env.reward_log["A"], np.ones(w)/w, mode="valid")
    r_b = np.convolve(env.reward_log["B"], np.ones(w)/w, mode="valid")
    ax.plot(r_a, label="Agent A (smoothed)", color="blue",   alpha=0.8)
    ax.plot(r_b, label="Agent B (smoothed)", color="orange", alpha=0.8)
    draw_shifts(ax)
    ax.set_ylabel("Reward (smoothed)")
    ax.set_ylim(0, 1.05)
    ax.set_title("Smoothed Reward  |  Dips at red lines = disruption at target shift")
    ax.legend(); ax.grid(True, alpha=0.2)

    # ── Plot 5: Summary stats as text panel ──────────────────────
    ax = axes[4]
    ax.axis("off")
    baseline = stats["random_baseline"]
    lines = [
        ("SUMMARY", "", "black", True),
        ("", "", "black", False),
        ("Collapse check",
         f"A uses one number {stats['collapse_A'] and 'YES ⚠' or 'No ✓'}   "
         f"B uses one number {stats['collapse_B'] and 'YES ⚠' or 'No ✓'}",
         "black", False),
        ("Tracking accuracy",
         f"A→B target distance: {stats['a_tracks_b']:.2f}  "
         f"B→A target distance: {stats['b_tracks_a']:.2f}  "
         f"(random = {baseline:.2f})",
         "black", False),
        ("Signal detection",
         f"MI(A predicts B next step): {stats['mi_a_to_b']:.4f}   "
         f"MI(B predicts A next step): {stats['mi_b_to_a']:.4f}",
         "black", False),
        ("", "", "black", False),
        ("Interpretation", "", "black", True),
    ]

    # plain english interpretation
    if stats["collapse_A"] and not stats["collapse_B"]:
        interp = ("A stopped trying to signal and locked onto one number. "
                  "B adapted to reading A's static behavior. "
                  "No genuine mutual communication — A is broadcasting noise, B is coping.")
    elif stats["a_tracks_b"] < baseline * 0.85 and stats["b_tracks_a"] < baseline * 0.85:
        interp = ("Both agents are tracking each other's targets better than random. "
                  "Mutual communication is occurring.")
    elif stats["mi_a_to_b"] > 0.01 or stats["mi_b_to_a"] > 0.01:
        interp = ("Picks carry statistical dependency across time — hidden structure present "
                  "even if tracking is imperfect.")
    else:
        interp = "No clear coordination signal detected. Agents may need more training."

    lines.append((interp, "", "navy", False))

    y = 0.95
    for label, value, color, bold in lines:
        weight = "bold" if bold else "normal"
        text   = f"{label}  {value}" if value else label
        ax.text(0.01, y, text, transform=ax.transAxes, fontsize=10,
                verticalalignment="top", color=color, fontweight=weight,
                wrap=True)
        y -= 0.13

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"\nVisualization saved -> {output_path}")
    plt.close()


# ─────────────────────────────────────────────
#  Run
# ─────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 55)
    print("BRAIN ANALYZER")
    print("=" * 55)

    env   = run_frozen(
        path_A         = "outputs/brain_A.pkl",
        path_B         = "outputs/brain_B.pkl",
        max_steps      = 3000,
        shift_interval = 400,
        seed           = 77,
    )
    stats = detect_patterns(env)
    visualize(env, stats, output_path="outputs/brain_analysis.png")

    print("\nDone. Check outputs/brain_analysis.png")


# ─────────────────────────────────────────────
#  Asymmetric Test
# ─────────────────────────────────────────────
#
# Freeze one agent so it acts on its saved brain but learns nothing new.
# Give the other agent a completely fresh untrained brain.
#
# What this tells you:
#   If the fresh agent earns HIGH reward and improves over time:
#       the frozen agent is a GOOD SIGNALER — its behavior is readable.
#   If the fresh agent earns LOW reward and flatlines:
#       the frozen agent's signals are unclear or absent.
#
# Run both directions:
#   asymmetric_test("outputs/brain_B.pkl", fresh_side="A")
#       freeze B, give A a fresh brain  ->  tests if B is a good signaler
#   asymmetric_test("outputs/brain_A.pkl", fresh_side="B")
#       freeze A, give B a fresh brain  ->  tests if A is a good signaler

def asymmetric_test(frozen_path, fresh_side="A", max_steps=2000,
                    shift_interval=400, seed=55):

    frozen_side = "B" if fresh_side == "A" else "A"
    print(f"\n{'='*55}")
    print(f"ASYMMETRIC TEST  frozen={frozen_side}  fresh={fresh_side}")
    print(f"{'='*55}")

    env    = MutualRewardEnv(max_steps=max_steps, shift_interval=shift_interval)
    frozen = QLearningAgent.load(frozen_path)
    fresh  = QLearningAgent(n_actions=5)
    fresh.epsilon = 1.0          # start fully random, learn from scratch

    brains = {fresh_side: fresh, frozen_side: frozen}

    obs, _   = env.reset(seed=seed)
    prev_obs = {ag: obs[ag].copy() for ag in env.possible_agents}
    totals   = {"A": 0.0, "B": 0.0}

    chunk_size    = 200
    chunk_rewards = {fresh_side: [], frozen_side: []}
    chunk_buf     = {fresh_side: 0.0, frozen_side: 0.0}
    chunk_counter = 0

    while env.agents:
        # frozen agent uses its notebook but does NOT explore randomly
        # fresh agent still explores and learns
        actions = {
            ag: brains[ag].act(obs[ag], frozen=(ag == frozen_side))
            for ag in env.agents
        }

        next_obs, rewards, _, _, _ = env.step(actions)

        for ag in env.possible_agents:
            if ag in rewards:
                totals[ag]    += rewards[ag]
                chunk_buf[ag] += rewards[ag]
                if ag == fresh_side:
                    brains[ag].update(
                        prev_obs[ag], actions[ag],
                        rewards[ag], next_obs[ag]
                    )
                    brains[ag].decay_epsilon(rate=0.999, min_eps=0.05)

        prev_obs = {ag: next_obs[ag].copy() for ag in env.possible_agents}
        obs      = next_obs
        chunk_counter += 1

        if chunk_counter % chunk_size == 0:
            for ag in [fresh_side, frozen_side]:
                chunk_rewards[ag].append(chunk_buf[ag])
                chunk_buf[ag] = 0.0

    print(f"\nTotal reward  frozen {frozen_side}: {totals[frozen_side]:.1f}  "
          f"fresh {fresh_side}: {totals[fresh_side]:.1f}")

    print(f"\nFresh {fresh_side} reward per {chunk_size}-step chunk  "
          f"(improving = frozen {frozen_side} is readable):")
    for i, r in enumerate(chunk_rewards[fresh_side]):
        bar = "█" * int(r / chunk_size * 20)
        print(f"  chunk {i+1:>2}: {r:>6.1f}  {bar}")

    random_expected = chunk_size * 0.5
    early_chunks = chunk_rewards[fresh_side][:3]
    late_chunks  = chunk_rewards[fresh_side][-3:]
    early_avg    = np.mean(early_chunks) if early_chunks else 0.0
    late_avg     = np.mean(late_chunks)  if late_chunks  else 0.0

    print(f"\n  Early avg (first 3 chunks): {early_avg:.1f}")
    print(f"  Late  avg (last  3 chunks): {late_avg:.1f}")
    print(f"  Random play expected:       {random_expected:.1f}")

    if late_avg > random_expected * 1.2:
        print(f"\n  VERDICT: frozen {frozen_side} IS a readable signaler  "
              f"(fresh {fresh_side} earned above random)")
    else:
        print(f"\n  VERDICT: frozen {frozen_side} is NOT a readable signaler  "
              f"(fresh {fresh_side} never beat random)")

    if late_avg > early_avg * 1.15:
        print(f"           Fresh {fresh_side} improved over time  "
              f"— learning is happening")
    else:
        print(f"           Fresh {fresh_side} did not improve  "
              f"— no useful signal to learn from")

    # plot the fresh agent's learning curve
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(chunk_rewards[fresh_side], marker="o", color="blue",
            label=f"Fresh {fresh_side}  (reward per {chunk_size} steps)")
    ax.axhline(y=random_expected, color="gray", linestyle="--",
               label=f"Random baseline ({random_expected:.0f})")
    ax.set_xlabel(f"Chunk  (1 chunk = {chunk_size} timesteps)")
    ax.set_ylabel("Reward earned in chunk")
    ax.set_title(
        f"Asymmetric Test — can fresh {fresh_side} learn from frozen {frozen_side}?\n"
        f"Rising curve = frozen {frozen_side} is signaling something readable"
    )
    ax.legend()
    ax.grid(True, alpha=0.3)
    out = f"outputs/asymmetric_fresh{fresh_side}.png"
    plt.tight_layout()
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Plot saved -> {out}")

    return chunk_rewards


# ─────────────────────────────────────────────
#  Main  (replaces the earlier __main__ block)
# ─────────────────────────────────────────────
#
# Comment out whichever sections you don't need.

def main():
    print("=" * 55)
    print("BRAIN ANALYZER")
    print("=" * 55)

    # ── 1. Standard frozen test (both brains loaded, no learning) ──
    env   = run_frozen(
        path_A="outputs/brain_A.pkl",
        path_B="outputs/brain_B.pkl",
        max_steps=3000,
        shift_interval=400,
        seed=77,
    )
    stats = detect_patterns(env)
    visualize(env, stats, output_path="outputs/brain_analysis.png")

    # ── 2. Asymmetric test: freeze B, fresh A ───────────────────
    #    Asks: is B a good signaler? Can a new A learn from it?
    asymmetric_test(
        frozen_path="outputs/brain_B.pkl",
        fresh_side="A",
        max_steps=2000,
        shift_interval=400,
        seed=55,
    )

    # ── 3. Asymmetric test: freeze A, fresh B ───────────────────
    #    Asks: is A a good signaler? Can a new B learn from it?
    asymmetric_test(
        frozen_path="outputs/brain_A.pkl",
        fresh_side="B",
        max_steps=2000,
        shift_interval=400,
        seed=55,
    )

    print("\nAll done.")


if __name__ == "__main__":
    main()


# ─────────────────────────────────────────────
#  Random Control
# ─────────────────────────────────────────────
#
# This is the scientific control for the entire experiment.
#
# Two agents that pick PURELY randomly — no Q-table, no learning,
# no memory, no response to anything. They just sample uniformly
# from 1-5 every step forever.
#
# We run them through the IDENTICAL environment with IDENTICAL
# measurements as the trained agents.
#
# Why this matters:
#   If the random control produces similar MI values, similar
#   action distributions, and similar asymmetric curves as the
#   trained agents — then nothing we measured was meaningful.
#   We were just measuring noise.
#
#   If the random control produces LOWER MI, FLATTER distributions,
#   and FLAT asymmetric curves — then the trained agents really did
#   develop something beyond random. The difference is the signal.
#
# This is the only way to know if any of our findings are real.

class RandomAgent:
    """
    Picks a uniformly random action every step.
    No memory. No learning. No Q-table.
    This is the null hypothesis made concrete.
    """
    def __init__(self, n_actions=5):
        self.n_actions = n_actions

    def act(self, obs, frozen=False):
        return np.random.randint(self.n_actions)

    def update(self, obs, action, reward, next_obs):
        pass   # random agents do not learn

    def decay_epsilon(self, **kwargs):
        pass


def run_random_control(max_steps=3000, shift_interval=400, seed=77):
    """
    Run two random agents through the same environment the trained
    agents faced. Collect identical measurements.
    """
    print("\n" + "=" * 55)
    print("RANDOM CONTROL — two purely random agents")
    print("No learning. No memory. Just uniform random picks.")
    print("=" * 55)

    env      = MutualRewardEnv(max_steps=max_steps, shift_interval=shift_interval)
    agent_A  = RandomAgent()
    agent_B  = RandomAgent()

    obs, _   = env.reset(seed=seed)
    while env.agents:
        actions = {
            "A": agent_A.act(obs["A"]),
            "B": agent_B.act(obs["B"]),
        }
        obs, _, _, _, _ = env.step(actions)

    return env


def run_random_asymmetric(max_steps=2000, shift_interval=400, seed=55):
    """
    Mimic the asymmetric test but with random agents.
    One random agent is 'frozen' (just keeps picking randomly).
    The other is a Q-learning agent trying to learn from it.

    If the Q-learner improves: even random behavior is learnable
    because the environment itself (not the partner) provides signal.
    If the Q-learner stays flat: improvement in real asymmetric tests
    was genuinely due to the trained partner's signals.
    """
    print("\n" + "=" * 55)
    print("RANDOM ASYMMETRIC CONTROL")
    print("Q-learner trying to learn from a purely random partner.")
    print("=" * 55)

    env      = MutualRewardEnv(max_steps=max_steps, shift_interval=shift_interval)
    random_frozen = RandomAgent()
    learner       = QLearningAgent(n_actions=5)
    learner.epsilon = 1.0

    # run with random as frozen side, learner as fresh side
    brains   = {"A": learner, "B": random_frozen}
    obs, _   = env.reset(seed=seed)
    prev_obs = {ag: obs[ag].copy() for ag in env.possible_agents}
    totals   = {"A": 0.0, "B": 0.0}

    chunk_size    = 200
    chunk_rewards = {"A": [], "B": []}
    chunk_buf     = {"A": 0.0, "B": 0.0}
    chunk_counter = 0

    while env.agents:
        actions = {
            "A": brains["A"].act(obs["A"]),
            "B": brains["B"].act(obs["B"]),
        }
        next_obs, rewards, _, _, _ = env.step(actions)

        for ag in env.possible_agents:
            if ag in rewards:
                totals[ag]    += rewards[ag]
                chunk_buf[ag] += rewards[ag]
                if ag == "A":   # only the learner updates
                    brains["A"].update(prev_obs[ag], actions[ag],
                                       rewards[ag], next_obs[ag])
                    brains["A"].decay_epsilon(rate=0.999, min_eps=0.05)

        prev_obs = {ag: next_obs[ag].copy() for ag in env.possible_agents}
        obs      = next_obs
        chunk_counter += 1

        if chunk_counter % chunk_size == 0:
            for ag in ["A", "B"]:
                chunk_rewards[ag].append(chunk_buf[ag])
                chunk_buf[ag] = 0.0

    print(f"\nTotal — learner A: {totals['A']:.1f}  random B: {totals['B']:.1f}")
    print(f"\nLearner A reward per chunk  (should stay ~flat if random partner gives no signal):")
    for i, r in enumerate(chunk_rewards["A"]):
        bar = "█" * int(r / chunk_size * 20)
        print(f"  chunk {i+1:>2}: {r:>6.1f}  {bar}")

    return env, chunk_rewards


def plot_control_comparison(trained_env, control_env,
                            trained_chunks, control_chunks,
                            output_path="outputs/random_control_comparison.png"):
    """
    Side-by-side comparison of trained agents vs random agents
    across every key metric. This is the definitive verdict plot.
    """
    from sklearn.metrics import mutual_info_score

    fig, axes = plt.subplots(3, 2, figsize=(15, 13))
    fig.suptitle(
        "Trained Agents vs Random Control — Is anything meaningful happening?",
        fontsize=13, fontweight="bold"
    )

    window = 200

    def get_rolling_mi(env_obj):
        gave_A = env_obj.action_log["A"]
        gave_B = env_obj.action_log["B"]
        out = []
        for i in range(window, len(gave_A)+1):
            out.append(mutual_info_score(gave_A[i-window:i], gave_B[i-window:i]))
        return out

    # ── Row 1: Rolling MI comparison ────────────────────────────
    for col, (env_obj, label, color) in enumerate([
        (trained_env, "Trained agents", "purple"),
        (control_env, "Random control", "gray"),
    ]):
        ax  = axes[0][col]
        mi  = get_rolling_mi(env_obj)
        ax.plot(range(window, window+len(mi)), mi, color=color, linewidth=1)
        ax.set_title(f"Rolling MI — {label}")
        ax.set_ylabel("Mutual Information (nats)")
        ax.set_xlabel("Timestep")
        ax.set_ylim(bottom=0)
        ax.grid(True, alpha=0.2)
        # annotate mean
        ax.axhline(y=np.mean(mi), color=color, linestyle="--", alpha=0.6,
                   label=f"Mean: {np.mean(mi):.4f}")
        ax.legend()

    # ── Row 2: Action distribution comparison ───────────────────
    for col, (env_obj, label, color) in enumerate([
        (trained_env, "Trained agents", ["blue", "orange"]),
        (control_env, "Random control", ["steelblue", "sandybrown"]),
    ]):
        ax = axes[1][col]
        picks_A = np.array(env_obj.action_log["A"])
        picks_B = np.array(env_obj.action_log["B"])
        x = np.arange(1, 6)
        dist_A = [np.sum(picks_A == v)/len(picks_A)*100 for v in x]
        dist_B = [np.sum(picks_B == v)/len(picks_B)*100 for v in x]
        w = 0.35
        ax.bar(x - w/2, dist_A, w, label="Agent A", color=color[0], alpha=0.7)
        ax.bar(x + w/2, dist_B, w, label="Agent B", color=color[1], alpha=0.7)
        ax.axhline(y=20, color="gray", linestyle="--", alpha=0.5, label="Random (20%)")
        ax.set_title(f"Action Distribution — {label}")
        ax.set_xlabel("Number picked"); ax.set_ylabel("% of timesteps")
        ax.set_xticks(x); ax.legend()
        ax.grid(True, alpha=0.2, axis="y")

    # ── Row 3: Asymmetric learning curves ───────────────────────
    ax = axes[2][0]
    ax.plot(trained_chunks, marker="o", color="blue",
            label="Fresh A with trained B (real test)")
    ax.axhline(y=100, color="gray", linestyle="--", label="Random baseline (100)")
    ax.set_title("Asymmetric Test — Trained partner")
    ax.set_xlabel(f"Chunk (200 steps each)")
    ax.set_ylabel("Reward per chunk")
    ax.legend(); ax.grid(True, alpha=0.2)

    ax = axes[2][1]
    ax.plot(control_chunks["A"], marker="o", color="gray",
            label="Fresh A with random B (control)")
    ax.axhline(y=100, color="gray", linestyle="--", label="Random baseline (100)")
    ax.set_title("Asymmetric Test — Random partner (control)")
    ax.set_xlabel(f"Chunk (200 steps each)")
    ax.set_ylabel("Reward per chunk")
    ax.legend(); ax.grid(True, alpha=0.2)

    # match y-axis scales so comparison is honest
    all_vals = (trained_chunks + control_chunks["A"] +
                [100, 200])
    y_min = min(all_vals) - 10
    y_max = max(all_vals) + 10
    axes[2][0].set_ylim(y_min, y_max)
    axes[2][1].set_ylim(y_min, y_max)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"\nComparison plot saved -> {output_path}")
    plt.close()

    # print the verdict clearly
    trained_mean_mi = np.mean(get_rolling_mi(trained_env))
    control_mean_mi = np.mean(get_rolling_mi(control_env))
    ratio = trained_mean_mi / control_mean_mi if control_mean_mi > 0 else float("inf")

    print("\n" + "=" * 55)
    print("VERDICT")
    print("=" * 55)
    print(f"  Trained agents mean MI:  {trained_mean_mi:.4f}")
    print(f"  Random control mean MI:  {control_mean_mi:.4f}")
    print(f"  Ratio:                   {ratio:.2f}x")
    if ratio > 2.0:
        print(f"\n  REAL SIGNAL — trained agents have {ratio:.1f}x more MI")
        print("  than random. The coupling is not just measurement noise.")
    elif ratio > 1.3:
        print(f"\n  WEAK SIGNAL — trained agents have {ratio:.1f}x more MI")
        print("  than random. Modest but possibly real.")
    else:
        print(f"\n  NO SIGNAL — trained agents MI is only {ratio:.1f}x random.")
        print("  Cannot distinguish from noise. Results were likely artifacts.")
    print("=" * 55)


def main_with_control():
    """
    Run everything including the random control comparison.
    Call this instead of main() to get the full scientific picture.
    """
    print("=" * 55)
    print("BRAIN ANALYZER — FULL RUN WITH CONTROL")
    print("=" * 55)

    # ── Trained agents frozen test ───────────────────────────────
    trained_env = run_frozen(
        path_A="outputs/brain_A.pkl",
        path_B="outputs/brain_B.pkl",
        max_steps=3000,
        shift_interval=400,
        seed=77,
    )
    stats = detect_patterns(trained_env)
    visualize(trained_env, stats, output_path="outputs/brain_analysis.png")

    # ── Random control frozen test ───────────────────────────────
    control_env = run_random_control(
        max_steps=3000,
        shift_interval=400,
        seed=77,    # SAME seed as trained test — identical environment
    )

    # ── Trained asymmetric test ──────────────────────────────────
    trained_chunks = asymmetric_test(
        frozen_path="outputs/brain_B.pkl",
        fresh_side="A",
        max_steps=2000,
        shift_interval=400,
        seed=55,
    )

    # ── Random asymmetric control ────────────────────────────────
    _, control_chunks = run_random_asymmetric(
        max_steps=2000,
        shift_interval=400,
        seed=55,    # SAME seed
    )

    # ── Side by side comparison ──────────────────────────────────
    plot_control_comparison(
        trained_env    = trained_env,
        control_env    = control_env,
        trained_chunks = trained_chunks["A"],
        control_chunks = control_chunks,
        output_path    = "outputs/random_control_comparison.png",
    )

    print("\nAll done.")
    print("Key file: outputs/random_control_comparison.png")
    print("This is the verdict on whether any of this was real.")


if __name__ == "__main__":
    main_with_control()