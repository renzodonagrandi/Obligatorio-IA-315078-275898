# src/qlearning/train_qlearning.py
import argparse
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import gymnasium as gym
import numpy as np
import os
import time
import csv
import pickle

from utils.discretization import Discretizer, create_discretizers  # espera que tengas utils/discretization.py
# si lo pusiste distinto, ajustá el import

def make_q_table(bins, n_actions):
    shape = tuple(bins) + (n_actions,)
    return np.zeros(shape, dtype=np.float32)

def argmax_q(q_vals):
    return int(np.argmax(q_vals))

def epsilon_greedy_action(q_table, state, n_actions, epsilon):
    if np.random.rand() < epsilon:
        return int(np.random.randint(n_actions))
    else:
        return argmax_q(q_table[state])

def run_training(env, discretizer, bins, alpha, gamma, epsilon_start, epsilon_min, epsilon_decay, n_episodes, max_steps, out_dir, seed):
    np.random.seed(seed)
    env.reset(seed=seed)
    n_actions = env.action_space.n
    Q = make_q_table(bins, n_actions)

    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, "results.csv")
    pkl_path = os.path.join(out_dir, "final_q.pkl")

    # header CSV
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["episode","total_reward","epsilon","alpha","gamma","elapsed"])

    epsilon = epsilon_start
    start_time = time.time()

    for ep in range(1, n_episodes+1):
        obs, _ = env.reset()
        state = discretizer.discretize(obs)
        total_reward = 0.0

        for t in range(max_steps):
            action = epsilon_greedy_action(Q, state, n_actions, epsilon)
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            next_state = discretizer.discretize(next_obs)

            # Q-learning update
            best_next = np.max(Q[next_state])
            td_target = reward + gamma * best_next
            td_error = td_target - Q[state + (action,)]
            Q[state + (action,)] += alpha * td_error

            state = next_state
            total_reward += reward
            if done:
                break

        # log
        elapsed = time.time() - start_time
        with open(csv_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([ep, total_reward, epsilon, alpha, gamma, elapsed])

        # decay epsilon
        epsilon = max(epsilon_min, epsilon * epsilon_decay)

        # periodic checkpoint every 100 episodes
        if ep % 100 == 0:
            with open(os.path.join(out_dir, f"q_ep{ep}.pkl"), "wb") as pf:
                pickle.dump(Q, pf)

    # final save
    with open(pkl_path, "wb") as pf:
        pickle.dump(Q, pf)

    return csv_path, pkl_path

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--discret", choices=["A_coarse","A_5000","A_fine"], required=True)
    p.add_argument("--alpha", type=float, default=0.05)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--epsilon_start", type=float, default=1.0)
    p.add_argument("--epsilon_min", type=float, default=0.05)
    p.add_argument("--epsilon_decay", type=float, default=0.99)
    p.add_argument("--episodes", type=int, default=3000)
    p.add_argument("--max_steps", type=int, default=500)
    p.add_argument("--outdir", type=str, default="experiments")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()

if __name__ == "__main__":
    args = parse_args()
    env = gym.make("CartPole-v1")
    discretizers = create_discretizers(env)

    # mapping discret names to bins (must match your discretizer implementation)
    bins_map = {
        "A_coarse": [3,3,6,3],      # 162 states — if tenés otra A_coarse, ajustar
        "A_5000":  [7,7,15,7],      # ≈5,145 estados
        "A_fine":  [10,10,10,10],   # 10,000 estados (fina)
    }

    if args.discret not in discretizers:
        print("Discretizer not found. Available:", list(discretizers.keys()))
        raise SystemExit(1)

    discretizer = discretizers[args.discret]
    bins = bins_map[args.discret]

    out_name = f"{args.discret}_alpha{args.alpha}_gamma{args.gamma}_decay{args.epsilon_decay}_ep{args.episodes}_seed{args.seed}"
    out_dir = os.path.join(args.outdir, out_name)
    print("Running experiment:", out_name, "-> saving to", out_dir)

    csv_path, pkl_path = run_training(
        env=env,
        discretizer=discretizer,
        bins=bins,
        alpha=args.alpha,
        gamma=args.gamma,
        epsilon_start=args.epsilon_start,
        epsilon_min=args.epsilon_min,
        epsilon_decay=args.epsilon_decay,
        n_episodes=args.episodes,
        max_steps=args.max_steps,
        out_dir=out_dir,
        seed=args.seed
    )

    print("Finished. CSV:", csv_path, "Q-table:", pkl_path)