import os
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

def moving_average(x, w=50):
    return np.convolve(x, np.ones(w), 'valid') / w

def load_all_experiments(indir):
    experiments = []
    for folder in os.listdir(indir):
        exp_dir = os.path.join(indir, folder)
        csv_path = os.path.join(exp_dir, "results.csv")

        if not os.path.isfile(csv_path):
            continue

        df = pd.read_csv(csv_path)

                # --- parser robusto y compatible con nombres reales ---
        parts = folder.split("_")
        exp = {
            "folder": folder,
            "df": df,
            "alpha": None,
            "gamma": None,
            "decay": None,
            "episodes": None,
            "discret": None
        }

        # Detectar discretización tipo "A_5000", "A_coarse" o "A_fine"
        if folder.startswith("A_5000"):
            exp["discret"] = "A_5000"
        elif folder.startswith("A_coarse"):
            exp["discret"] = "A_coarse"
        elif folder.startswith("A_fine"):
            exp["discret"] = "A_fine"

        for part in parts:
            if part.startswith("alpha"):
                exp["alpha"] = float(part.replace("alpha", ""))

            elif part.startswith("gamma"):
                exp["gamma"] = float(part.replace("gamma", ""))

            elif part.startswith("decay"):
                exp["decay"] = float(part.replace("decay", ""))

            elif part.startswith("ep"):
                exp["episodes"] = int(part.replace("ep", ""))

        # Sólo agregamos si está todo
        if None not in (exp["alpha"], exp["gamma"], exp["decay"], exp["episodes"], exp["discret"]):
            experiments.append(exp)
        else:
            print(f"⚠ Warning: folder {folder} could not be parsed completely!")
    return experiments

def plot_rewards(experiments, outdir):
    plt.figure(figsize=(12, 7))
    for exp in experiments:
        df = exp["df"]
        y = moving_average(df["total_reward"], w=50)
        plt.plot(y, label=exp["folder"])
    plt.title("Reward (Moving Avg. 50) - All Experiments")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "summary_reward_curves.png"))
    plt.close()

def plot_by_discret(experiments, outdir):
    groups = {}
    for exp in experiments:
        groups.setdefault(exp["discret"], []).append(exp)

    for key, exps in groups.items():
        plt.figure(figsize=(12, 7))
        for exp in exps:
            df = exp["df"]
            y = moving_average(df["total_reward"], 50)
            plt.plot(y, label=exp["folder"])
        plt.title(f"Reward Curves - {key}")
        plt.xlabel("Episode")
        plt.ylabel("Reward")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, f"{key}_reward_curves.png"))
        plt.close()

def plot_heatmap(experiments, outdir):
    rows = []
    for exp in experiments:
        max_reward = exp["df"]["total_reward"].rolling(50).mean().max()
        rows.append([exp["alpha"], exp["gamma"], exp["decay"], max_reward])

    df = pd.DataFrame(rows, columns=["alpha","gamma","decay","best_reward"])

    pivot = df.pivot_table(index="alpha", columns="gamma", values="best_reward", aggfunc=np.max)

    plt.figure(figsize=(10, 8))
    sns.heatmap(pivot, annot=True, fmt=".1f", cmap="viridis")
    plt.title("Heatmap - Best Reward (moving avg 50)")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "heatmap_best_reward.png"))
    plt.close()

def save_best_params(experiments, outdir):
    best_exp = None
    best_val = -999999

    for exp in experiments:
        max_reward = exp["df"]["total_reward"].rolling(50).mean().max()
        if max_reward > best_val:
            best_val = max_reward
            best_exp = exp

    with open(os.path.join(outdir, "best_experiment.txt"), "w") as f:
        f.write("Best Experiment Found:\n")
        f.write(f"Folder: {best_exp['folder']}\n")
        f.write(f"Alpha: {best_exp['alpha']}\n")
        f.write(f"Gamma: {best_exp['gamma']}\n")
        f.write(f"Epsilon Decay: {best_exp['decay']}\n")
        f.write(f"Best Moving Avg Reward: {best_val}\n")

def plot_single_experiment(exp, outdir):
    df = exp["df"]
    plt.figure(figsize=(12, 6))
    plt.plot(df["total_reward"], alpha=0.5, label="Raw reward")
    plt.plot(moving_average(df["total_reward"]), label="Moving avg (50)")
    plt.legend()
    plt.title("Reward Curve")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, f"{exp['folder']}_plot.png"))
    plt.close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--indir", type=str, required=True)
    parser.add_argument("--single", type=str, default=None)
    args = parser.parse_args()

    experiments = load_all_experiments(args.indir)

    if args.single:
        exp = next((e for e in experiments if e["folder"] == args.single), None)
        if exp:
            plot_single_experiment(exp, args.indir)
        return

    plot_rewards(experiments, args.indir)
    plot_by_discret(experiments, args.indir)
    plot_heatmap(experiments, args.indir)
    save_best_params(experiments, args.indir)

    print("All plots generated.")

if __name__ == "__main__":
    main()
