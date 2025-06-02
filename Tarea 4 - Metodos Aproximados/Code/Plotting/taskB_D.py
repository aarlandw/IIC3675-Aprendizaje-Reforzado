import plotly.express as px
import numpy as np
import polars as pl

def get_num_skip_lines(file_path):
    skip_lines = 0
    with open(file_path) as f:
        for line in f:
            if line.startswith("#"):
                skip_lines += 1
            else:
                break
    return skip_lines

def load_and_aggregate_runs(
    file_pattern,
    num_runs=30,
    max_episodes=1500,
    reward_col_name="r"
):
    run_data = []
    for i in range(num_runs):
        file_path = file_pattern.format(i)
        skip = get_num_skip_lines(file_path)
        df_i = pl.read_csv(file_path, skip_rows=skip)
        values = df_i[reward_col_name].to_numpy()
        # Truncate or pad to max_episodes
        if len(values) < max_episodes:
            values = np.pad(values, (0, max_episodes - len(values)), constant_values=0)
        else:
            values = values[:max_episodes]
        run_data.append(values)
    # Build DataFrame
    data = {f"Run_{i+1}": run_data[i] for i in range(num_runs)}
    data["Episode"] = np.arange(1, max_episodes + 1)
    df = pl.DataFrame(data).select(["Episode"] + [f"Run_{i+1}" for i in range(num_runs)])
    # Add mean column
    df = df.with_columns(
        pl.mean_horizontal([f"Run_{i+1}" for i in range(num_runs)]).alias("Episode Mean")
    )
    return df

def plot_episode_means(
    df,
    algo_name="Algorithm",
    env_name="Environment",
    y_label="Average Episode Reward",
    every_n=10,
    width=900,
    height=600
):
    df_plot = df.filter(pl.col("Episode") % every_n == 0)
    fig = px.line(
        df_plot,
        x="Episode",
        y="Episode Mean",
        title=f"{algo_name} Average Episode Reward in {env_name} Environment",
        labels={"Episode": "Episode", "Episode Mean": y_label},
        width=width,
        height=height,
    )
    fig.update_layout(template="plotly_white")
    fig.show()

if __name__ == "__main__":
    # For DQN
    dqn_df = load_and_aggregate_runs(
        file_pattern="../../Data/DQN/dqn_results_{}.monitor.csv",
        num_runs=30,
        max_episodes=1500,
        reward_col_name="r"  # or "l" for episode length
    )
    plot_episode_means(dqn_df, algo_name="DQN")

    # For DDPG (example, adjust file_pattern and value_col as needed)
    ddpg_df = load_and_aggregate_runs(
        file_pattern="../../Data/DDPG/ddpg_results_{}.monitor.csv",
        num_runs=10,
        max_episodes=1000,
        reward_col_name="r"
    )
    plot_episode_means(ddpg_df, algo_name="DDPG", env_name="MountainCarContinuous-v0")