import polars as pl
import json


CONFIG_PATH = f"../Data/DDPG_Trials/ddpg_mountain_car_configs.json"
RESULT_PATH = f"../Data/DDPG_Trials/ddpg_results"

# Load configs
def get_configs(json_path):
    with open(json_path) as f:
        configs = json.load(f)
    assert len(configs) > 0, "No configurations found in the JSON file."
    return configs
    
    
def get_num_skip_lines(file_path):
    """
    Returns the number of lines to skip in the CSV file.
    This is used to skip the header lines that start with '#'.
    """
    skip_lines = 0
    with open(file_path) as f:
        for line in f:
            if line.startswith("#"):
                skip_lines += 1
            else:
                break
    return skip_lines


def compare_config_results(json_config_summary_path):
    configs = get_configs(json_config_summary_path)
    results = []
    for i, config in enumerate(configs):
        file_path = f"{RESULT_PATH}_{i}.monitor.csv"
        try:
            num_skip_lines = get_num_skip_lines(file_path)
            df = pl.read_csv(file_path, skip_rows=num_skip_lines)
            
        except FileNotFoundError:
            print(f"File {file_path} not found. Skipping config {i}.")
            continue
        
        mean_reward = df["r"].mean()
        last_100_mean = df["r"].tail(100).mean()
        results.append({**config, "mean_reward": mean_reward, "last_100_mean": last_100_mean, "index": i})

    results_df = pl.DataFrame(results)
    results_df = results_df.sort("last_100_mean", descending=True)
    print(results_df)
    
if __name__ == "__main__":
    compare_config_results(CONFIG_PATH)

# results_df.write_csv("../Data/dqn_trial_results_summary.csv")
# print(results_df.sort_values("last_100_mean", ascending=False).head())

