# import polars as pl
# import json


# CONFIG_PATH = f"../Data/DQN_Trials/dqn_mountain_car_configs.json"
# RESULT_PATH = f"../Data/DQN_Trials/dqn_results"


# # Load configs
# def get_configs(json_path):
#     with open(json_path) as f:
#         configs = json.load(f)
#     assert len(configs) > 0, "No configurations found in the JSON file."
#     return configs


# def get_num_skip_lines(file_path):
#     """
#     Returns the number of lines to skip in the CSV file.
#     This is used to skip the header lines that start with '#'.
#     """
#     skip_lines = 0
#     with open(file_path) as f:
#         for line in f:
#             if line.startswith("#"):
#                 skip_lines += 1
#             else:
#                 break
#     return skip_lines


# def compare_config_results(json_config_summary_path):
#     configs = get_configs(json_config_summary_path)
#     results = []
#     for i, config in enumerate(configs):
#         file_path = f"{RESULT_PATH}_{i}.monitor.csv"
#         try:
#             num_skip_lines = get_num_skip_lines(file_path)
#             df = pl.read_csv(file_path, skip_rows=num_skip_lines)

#         except FileNotFoundError:
#             print(f"File {file_path} not found. Skipping config {i}.")
#             continue

#         mean_reward = df["r"].mean()
#         last_100_mean = df["r"].tail(100).mean()
#         results.append(
#             {
#                 **config,
#                 "mean_reward": mean_reward,
#                 "last_100_mean": last_100_mean,
#                 "index": i,
#             }
#         )

#     results_df = pl.DataFrame(results)

#     ### Workaround of the fucking century. CTM!
#     if "net_arch" in results_df.columns:
#         results_df = results_df.with_columns(
#                 "[" + pl.col("net_arch").cast(pl.List(pl.String)).list.join(", ") + "]",
#             )
#         results_df = results_df.drop("net_arch")
#         results_df = results_df.rename({"literal": "net_arch"})

#     results_df = results_df.sort("last_100_mean", descending=True)
#     print(results_df)
#     results_df.write_csv(f"{RESULT_PATH}_summary.csv")


# if __name__ == "__main__":
#     compare_config_results(CONFIG_PATH)

# results_df.write_csv("../Data/dqn_trial_results_summary.csv")
# print(results_df.sort_values("last_100_mean", ascending=False).head())

import polars as pl
import json
import os


def get_configs(json_path):
    with open(json_path) as f:
        configs = json.load(f)
    assert len(configs) > 0, "No configurations found in the JSON file."
    return configs


def get_num_skip_lines(file_path):
    skip_lines = 0
    with open(file_path) as f:
        for line in f:
            if line.startswith("#"):
                skip_lines += 1
            else:
                break
    return skip_lines


def serialize_list_columns(df, columns):
    for col in columns:
        if col in df.columns:
            df = df.with_columns(pl.col(col).map_elements(json.dumps).alias(col))
    return df


def write_list_to_string(df, col: str):
    """Rewrites a column containing lists to a string representation.
    This is useful for when writing dataframes to csv.

    Args:
        df (DataFrame)
        col (str): Column with lists to be converted to strings.

    Returns:
        DataFrame
    """
    df = df.with_columns(
        "[" + pl.col(col).cast(pl.List(pl.String)).list.join(", ") + "]",
    )
    df = df.drop(col)
    df = df.rename({"literal": col})
    return df


def summarize_monitor_results(
    config_path,
    result_path_pattern,
    reward_col="r",
    list_columns=["net_arch"],
    summary_out_path=None,
    num_configs=None,
):
    configs = get_configs(config_path)
    results = []
    if num_configs is None:
        num_configs = len(configs)
    for i, config in enumerate(configs[:num_configs]):
        file_path = result_path_pattern.format(i)
        if not os.path.exists(file_path):
            print(f"File {file_path} not found. Skipping config {i}.")
            continue
        num_skip_lines = get_num_skip_lines(file_path)
        df = pl.read_csv(file_path, skip_rows=num_skip_lines)
        mean_reward = df[reward_col].mean()
        last_100_mean = df[reward_col].tail(100).mean()
        results.append(
            {
                **config,
                "mean_reward": mean_reward,
                "last_100_mean": last_100_mean,
                "index": i,
            }
        )
    results_df = pl.DataFrame(results)
    for col in list_columns:
        if col in results_df.columns:
            results_df = write_list_to_string(results_df, col)
    # results_df = write_list_to_string(results_df, list_columns)
    results_df = results_df.sort("last_100_mean", descending=True)
    print(results_df)
    if summary_out_path:
        results_df.write_csv(summary_out_path)
    return results_df


if __name__ == "__main__":
    # Example for DQN
    DQN_CONFIG_PATH = "../Data/DQN_Trials/dqn_mountain_car_configs.json"
    DQN_RESULT_PATTERN = "../Data/DQN_Trials/dqn_results_{}.monitor.csv"
    DQN_SUMMARY_PATH = "../Data/DQN_Trials/dqn_results_summary.csv"
    summarize_monitor_results(
        config_path=DQN_CONFIG_PATH,
        result_path_pattern=DQN_RESULT_PATTERN,
        reward_col="r",
        list_columns=("net_arch",),
        summary_out_path=DQN_SUMMARY_PATH,
    )

    # Example for DDPG (uncomment and adjust paths as needed)
    DDPG_CONFIG_PATH = "../Data/DDPG_Trials/ddpg_mountain_car_configs.json"
    DDPG_RESULT_PATTERN = "../Data/DDPG_Trials/ddpg_results_{}.monitor.csv"
    DDPG_SUMMARY_PATH = "../Data/DDPG_Trials/ddpg_results_summary.csv"
    summarize_monitor_results(
        config_path=DDPG_CONFIG_PATH,
        result_path_pattern=DDPG_RESULT_PATTERN,
        reward_col="r",
        list_columns=("net_arch",),
        summary_out_path=DDPG_SUMMARY_PATH
    )
