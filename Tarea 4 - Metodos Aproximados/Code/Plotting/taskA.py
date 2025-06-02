import plotly.express as px
import numpy as np
import polars as pl
import os

WIDTH = 900
HEIGHT = WIDTH / 1.5

df_sarsa = pl.read_csv("../../Data/taskA_sarsa_results.csv")
df_q_learning = pl.read_csv("../../Data/taskA_q_learning_results.csv")
header_name = "Episode Number"


def prepare_df(df, header_name: str, alg_name: str):

    df = df.transpose(
        include_header=True,
        header_name=header_name,
        column_names=[f"Run {i}" for i in range(1, 31)],
    )
    df = df.with_columns(pl.mean_horizontal(df.columns[1:]).alias("Episode Mean"))
    df = df.with_columns(pl.lit(alg_name).alias("Algorithm"))
    df = df.with_columns(
        pl.col(header_name).cast(pl.Int16),
    )
    df = df.filter(pl.col(header_name) % 10 == 0)
    return df


df_sarsa = prepare_df(df_sarsa, header_name, "Sarsa")
df_q_learning = prepare_df(df_q_learning, header_name, "Q-Learning")
df_all = pl.concat([df_sarsa, df_q_learning], how="vertical")
print(df_all)

fig = px.line(
    df_all,
    x=header_name,
    y="Episode Mean",
    color="Algorithm",
    title="Sarsa vs. Q-Learning Agent Performance",
    labels={
        header_name: "Episode Number",
        "Episode Mean": "Average Episode Reward",
        "Algorithm": "Algorithm",
    },
    width=WIDTH,
    height=HEIGHT,
)
fig.update_layout(template="plotly_white")
fig.show()
