import seaborn as sns
import pandas as pd
from matplotlib import pyplot as plt


if __name__ == "__main__":
    filename = "results.csv" 
    df = pd.read_csv(filename) 
    sns.lineplot(x="Step", y="Reward", data=df)
    # sns.lineplot(x="Step", y="Optimal action (%)", data=df)
    plt.show()

    
