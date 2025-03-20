from BanditEnv import BanditEnv
from BanditResults import BanditResults
from agents.RandomAgent import RandomAgent
from agents.EpsilonGreedyAgent import EpsilonGreedyAgent 
from agents.GradientBanditAgent import GradientBanditAgent
import matplotlib.pyplot as plt
import numpy as np
import os
import csv
from tqdm import tqdm


def show_results(bandit_results: type(BanditResults)) -> None:
    print("\nAverage results")
    print("Step\tReward\tOptimal action (%)")
    average_rewards = bandit_results.get_average_rewards()
    optimal_action_percentage = bandit_results.get_optimal_action_percentage()
    for step in range(NUM_OF_STEPS):
        print(f"{step+1}\t{average_rewards[step]:0.3f}\t{optimal_action_percentage[step]:0.3f}")
   
def write_results(bandit_results: type(BanditResults), filename: str) -> None: 
    avg_rewards = bandit_results.get_average_rewards()
    optimal_action_optimistica = bandit_results.get_optimal_action_percentage()
    
    file_exists = os.path.isfile(filename)
    
    with open(filename, mode='a', newline='') as file:
        writer = csv.writer(file)

        if not file_exists:
            writer.writerow(["Step", "Reward", "Optimal action (%)"])
        
        for step in range(NUM_OF_STEPS):
            writer.writerow([step+1, avg_rewards[step], optimal_action_percentage[step]])

def plot_results(resultado_01,resultado_04,resultado_01_no,resultado_04_no):

    optimal_action_01 = resultado_01.get_optimal_action_percentage()
    optimal_action_04  = resultado_04.get_optimal_action_percentage()
    optimal_action_01_no = resultado_01_no.get_optimal_action_percentage()
    optimal_action_04_no  = resultado_04_no.get_optimal_action_percentage()

    plt.figure(figsize=(10, 5))

    # Graficar porcentaje de acciones óptimas
    plt.plot(optimal_action_01, label=r"$\alpha$ = 0.1 con baseline", linestyle="dashed", color="blue")
    plt.plot(optimal_action_04, label=r"$\alpha$ = 0.4 con baseline", linestyle="dashed", color="brown")
    plt.plot(optimal_action_01_no, label=r"$\alpha$ = 0.1 sin baseline", linestyle="dashed", color="lightblue")
    plt.plot(optimal_action_04_no, label=r"$\alpha$ = 0.4 sin baseline", linestyle="dashed", color="orange")

    plt.xlabel("Pasos")
    plt.ylabel("% Accion optima")
    plt.title("% de Acciones Óptimas por Paso")
    plt.legend()
    plt.show()



if __name__ == "__main__":

    NUM_OF_RUNS = 2000
    NUM_OF_STEPS = 1000
    
    #alpha 0.1 baseline
    resultado_01 = BanditResults()
    for run_id in tqdm(range(NUM_OF_RUNS)):
        bandit = BanditEnv(seed=run_id, mean=4.0)
        num_of_arms = bandit.action_space
        agent = GradientBanditAgent(num_of_arms, alpha=0.1)  # here you might change the agent that you want to use
        best_action = bandit.best_action
        for _ in range(NUM_OF_STEPS):
            action = agent.get_action()
            reward = bandit.step(action)
            agent.learn(action, reward)
            is_best_action = action == best_action
            resultado_01.add_result(reward, is_best_action)
        resultado_01.save_current_run()
    #alpha 0.4 baseline
    resultado_04 = BanditResults()
    for run_id in range(NUM_OF_RUNS):
        bandit = BanditEnv(seed=run_id, mean=4.0)
        num_of_arms = bandit.action_space
        agent = GradientBanditAgent(num_of_arms, alpha=0.4)  # here you might change the agent that you want to use
        best_action = bandit.best_action
        for _ in range(NUM_OF_STEPS):
            action = agent.get_action()
            reward = bandit.step(action)
            agent.learn(action, reward)
            is_best_action = action == best_action
            resultado_04.add_result(reward, is_best_action)
        resultado_04.save_current_run()
    #alpha 0.1 no baseline
    resultado_01_no = BanditResults()
    for run_id in range(NUM_OF_RUNS):
        bandit = BanditEnv(seed=run_id, mean=4.0)
        num_of_arms = bandit.action_space
        agent = GradientBanditAgent(num_of_arms, alpha=0.1)  # here you might change the agent that you want to use
        agent.baseline = False
        best_action = bandit.best_action
        for _ in range(NUM_OF_STEPS):
            action = agent.get_action()
            reward = bandit.step(action)
            agent.learn(action, reward)
            is_best_action = action == best_action
            resultado_01_no.add_result(reward, is_best_action)
        resultado_01_no.save_current_run()
    #alpha 0.4 no baseline
    resultado_04_no = BanditResults()
    for run_id in range(NUM_OF_RUNS):
        bandit = BanditEnv(seed=run_id, mean=4.0)
        num_of_arms = bandit.action_space
        agent = GradientBanditAgent(num_of_arms, alpha=0.4)  # here you might change the agent that you want to use
        agent.baseline = False
        best_action = bandit.best_action
        for _ in range(NUM_OF_STEPS):
            action = agent.get_action()
            reward = bandit.step(action)
            agent.learn(action, reward)
            is_best_action = action == best_action
            resultado_04_no.add_result(reward, is_best_action)
        resultado_04_no.save_current_run()

 #   show_results(results)
#    write_results(results, "results.csv")
    plot_results(resultado_01,resultado_04,resultado_01_no,resultado_04_no)
