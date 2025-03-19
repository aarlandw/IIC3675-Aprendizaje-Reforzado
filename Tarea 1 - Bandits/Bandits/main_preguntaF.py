from BanditEnv import BanditEnv
from BanditResults import BanditResults
from agents.RandomAgent import RandomAgent
from agents.EpsilonGreedyAgent import EpsilonGreedyAgent 
from agents.FixedStepSizeAgent import FixedStepSizeAgent
import matplotlib.pyplot as plt
import numpy as np
import os
import csv


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

def plot_results(bandit_results_optimista, bandit_results_realista):
    average_rewards_optimista = bandit_results_optimista.get_average_rewards()
    average_rewards_realista = bandit_results_realista.get_average_rewards()
    optimal_action_optimista = bandit_results_optimista.get_optimal_action_percentage()
    optimal_action_realista  = bandit_results_realista.get_optimal_action_percentage()

    plt.figure(figsize=(10, 5))

    # Graficar porcentaje de acciones óptimas
    plt.plot(optimal_action_optimista, label="Optimista Q1=5 epsilon = 0", linestyle="dashed", color="blue")
    plt.plot(optimal_action_realista, label=" Realista Q1=0 epsilon = 0.1", linestyle="dashed", color="gray")

    plt.xlabel("Pasos")
    plt.ylabel("% Accion optima")
    plt.title("% de Acciones Óptimas por Paso")
    plt.legend()
    plt.show()



if __name__ == "__main__":

    NUM_OF_RUNS = 2000
    NUM_OF_STEPS = 1000
    
    #inicicalizacion  optimistica
    resultado_optimistico = BanditResults()
    for run_id in range(NUM_OF_RUNS):
        bandit = BanditEnv(seed=run_id)
        num_of_arms = bandit.action_space
        agent = FixedStepSizeAgent(num_of_arms, alpha=0.1, epsilon=0)  # here you might change the agent that you want to use
        agent.q_values = np.full(num_of_arms, 5.0)
        best_action = bandit.best_action
        for _ in range(NUM_OF_STEPS):
            action = agent.get_action()
            reward = bandit.step(action)
            agent.learn(action, reward)
            is_best_action = action == best_action
            resultado_optimistico.add_result(reward, is_best_action)
        resultado_optimistico.save_current_run()
    # inicializacion realista 
    resultado_realista = BanditResults()
    for run_id in range(NUM_OF_RUNS):
        bandit = BanditEnv(seed=run_id)
        num_of_arms = bandit.action_space
        agent = FixedStepSizeAgent(num_of_arms, alpha=0.1, epsilon=0.1)  # here you might change the agent that you want to use
        agent.q_values = np.zeros(num_of_arms)
        best_action = bandit.best_action
        for _ in range(NUM_OF_STEPS):
            action = agent.get_action()
            reward = bandit.step(action)
            agent.learn(action, reward)
            is_best_action = action == best_action
            resultado_realista.add_result(reward, is_best_action)
        resultado_realista.save_current_run()

 #   show_results(results)
#    write_results(results, "results.csv")
    plot_results(resultado_optimistico,resultado_realista)
