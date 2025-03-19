from BanditEnv import BanditEnv
from BanditResults import BanditResults
from agents.RandomAgent import RandomAgent
from agents.EpsilonGreedyAgent import EpsilonGreedyAgent 
import matplotlib.pyplot as plt
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
    optimal_action_percentage = bandit_results.get_optimal_action_percentage()
    
    file_exists = os.path.isfile(filename)
    
    with open(filename, mode='a', newline='') as file:
        writer = csv.writer(file)

        if not file_exists:
            writer.writerow(["Step", "Reward", "Optimal action (%)"])
        
        for step in range(NUM_OF_STEPS):
            writer.writerow([step+1, avg_rewards[step], optimal_action_percentage[step]])

def plot_results(bandit_results: BanditResults) -> None:
    average_rewards = bandit_results.get_average_rewards()
    optimal_action_percentage = bandit_results.get_optimal_action_percentage()

    plt.figure(figsize=(12, 6))

    # Gráfico de recompensa promedio
    plt.subplot(1, 2, 1)
    plt.plot(average_rewards, label="Recompensa Promedio")
    plt.xlabel("Pasos")
    plt.ylabel("Recompensa Promedio")
    plt.title("Recompensa Promedio por Paso")
    plt.legend()

    # Gráfico de porcentaje de acciones óptimas
    plt.subplot(1, 2, 2)
    plt.plot(optimal_action_percentage, label="% Acción Óptima")
    plt.xlabel("Pasos")
    plt.ylabel("% Acción Óptima")
    plt.title("% de Acciones Óptimas por Paso")
    plt.legend()

    plt.show()


if __name__ == "__main__":

    NUM_OF_RUNS = 2000
    NUM_OF_STEPS = 1000

    results = BanditResults()
    for run_id in range(NUM_OF_RUNS):
        bandit = BanditEnv(seed=run_id)
        num_of_arms = bandit.action_space
        agent = FixedStepSizeAgent(num_of_arms)  # here you might change the agent that you want to use
        best_action = bandit.best_action
        for _ in range(NUM_OF_STEPS):
            action = agent.get_action()
            reward = bandit.step(action)
            agent.learn(action, reward)
            is_best_action = action == best_action
            results.add_result(reward, is_best_action)
        results.save_current_run()

    show_results(results)
    write_results(results, "results.csv")
    plot_results(results)
