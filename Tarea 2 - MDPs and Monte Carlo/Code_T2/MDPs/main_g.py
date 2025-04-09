import random
import numpy as np
import time
from Problems.CookieProblem import CookieProblem
from Problems.GridProblem import GridProblem
from Problems.GamblerProblem import GamblerProblem


def get_action_from_user(actions):
    print("Valid actions:")
    for i in range(len(actions)):
        print(f"{i}. {actions[i]}")
    print("Please select an action:")
    selected_id = -1
    while not (0 <= selected_id < len(actions)):
        selected_id = int(input())
    return actions[selected_id]


def sample_transition(transitions):
    probs = [prob for prob, _, _ in transitions]
    transition = random.choices(population=transitions, weights=probs)[0]
    prob, s_next, reward = transition
    return s_next, reward


def play(problem):
    state = problem.get_initial_state()
    done = False
    total_reward = 0.0
    while not done:
        problem.show(state)
        actions = problem.get_available_actions(state)
        action = get_action_from_user(actions)
        transitions = problem.get_transitions(state, action)
        s_next, reward = sample_transition(transitions)
        done = problem.is_terminal(s_next)
        state = s_next
        total_reward += reward
    print("Done.")
    print(f"Total reward: {total_reward}")


def play_gambler_problem():
    p = 0.4
    problem = GamblerProblem(p)
    play(problem)


def play_grid_problem():
    size = 4
    problem = GridProblem(size)
    play(problem)


def play_cookie_problem():
    size = 3
    problem = CookieProblem(size)
    play(problem)




def iterative_policy_evaluation(problem, gamma):
    V = {s: 0 for s in problem.states}
    theta = 1e-10
    while True:
        delta = 0
        for s in problem.states:
            if problem.is_terminal(s):
                continue  # Mantener V(s) = 0 para estados terminales
            
            v_old = V[s]
            total = 0
            
            available_actions = problem.get_available_actions(s)

            pi = 1.0 / len(available_actions) #if available_actions else 0
            
            for a in available_actions:
                transitions = problem.get_transitions(s, a)
                for prob, s_next, r in transitions:
                    total += pi * prob * (r + gamma * V[s_next])
            
            V[s] = total
            delta = max(delta, abs(v_old - V[s]))
        
        if delta < theta:
            break
    
    return V

def evaluate_policy(problem, policy, gamma):
    V = {s: 0 for s in problem.states}
    theta=1e-10
    while True:
        delta = 0
        for s in problem.states:
            if problem.is_terminal(s):
                continue
            
            v_old = V[s]
            a = policy[s]
            total = 0
            transitions = problem.get_transitions(s, a)
            for prob, s_next, r in transitions:
                total += prob * (r + gamma * V[s_next])
            
            V[s] = total
            delta = max(delta, abs(v_old - V[s]))
        
        if delta < theta:
            break
    return V

def greedy_policy(problem, V, gamma):
    policy = {}
    for s in problem.states:
        if problem.is_terminal(s):
            policy[s] = None
            continue
        
        best_action = None
        max_value = -float('inf')
        
        for a in problem.get_available_actions(s):
            total = 0
            transitions = problem.get_transitions(s, a)
            for prob, s_next, r in transitions:
                total += prob * (r + gamma * V[s_next])
            
            if total > max_value:
                max_value = total
                best_action = a
        
        policy[s] = best_action
    
    return policy

if __name__ == '__main__':
    # play_grid_problem()
    #play_cookie_problem()
    # play_gambler_problem()
    for size in range(3, 11):
      problem = GridProblem(size)
      V_random = iterative_policy_evaluation(problem, gamma=1.0)
      initial_state = problem.get_initial_state()
      greedy = greedy_policy(problem, V_random, gamma=1.0)
      V_greedy = evaluate_policy(problem, greedy, gamma=1.0)
      print(f"GridProblem {size}x{size}: =  V_greedy(s0) ={V_greedy[initial_state]:.3f}")

    
    for size in range(3, 11):
      problem = CookieProblem(size)
      V_random = iterative_policy_evaluation(problem, gamma=0.99)
      initial_state = problem.get_initial_state()
      greedy = greedy_policy(problem, V_random, gamma=0.99)
      V_greedy = evaluate_policy(problem, greedy, gamma=0.99)
      print(f"CookieProblem {size}x{size}: V_greedy(s0) = {V_greedy[initial_state]:.3f}")
    
    for ph in [0.25, 0.4, 0.55]:
      problem = GamblerProblem(ph)
      V_random = iterative_policy_evaluation(problem, gamma=1.0)
      initial_state = problem.get_initial_state()
      greedy = greedy_policy(problem, V_random, gamma=1.0)
      V_greedy = evaluate_policy(problem, greedy, gamma=1.0)
      print(f"GamblerProblem ph={ph}: V_greedy(s0) = {V_greedy[initial_state]:.3f}")