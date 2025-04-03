import random
import time
from collections import defaultdict
from tqdm import tqdm

from SimpleCliffEnv import SimpleCliffEnv


def run_rollout(env, actions, first_action=None):
    trace = []
    s = env.reset()
    done = False
    
    # By doing this, we can determine what the first action of the agent is.
    if first_action is not None:
        a = first_action
        s_next, r, done = env.step(a)
        trace.append([s, a, r])
        s = s_next
        
    
    while not done:
        # env.show()
        # time.sleep(0.1)
        a = random.choice(actions)
        s_next, r, done = env.step(a)
        trace.append([s, a, r])
        s = s_next
    # env.show()
    return trace


def run_montecarlo(env, gamma, num_of_episodes):
    print(f"Initial state: {env.initial_state}")
    actions = env.action_space
    v_initial_state = []
    q_values = defaultdict(list)
    for a0 in actions:
        for episode in tqdm(range(num_of_episodes), desc=f"Episodes with action {a0}", unit="episode"):
            trace = run_rollout(env, actions, first_action=a0)

            # Computing the return at the initial state
            g = 0.0
            for s, a, r in trace[::-1]:
                g = r + gamma * g
            v_initial_state.append(g)
            q_values[a0].append(g)
            

            # Showing current estimate
            # if episode % 10000 == 0:
            #     print(f"Ep. {episode}. Current return: {g:0.3f}. Avg return: {sum(v_initial_state)/len(v_initial_state):0.3f}")
                # Here the average return is the estimated value function of the initial state.
    print(f"Estimated value function for initial state {env.initial_state}: {sum(v_initial_state)/len(v_initial_state):0.3f}")
    for a in actions:
        print(f"Estimated action value for Action {a}: {sum(q_values[a])/len(q_values[a]):0.3f}")
            


if __name__ == '__main__':
    num_of_episodes = 500001
    gamma = 0.99
    epsilon = 0.1
    env = SimpleCliffEnv((3, 2))
    run_montecarlo(env, gamma, num_of_episodes)
    
