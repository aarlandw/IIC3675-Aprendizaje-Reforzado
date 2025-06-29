import numpy as np

from BairdEnv import BairdEnv
from Policies.BehaviorPolicy import BehaviorPolicy
from Policies.TargetPolicy import TargetPolicy


def run_policy_evaluation():
    num_of_steps = 1000
    gamma = 0.99
    alpha = 0.05

    env = BairdEnv()
    weights = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 10.0, 1.0])
    target_policy = TargetPolicy()
    behavior_policy = BehaviorPolicy()

    observation = env.reset()
    for step in range(num_of_steps):
        action = behavior_policy.sample(observation)
        next_observation, reward, _ = env.step(action)

        # TODO: Completa la regla de aprendizaje de off-policy evaluation con importance sampling, TD(0) y aproximación lineal
        # ----------------------------        
        td_error = 0.0
        p_target = target_policy.get_probability(observation, action)
        p_behavior = behavior_policy.get_probability(observation, action)
        # If this was On-Policy, p_target == p_behavior
        v_obs = np.dot(weights, observation)
        grad_v = observation
        v_next_obs = np.dot(weights, next_observation)
        td_error = reward + gamma * v_next_obs - v_obs
        weights += alpha * td_error * grad_v

        # ----------------------------

        print(f"Step {step+1}.\tTD Error {td_error: 0.2f}.\tW8 {weights[-1]: 0.2f}")

        observation = next_observation


if __name__ == '__main__':
    run_policy_evaluation()
