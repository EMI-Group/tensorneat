import gym

env = gym.make("CartPole-v1", new_step_api=True)
print(env.reset())
obs = env.reset()

print(obs)
while True:
    action = env.action_space.sample()
    obs, reward, terminate, truncate, info = env.step(action)
    print(obs, info)
    if terminate | truncate:
        break

