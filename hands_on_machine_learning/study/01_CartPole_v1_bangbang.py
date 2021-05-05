import gym, numpy as np, sys, time

def basic_policy(obs):
    angle = obs[2]
    return 0 if angle < 0 else 1

env = gym.make("CartPole-v1")
obs = env.reset()
start = time.time()
env.render()
totals = []
iter = 500
for episode in range(iter):
    sys.stdout.write('\r%.1f%s' %((episode+1)/iter*100,'%' if episode < iter - 1 else '%\n'))
    episode_rewards = 0
    obs = env.reset()
    for step in range(200):
        action = basic_policy(obs)
        obs, reward, done, info = env.step(action)
        episode_rewards += reward

        if done:
            break
    totals.append(episode_rewards)
print('elapsed time is %.1fs' %(time.time() - start))
print(np.mean(totals), np.min(totals), np.max(totals))

