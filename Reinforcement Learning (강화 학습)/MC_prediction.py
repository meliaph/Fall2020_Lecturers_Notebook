import env_blackjack
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

env = env_blackjack.BlackjackEnv()
V = np.zeros((2, 10, 10))        # empirical V values
N = np.zeros((2, 10, 10))        # number of visits
gamma = 1       # discount factor

num_episodes = 500000

def state_idx(state):
    return state[2]*1, state[0]-12, state[1]-1

def policy(state):
    action = 1
    if state[0]>19:
        action = 0
    return action

winning_counter = 0
for epi in range(num_episodes):
    state = env.reset()
    done = False
    state_history = []
    reward_history = []
    while state[0]<12:
       state, _, done, _ = env.step(1)

    while done == False:
        action = policy(state)
        next_state, reward, done, _ = env.step(action)
        state_history.append(state)
        reward_history.append(reward)
        state = next_state
    winning_counter += (reward>0)*1.0

    T = len(state_history)
    for t in range(T):
        G_t = np.sum(np.array(reward_history[t:T]) * gamma**np.arange(T-t) )
        index = state_idx(state_history[t])
        N[index] += 1
        V[index] += 1/N[index] * (G_t-V[index])
    if (epi+1) % 10000 == 0:
        print("Episode: %6d, Winning rate: %.2f"%(epi+1, winning_counter/(epi+1)))

## Plot Result
X = np.arange(1, 11)
Y = np.arange(12, 22)
X, Y = np.meshgrid(X, Y)

fig = plt.figure()

ax0 = fig.add_subplot(211, projection='3d')
ax0.plot_surface(X, Y, V[0], rstride=1, cstride=1, cmap='coolwarm')
ax0.set_xlabel('Dealer')
ax0.set_ylabel('Player sum')
ax0.set_zlabel('V value')
ax0.set_title('No usable ace')

ax1 = fig.add_subplot(212, projection='3d')
ax1.plot_surface(X, Y, V[1], rstride=1, cstride=1, cmap='coolwarm')
ax1.set_xlabel('Dealer')
ax1.set_ylabel('Player sum')
ax1.set_zlabel('V value')
ax1.set_title('Usable ace')
plt.show()