import env_blackjack
import numpy as np
import matplotlib.pyplot as plt

env = env_blackjack.BlackjackEnv()
V = np.zeros((2, 10, 10))        # empirical V values
gamma = 1       # discount factor
alpha = 0.01    # learning rate

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
    while state[0]<12:
       state, _, done, _ = env.step(1)

    while done == False:
        action = policy(state)
        next_state, reward, done, _ = env.step(action)
        V[state_idx(state)] += alpha*(reward + gamma*V[state_idx(next_state)]-V[state_idx(state)])
        state = next_state
    winning_counter += (reward>0)*1.0

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