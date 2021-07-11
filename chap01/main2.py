"""
OpenAI GymでDQN
"""

import gym
import matplotlib.pyplot as plt
import numpy as np
from rl.agents.dqn import DQNAgent
from rl.memory import SequentialMemory
from rl.policy import BoltzmannQPolicy
from tensorflow.keras.layers import Dense, Flatten, Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

episode_num = 100
max_step_num = 10000
window_length = 1
memory_size = 50000
ENV_NAME = "CartPole-v0"


env = gym.make(ENV_NAME)
env.seed(123)
np.random.seed(123)

input_shape = (window_length,) + env.observation_space.shape
nb_inputs = input_shape[1]
nb_actions = env.action_space.n
nb_dense1 = nb_inputs * 10
nb_dense3 = nb_actions * 10
nb_dense2 = int(np.sqrt(nb_dense1 * nb_dense3))


print("入力の次元", input_shape)
print("出力の次元", nb_actions)
print("報酬の範囲", env.reward_range)

# モデル作成
l_input = Input(input_shape)
l_flatten = Flatten()(l_input)
l_dense1 = Dense(nb_dense1, activation="relu")(l_flatten)
l_dense2 = Dense(nb_dense2, activation="relu")(l_dense1)
l_dense3 = Dense(nb_dense3, activation="relu")(l_dense2)
l_output = Dense(nb_actions, activation="linear")(l_dense3)
model = Model(l_input, l_output)
print(model.summary())
print("======")

# Agent
memory = SequentialMemory(limit=memory_size, window_length=window_length)
policy = BoltzmannQPolicy()
agent = DQNAgent(
    model=model, nb_actions=nb_actions, memory=memory, nb_steps_warmup=10, target_model_update=1e-2, policy=policy
)
agent.compile(Adam(learning_rate=1e-3), metrics=["mae"])

# 学習
history = agent.fit(env, nb_steps=max_step_num, visualize=False, verbose=0)

# テスト
agent.test(env, nb_episodes=episode_num, visualize=False)

# モデル保存
agent.save_weights(f"{ENV_NAME}_weight.h5", overwrite=True)

# 結果描画
plt.subplot(2, 1, 1)
plt.plot(history.history["nb_episode_steps"])
plt.ylabel("step")

plt.subplot(2, 1, 2)
plt.plot(history.history["episode_reward"])
plt.xlabel("episode")
plt.ylabel("reward")

plt.savefig("./step_reward.png")
