"""
OpenGymのテスト
"""

import gym

episode_num = 5
max_step_num = 200
ENV_NAME = "CartPole-v0"

env = gym.make(ENV_NAME)

print("入力の次元", env.observation_space)
print("出力の次元", env.action_space)
print("報酬の範囲", env.reward_range)

for episode in range(episode_num):
    env.reset()  # 初期化
    env.render()  # 描画
    for step in range(max_step_num):
        action = env.action_space.sample()  # 行動をランダムに取得
        observe, reward, done, info = env.step(action)  # 1Step進める
        env.render()

        if done:
            break
