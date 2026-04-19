import numpy as np
import random

# ==========================================
# 1. 环境与参数初始化
# ==========================================
# 学习参数 gamma (趋向于1表示考虑未来的奖励，文档设定为 0.8)
GAMMA = 0.8
# 目标房间编号
GOAL_STATE = 5
# 房间总数 (0到5)
NUM_STATES = 6
# 训练的 episode 数量
NUM_EPISODES = 1000

# 奖励矩阵 R (-1 表示节点之间没有直接连通的门)
# 根据文档中 图6 的 reward 值矩阵进行构建
R = np.array([
    [-1, -1, -1, -1, 0, -1],  # 状态 0
    [-1, -1, -1, 0, -1, 100],  # 状态 1
    [-1, -1, -1, 0, -1, -1],  # 状态 2
    [-1, 0, 0, -1, 0, -1],  # 状态 3
    [0, -1, -1, 0, -1, 100],  # 状态 4
    [-1, 0, -1, -1, 0, 100]  # 状态 5 (目标状态自己指向自己，奖励为100)
])

# 将 Q 矩阵初始化为全零矩阵
Q = np.zeros((NUM_STATES, NUM_STATES))


# 辅助函数：获取当前状态下所有可能的行为（即下一步可以去的房间）
def get_available_actions(state):
    # 在矩阵 R 中，当前行里值大于等于 0 的列对应的就是可通行的房间
    current_state_row = R[state,]
    available_actions = np.where(current_state_row >= 0)[0]
    return available_actions


# ==========================================
# 2. Q-learning 算法训练主体
# ==========================================
for episode in range(NUM_EPISODES):
    # Step 3.1 随机选择一个初始的状态
    current_state = random.randint(0, NUM_STATES - 1)

    # Step 3.2 若未达到目标状态，则不断执行探索
    while current_state != GOAL_STATE:

        # (1) 在当前状态的所有可能行为中随机选取一个行为
        available_actions = get_available_actions(current_state)
        action = random.choice(available_actions)

        # (2) 利用选定的行为，得到下一个状态
        next_state = action

        # 找出下一个状态下所有可能行为对应的最大 Q 值
        next_available_actions = get_available_actions(next_state)
        max_q = 0
        if len(next_available_actions) > 0:
            max_q = np.max(Q[next_state, next_available_actions])

        # (3) 按照核心公式计算并更新 Q(s,a)
        # 公式: Q(s,a) = R(s,a) + gamma * max(Q(s', a'))
        Q[current_state, action] = R[current_state, action] + GAMMA * max_q

        # (4) 令当前状态等于下一个状态，继续循环
        current_state = next_state

# ==========================================
# 3. 测试与输出结果
# ==========================================
print("--- 训练完成后的 Q 矩阵 (未规范化) ---")
print(np.round(Q, 1))

# 根据规范化逻辑：每个非零元素都除以矩阵Q的最大元素 (乘以100转为文档展示的百分比形式)
max_q_value = np.max(Q)
if max_q_value > 0:
    Q_normalized = (Q / max_q_value * 100).astype(int)
    print("\n--- 规范化后的 Q 矩阵 (类似图14) ---")
    print(Q_normalized)


def get_optimal_path(start_state):
    path = [start_state]
    current_state = start_state

    # 重复执行直到成为目标状态
    while current_state != GOAL_STATE:
        available_actions = get_available_actions(current_state)
        if len(available_actions) == 0:
            break
        # 确定满足最大 Q 值的下一步行动
        best_action = available_actions[np.argmax(Q[current_state, available_actions])]
        path.append(best_action)
        # 转移状态
        current_state = best_action

    return path


# 测试：从文档结尾的例子，即房间 2 出发
start_room = 2
optimal_path = get_optimal_path(start_room)
print(f"\n--- 从房间 {start_room} 到达目标房间的最佳路径 ---")
print(" -> ".join(map(str, optimal_path)))  # 期望输出: 2 -> 3 -> 1 -> 5 或 2 -> 3 -> 4 -> 5