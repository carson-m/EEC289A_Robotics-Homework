# Stage 2 训练改进记录

## 目标

Stage 1 训练出只会前进的机器狗。Stage 2 在此基础上引入后退（vx < 0）、横向（vy ≠ 0）和原地旋转（yaw_rate ≠ 0）指令，目标是让机器狗能够完整响应三轴速度指令。

---

## 修改列表

### 1. Alpha Scheduler：基于 Episode Progress 的 Sine 调度

**文件：** `go2_pg_env/joystick.py` — `_student_stage2_sampling_profile()`

**问题：** 原始代码直接将传入的 `episode_progress` 用作 `alpha`。由于 Brax 的 AutoReset 机制在每个 episode 结束时调用 `env.reset()`，`training_step` 会重置为 0，导致 `alpha` 在每个 episode 内最多只能到约 `episode_length / training_progress_steps ≈ 0.07`，curriculum 实际上从不展开。

**修改：**
1. 将 `training_step / training_progress_steps` 乘以 `training_progress_steps / episode_length`，使 `alpha` 在每个 episode 内从 0 线性增长到 1，不受跨 episode 计数器重置的影响。
2. 在线性 alpha 之上叠加 sine scheduler：`sin(clip(3α, 0, 1) × π/2)`。该调度在 episode 前 1/3 内平滑地将 alpha 推至最大值 1，然后保持不变，相比纯线性调度收敛更快且过渡更平滑。

```python
# 线性化为 0→1 的 episode 内进度
if self._training_progress_steps > 0:
    alpha = jp.clip(
        progress * self._training_progress_steps / float(self._config.episode_length),
        0.0, 1.0,
    )
else:
    alpha = jp.clip(progress, 0.0, 1.0)

# Sine 调度：前 1/3 episode 内平滑升至 1
alpha = jp.sin(jp.clip(3 * alpha, 0.0, 1.0) * jp.pi / 2)
```

**效果：** Command 范围在每个 episode 的前 1/3 即完成从 stage 1 起点到 stage 2 目标的完整过渡，机器狗在每次 episode 的大部分时间都会接触到后退、横向和旋转指令。

---

### 2. 增加 Stage 2 训练步数

**文件：** `configs/course_config.json`

| 字段 | 修改前 | 修改后 |
|------|--------|--------|
| `stage_2.num_timesteps` | 5,000,000 | 15,000,000 |

**理由：** Stage 2 需要同时学习四种新运动（后退、左横、右横、旋转），5M 步不足以充分训练。15M 步与 stage 1 总步数保持一致，给予新技能充足的学习时间。

> **注意：** notebook 中的 `runtime_overrides` 包含 `"stage_2_num_timesteps": 5_000_000`，该值优先级高于 JSON 配置。需将 notebook 中对应行改为 `15_000_000`，否则 JSON 里的 15M 设置将被忽略。

---

### 3. 提高 vy 和 yaw_rate 的 Keep Probability

**文件：** `configs/course_config.json` — `stage_2.student_stage2_goal.command_keep_prob`

| 维度 | 修改前 | 修改后 |
|------|--------|--------|
| vx | 1.0 | 1.0 |
| vy | 0.35 | 0.6 |
| yaw_rate | 0.35 | 0.6 |

**理由：** 原始 0.35 的概率结合 `sample_command` 中 50% 的 `blend_mask`，vy/yaw_rate 被实际激活的有效概率仅约 17.5%，机器狗大部分时间仍在执行前进指令。提升至 0.6 后有效激活概率约为 30%，新动作训练曝光量近乎翻倍。

---

### 4. 调整 Reward Clipping 下界

**文件：** `go2_pg_env/joystick.py` — `step()`

| 修改前 | 修改后 |
|--------|--------|
| `jp.clip(..., 0.0, 10000.0)` | `jp.clip(..., -100, 10000.0)` |

**理由：** 原来下界为 0，所有惩罚项（`termination`、`orientation`、`stand_still` 等）在累加后会被裁掉，机器狗收不到有效的负反馈信号。调整为 -100 后，机器狗在倾倒或做出危险动作时能接收到真实的惩罚梯度，有助于抑制"倾斜不移动"等 reward hacking 行为。

---

### 5. 增强 Stage 2 姿态约束

**文件：** `configs/course_config.json` — `stage_2.reward_scales`

新增：
```json
"orientation": -10.0
```

**理由：** 默认 `orientation=-5.0` 对机身倾斜的惩罚力度不够。机器狗面对横向或后退指令时，倾斜身体比学习新步态的代价更低，会以倾斜代替真正的运动。将 stage 2 的 orientation 惩罚加倍至 -10.0，直接打破倾斜即得分的局面，迫使机器狗学习正确的步态。

---

### 6. 减小 tracking_sigma

**文件：** `go2_pg_env/joystick.py` — `default_config()`

| 字段 | 修改前 | 修改后 |
|------|--------|--------|
| `reward_config.tracking_sigma` | 0.25 | 0.1 |

**理由：** Tracking 奖励为 `exp(-||cmd[:2] - vel[:2]||² / sigma)`。sigma=0.25 时，vy=0.2 的指令下机器狗完全不动仍能获得约 85% 的满分奖励，缺乏真正移动的动力。sigma 降至 0.1 后，同样不动只能得到约 67% 的奖励，差距扩大，迫使机器狗真正追踪目标速度。

---

## 修改文件汇总

| 文件 | 修改内容 |
|------|----------|
| `go2_pg_env/joystick.py` | Alpha scheduler（episode 内归一化 + sine 调度）、reward clip 下界 -100、tracking_sigma 0.1 |
| `configs/course_config.json` | stage_2 num_timesteps 15M、goal keep_prob [1,0.6,0.6]、stage_2 orientation -10.0 |
| `configs/colab_runtime_config.json` | goal keep_prob [1,0.6,0.6]（由 notebook 从 course_config 同步覆盖） |
