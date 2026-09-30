# HumanGen ICL RoboTwin 闭环测评

支持 reasoner/generator 的 window checkpoint，通过 HTTP 将模型推理与 RoboTwin 仿真分离。
7 个任务按 `evaluation/robotwin/tasks.txt` 的顺序绑定连续端口，默认 8000～8006。

## 配置与文件职责

**所有 Bash 默认配置集中在 `evaluation/robotwin/common.sh`**，同名环境变量可覆盖。
`server_defaults` 仅在模型侧调用，`client_defaults` 仅在仿真侧调用。
修改默认配置时不必在多个入口重复填写路径。

| 文件 | 职责 |
| --- | --- |
| `launch_server_multigpus.sh` | 模型机器一键启动并保持 7 个服务 |
| `launch_client_multigpus.sh` | 仿真机器连接 7 个服务，并行测评与汇总 |
| `launch_server.sh` / `launch_client.sh` | 单任务入口，供批量脚本复用 |
| `run_icl_eval.sh` / `run_task.sh` | 同机串行或分批测评及单任务生命周期管理 |
| `common.sh` | 默认配置、批量进程清理、设备校验 |
| `tasks.txt` | 唯一的任务排序定义 |
| `prepare_pairs.py` | 模型侧读取训练配置与 manifest，选择示范 |
| `batch_tools.py` | 标准库实现的健康检查和指标汇总，客户端不需模型依赖 |

模型与仿真 Python 代码分别位于 `cosmos_framework/evaluation/robotwin/policy.py`、
`cosmos_framework/evaluation/robotwin/simulation.py`，入口为 `scripts/action_policy_server_robotwin.py`、
`scripts/eval_robotwin.py`。后者只负责参数解析与调用；环境配置、episode 循环及指标保存
由 `cosmos_framework/evaluation/robotwin/runner.py` 承担。仿真模块不依赖 torch 或模型权重。

## 双机分别一键启动

两台机器都需要本仓库代码、Bash 5.1+ 和 `setsid`，不需要共享文件系统或 SSH 启动权限。

**模型机器**：在 `common.sh` 的 `server_defaults` 中填写 `CHECKPOINT`，核对
`SERVER_PYTHON`、`HUMAN_GEN_ROOT`、`ICL_PAIR_MANIFEST`、`WAN_VAE_PATH` 和
`COSMOS3_EDGE_PROCESSOR_PATH`。checkpoint 路径必须是 `checkpoints/iter_XXXXXXXXX/model`，
上方的训练 `config.yaml` 必须记录 `robot_window_frames=4k+1 >= 5`。
加载对应设备环境后，在 cosmos-framework 目录执行：

```bash
bash evaluation/robotwin/launch_server_multigpus.sh
```

默认 `MODE=generator`、`COSMOS_DEVICE=npu`、`DEVICES=0,1,2,3,4,5,6`，
监听 `0.0.0.0:8000～8006`。reasoner checkpoint 需设 `MODE=reasoner`；CUDA 模型设
`COSMOS_DEVICE=cuda`。每个服务只使用一个模型设备。
打印 `All seven services ready` 后保持前台运行；Ctrl-C 或任一服务退出时清理本批次服务。
服务日志与示范映射默认写入 `results/robotwin_servers/时间戳_PID/`，可用 `LOG_ROOT` 覆盖。

**仿真机器**：在 `common.sh` 的 `client_defaults` 中填写 `ROBOTWIN_ROOT`，
将 `CLIENT_PYTHON` 设为 RoboTwin 环境解释器（默认当前 `python`），然后运行：

```bash
SERVER_IP=192.168.1.100 bash evaluation/robotwin/launch_client_multigpus.sh
```

默认使用 7 个 CUDA GPU、每任务 100 个 episode，录像默认开启。示例覆盖：

```bash
SERVER_IP=192.168.1.100 START_PORT=8000 \
SIM_DEVICES=0,1,2,3,4,5,6 TEST_NUM=100 SEED=0 \
SAVE_ROOT=/path/to/new-results \
bash evaluation/robotwin/launch_client_multigpus.sh
```

客户端只需要 RoboTwin/SAPIEN、其任务与资产，以及 numpy/scipy/Pillow/PyYAML；
录像还需要 imageio/imageio-ffmpeg。不需要 checkpoint、HumanGen 或 manifest。
默认录像按 head / left wrist / right wrist 拼接，流式保存至 `<task>/videos/`；
`SAVE_VIDEO=0` 可关闭。两端 `START_PORT` 必须一致，且仿真机器能访问全部服务端口。

开始仿真前检查所有端口的任务名、动态指令支持和动作 horizon；客户端结束或中断不关闭
远端服务。某个任务失败后其他任务继续，最终返回非零状态。
结果默认写入 `results/robotwin_clients/时间戳_PID/`，包括 `services.json`、
`logs/<task>.log`、`<task>/metrics.json` 和 `summary.json`。每次使用新目录，不支持断点续测。

## 同机与单任务入口

同机组合入口需要同时具备模型与仿真运行环境：

```bash
# 默认 7 卡并行
bash evaluation/robotwin/run_icl_eval.sh
# 单卡顺序测评全部任务；2～6 卡时分批运行
DEVICES=0 SIM_DEVICES=0 bash evaluation/robotwin/run_icl_eval.sh
```

同机输出默认写入 `results/robotwin/时间戳_PID/`，可用 `SAVE_ROOT` 覆盖。
单任务入口也可在不同机器调用：

```bash
SERVER_BIND=0.0.0.0 bash evaluation/robotwin/launch_server.sh TASK PAIR_ID PORT MODEL_DEVICE
HOST=MODEL_HOST bash evaluation/robotwin/launch_client.sh TASK PORT SIM_GPU OUTPUT_DIR
```

## 任务、示范和评测参数

| 默认端口 | 任务 | 优选人类 sample |
| --- | --- | --- |
| 8000 | stack_blocks_three | 519 |
| 8001 | place_object_scale | 086 |
| 8002 | stamp_seal | 206 |
| 8003 | open_microwave | 242 |
| 8004 | move_stapler_pad | 074 |
| 8005 | place_bread_basket | 007 |
| 8006 | place_empty_cup | 273 |

自动选择 manifest **test split** 中足够长的 pair，优先使用上述 Zero-WAM 对应 sample，
否则按 pair_id 排序选择第一个合格 pair。任一任务缺失时启动前报错。
服务端保存实际 `pairs.json`；设置 `PAIR_MAP_JSON` 可覆盖任意任务，格式为
`{"stamp_seal": "实际pair_id"}`。服务端还会校验 pair 与任务名称匹配。

| 参数 | 默认 | 含义 |
| --- | --- | --- |
| `STEPS` | 20 | 模型采样步数 |
| `EXECUTE_STEPS` | 16 | 每次重规划后执行的原生动作数，不超过服务 horizon |
| `TASK_CONFIG` | demo_clean | RoboTwin 配置 |
| `SEED` / `START_SEED` | 0 / 不设置 | 候选环境 seed 默认从 100000 × (1 + SEED) 开始 |
| `MAX_ATTEMPTS` | 10000 | 专家筛选的最大候选 seed 数 |
| `INSTRUCTION_TYPE` | seen | 当前场景生成的 seen/unseen 指令 |
| `EPISODE_OFFSET` | 0 | episode 编号偏移，不改变 seed 起点 |
| `SAVE_VIDEO` | 1 | 保存真实三相机录像 |
| `VIDEO_FPS` / `VIDEO_STRIDE` | 10 / 5 | 录像帧率与动作采样间隔，另记录初始/终止观测 |
| `SKIP_RENDER_CHECK` | 0 | 显式设 1 才使用 RoboTwin 默认渲染配置 |
| `SERVER_START_TIMEOUT` / `REQUEST_TIMEOUT` | 900 / 300 | 服务就绪/单次推理超时秒数 |

模型设备由 DEVICES 控制；SIM_DEVICES 为仿真 CUDA GPU，可重复指定，但要考虑显存。
HTTP 端口从 START_PORT 起，模型分布式初始化端口再加 10000。

## 闭环语义与 Zero-WAM 差异

沿用专家筛选、同 seed 重置、实时观测和 ee 控制流程。客户端根据专家返回的 episode_info
生成场景指令，随每次预测发送；服务端通过训练预处理更新 generator 文本 token 或
reasoner 视频/文本联合输入，只缓存最近一条指令。新客户端要求服务声明
`supports_episode_instruction=true`，两端代码需同步更新。

每次指令变化会重读所选 pair 的首个窗口，因此模型侧仍需其 parquet 与机器人视频文件。
模板构建后清空离线机器人帧和动作，只填入实时三相机首帧；人类示范保持固定。
当前服务没有跨请求 KV cache，不使用 Zero-WAM 的 reset/compute_kv_cache 协议。

模型返回已反归一化的 `[T,16]` 动作：双臂 `[xyz, quaternion_xyzw, gripper]`。
每轮以当前观测位姿还原整个 chunk：世界坐标平移相加、旋转 `R_current * R_relative`，
gripper 保持绝对值。执行部分动作后重新观测，逐步检查成功和步数上限。
训练监督是 `[first,last)`，所以不照搬 Zero-WAM 的 episode 初始参考与首轮历史动作块跳过。

渲染默认匹配 Zero-WAM：材质/纹理上限 50000、rt shader、32 samples/pixel、路径深度 8、
oidn 降噪，并做初始化检查。环境补齐 policy_name、ckpt_setting、save_root。
专家不可解/UnStableError 跳过 seed，其他错误中止，避免吞掉配置错误。

HTTP 接口为 `GET /info` 和 `POST /predict`；预测请求包含三个相机名对应的 base64 PNG、
seed、必填 instruction，返回 action。与 Zero-WAM WebSocket/msgpack 不兼容。
当前录像不含 Zero-WAM 的动作曲线或预测视频，也不会自动安装其自定义任务文件。

## 验证范围与剩余限制

已有数据集指令更新、相对动作逆变换、未来信息清空、闭环终止、HTTP 通信、
七任务串行/并行与两端独立调度的测试；实际 MP4 编解码和视角排列检查已通过。
曾发现并修复任务与示范缺少校验、健康检查与推理代理行为不一致、固定 caption 未随场景更新的问题。

尚未完成真实 SAPIEN/checkpoint 联调。SciPy xyzw 内部逆变换通过不等于外部接口已验证：
实际数据 schema 只标注 q1～q4，仍需核对部署版本的 get_obs/take_action、四元数顺序、
gripper 范围、控制步长、相机分辨率和 RGB 范围。没有证据时不自动交换四元数分量。
先完成已知动作回放和单 episode，再运行 7 任务成功率测评。

```bash
# 昇腾环境先加载 CANN；纯 CPU 测试可禁用设备自动加载。
TORCH_DEVICE_BACKEND_AUTOLOAD=0 python -m pytest \
  tests/robotwin_policy_test.py tests/robotwin_launchers_test.py \
  --confcutdir=tests -o addopts='' -q
```
