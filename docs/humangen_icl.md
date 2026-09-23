# Cosmos3-Edge 的 HumanGen ICL 训练与推理

本功能使用 HumanGen 中显式配对的人类示范视频与机器人轨迹，训练 Cosmos3-Edge 的视频／动作策略。两种模式采用相同的机器人监督目标，仅改变人类示范进入模型的位置。

| 模式 | 人类示范的处理方式 | 机器人条件 | 监督目标 |
| --- | --- | --- | --- |
| `reasoner` | 使用冻结的 Edge 视觉编码器，替换 Reasoner 因果文本流中的视频占位符 | 机器人首个潜帧 | 后续机器人视频及原始控制频率的动作 |
| `generator` | 使用 Wan VAE 编码，作为 Generator 流中的干净视觉条件前缀 | 机器人首个潜帧 | 后续机器人视频及原始控制频率的动作 |

Generator 模式分别调用因果 VAE 编码人类视频和机器人视频，再拼接潜变量，避免编码状态跨越两段视频。示范和机器人首帧不参与噪声预测损失；所有动作均为预测目标。模型保留 Cosmos 原生视频／动作联合注意力，不引入额外的因果世界模型目标。

Reasoner 模式复用 Edge 的视觉编码器和多模态位置编码实现。视觉编码器从 `COSMOS3_EDGE_PROCESSOR_PATH/vision_encoder` 加载并冻结，不写入训练 DCP，因此推理也需要同一份 Edge 视觉权重和 processor。当前配方训练生成与动作相关参数，不微调 Reasoner 主干；当前推理只支持 `guidance=1`，不支持负提示词和提示词扩写。

## 数据读取与 Zero-WAM 的对应关系

支持 `robotwin`、`agibot`、`robocoin`、`robomind`、`interna1`、`oxe` 六种来源。适配器读取原有 LeRobot parquet、视频、元数据和 `icl_configs/ICL_config_<source>.json`，不会修改原始 HumanGen 目录。

对照实现为 Zero-WAM 的 `wan_va/dataset/icl_lerobot_latent_dataset.py`、`lerobot_latent_dataset.py`、`lerobot_action.py` 和 `robotwin_action.py`。

### 视频采样与时间对齐

1. 从已有 latent 文件读取 `frame_ids`、采样 `fps`、原始 `ori_fps` 和区间信息，**只复用采样元数据，不把外部 latent 张量直接输入 Cosmos**。
2. 将原始轨迹坐标映射到配对片段的局部坐标，筛选区间内的机器人帧；一个配对片段跨多个缓存文件时合并采样帧。
3. 机器人视频保留最长的 `4k+1` 帧前缀，截去不完整尾部，不复制尾帧补齐。
4. 所有机器人相机解码相同的源帧；人类示范按其缓存中的 `frame_ids` 解码。
5. `SequencePlan.vision_temporal_positions` 保留精确源时间，支持非均匀采样，不强制把间隔当作固定整数步长。

两种模式都保留所选人类示范序列。视频长度未设置统一上限，因此长片段可能占用较大显存；短样本冒烟测试不能证明全部长度均能训练。

### 动作语义与归一化

机器人视频可稀疏采样，但动作保留原始控制频率。首末机器人观测对应源帧 `f0`、`f1` 时，动作采用 `[f0, f1)` 的原生动作行。

元数据驱动的来源按 `action_transform.yaml` 解析 `origin_keys`、状态偏移、绝对／相对动作、姿态格式和局部坐标系；状态偏移在整段 parquet 上完成，再选择窗口。相对动作的参考状态为采样窗口起点状态。Robotwin 使用双臂 xyz／xyzw／夹爪的 16 维布局：位置、旋转相对窗口起点，四元数统一符号，夹爪保持绝对量。

按 `action_stats.json` 的 q01/q99 归一化：

```text
normalized = 2 * (action - q01) / (q99 - q01 + 1e-6) - 1
```

归一化值截断至 `[-2, 2]` 后，再补齐到模型动作宽度（默认 64）。保留 `action_raw` 和 Cosmos 原生 `ActionProcessingRecord`，推理使用同一记录去除补齐并反归一化。截断不可逆，超界原始动作不能通过反归一化精确恢复；验证只要求未饱和通道可逆。

与 Zero-WAM 的表示差异是有意的：这里不加入其零动作历史块，也不使用固定 30 维分组布局，而是使用 Cosmos 的稠密动作序列、源级 domain ID 和实际通道宽度。`action_start_frame_offset` 与动作 FPS 表达时间对齐；Generator 模式额外计入人类前缀的时间偏移。输出动作仍处于上述相对动作语义空间，不能直接视作已经恢复的机器人绝对控制命令。

当前 domain ID 按六种数据来源分配，同一来源中不同机器人／动作 schema 共享投影头。例如 robocoin 同时包含 14 维和 16 维布局。这可运行，但不能据此认定跨机器人语义已完全统一；如需严格隔离 embodiment，应另行设计 schema 级 domain 映射并重新训练。

### 相机与数据划分

相机顺序来自元数据，Robotwin 使用高位、左腕、右腕三相机。多个视图先等比例缩放并补边到共同大小，按宽度拼接，再走 Cosmos 的 resize/VAE 流程。这与 Zero-WAM 的 latent 拼接布局不同；较宽的拼图在固定分辨率下会降低每个视图的空间细节。

manifest 保存动作／相机元数据文件哈希，文件变化后需要重新生成。其余样本按来源和完整任务名做确定性 train/val 划分，默认 seed 为 42、val 比例为 0.1。以下七个 Robotwin 任务固定为 test，加载训练或验证集时再次检查：

```text
place_object_scale
stamp_seal
open_microwave
move_stapler_pad
place_bread_basket
place_empty_cup
stack_blocks_three
```

按任务划分可隔离同任务样本；不能仅凭任务名证明跨任务复用的视频没有泄漏，需要另外检查实际配对路径。

## 准备和验证数据

在 `cosmos-framework/` 下执行：

```bash
export PYTHON_BIN=/mnt/sfs_turbo/public/apps/miniforge3/envs/cosmos-icl/bin/python
export HUMAN_GEN_ROOT=/mnt/sfs_turbo/public/datasets/HumanGen
export ICL_PAIR_MANIFEST="$PWD/../outputs/humangen/pairs_sampled_v4.json"

"$PYTHON_BIN" -m cosmos_framework.scripts.prepare_humangen_pairs \
  --root "$HUMAN_GEN_ROOT" --output "$ICL_PAIR_MANIFEST"
```

默认包含六种来源；`--sources robotwin,agibot` 可选择子集。输出必须位于原始数据目录外，已有 manifest 不会被覆盖。v1–v3 manifest 需要重新生成。

```bash
"$PYTHON_BIN" -m cosmos_framework.scripts.validate_humangen_pairs \
  --manifest "$ICL_PAIR_MANIFEST" --mode generator \
  --output ../outputs/humangen/validate_generator.json

"$PYTHON_BIN" -m cosmos_framework.scripts.validate_humangen_pairs \
  --manifest "$ICL_PAIR_MANIFEST" --mode reasoner \
  --processor /mnt/sfs_turbo/public/ckpts/Cosmos/Cosmos3-Edge \
  --output ../outputs/humangen/validate_reasoner.json
```

验证脚本在每个 split 的每种来源／预处理元数据／相机组合中读取首尾代表样本，检查真实视频解码、动作数值和推理输入遮蔽，不是全量逐样本扫描。

## 训练

```bash
NGPU=8 ICL_MODE=generator bash examples/train_humangen.sh job.name=humangen_generator_run1
NGPU=8 ICL_MODE=reasoner bash examples/train_humangen.sh job.name=humangen_reasoner_run1
```

两份 `examples/toml/sft_config/human_video_icl_*_edge.toml` 显式定义学习率、AdamW 参数、调度器、训练步数、梯度累积、每卡样本上限和保存周期。默认 200 步用于初步实验；调度周期为 1000 步，200 步不会走完整个衰减周期。学习率和训练长度需要按实际任务验证。

模型／数据接线保留在 `cosmos_framework/configs/base/experiment/action/posttrain_config/human_video_icl_edge.py`。FP32 主参数配合低精度 FSDP 前向由该配方启用。每卡一个样本、累积一次时，8 卡的有效 batch 为 8。多节点使用 `examples/train_humangen_dist.sh`，设置 `NNODES`、`NODE_RANK`、`MASTER_ADDR`、`NGPU`。

路径默认值见 `examples/_humangen_env.sh`，可通过环境变量覆盖。Ascend 环境需要保留 CANN/HCCL 动态库路径；不要套用 CUDA 容器的清空 `LD_LIBRARY_PATH` 操作。

## 保存后推理与冒烟验收

### 多卡数据分片与长样本控制

HumanGen 训练 loader 使用 `distributed_shuffle=True`，按 `trainer.seed + epoch`
生成共同随机排列，再通过 DistributedSampler 分配给各 rank。每轮各 rank
样本互不重叠，样本数相同；不能整除 rank 数的随机尾部丢弃（每轮最多
`world_size - 1` 对），下一轮重新洗牌。样本数少于 rank 数时直接报错。
内部迭代器连续跨 epoch 读取，由 `trainer.max_iter` 控制训练结束。
这与 map-style dataset 默认的逐 rank 全量顺序遍历不同。

CP=1 时，有效 batch 为 `world_size × 每卡样本数 × grad_accum_iter`。
VFM 现使用显式 `data_parallel_replicate_degree`；分片度为 -1 时推导为
`world_size / replicate_degree`，显式配置须满足两者乘积等于 world_size。

可在启动命令末尾限制训练样本的采样帧数，例如：

```bash
ICL_MODE=generator NGPU=8 bash examples/train_humangen.sh \
  dataloader_train.dataloader.datasets.humangen.dataset.max_robot_frames=257 \
  dataloader_train.dataloader.datasets.humangen.dataset.max_human_frames=121
```

这两个数值仅为配置示例，不保证特定设备显存足够。默认 0 表示不限制。
超限配对在所有 rank 上一致过滤，并记录保留数量；不截断视频或改变动作对齐。
该过滤不限制原始动作行数，仍需实测长动作序列和长视频的内存开销。

目前 checkpoint 恢复模型、优化器和训练步数，但此 loader 不保存 shuffle
epoch、采样游标或 worker 预取队列；重启会从 seed 对应的第零轮数据重新读取，
不能声称精确恢复数据位置。训练期间 val 尚未启用。多节点通信、吞吐和
完整 checkpoint 恢复仍需在实际集群验收。

先用独立运行名完成两次更新并保存：

```bash
ICL_MODE=generator NGPU=8 bash examples/train_humangen.sh \
  job.name=generator_smoke trainer.max_iter=2 checkpoint.save_iter=2
```

再在新进程中加载 DCP：

```bash
ICL_MODE=generator NGPU=8 bash examples/infer_humangen.sh \
  --checkpoint ../outputs/humangen/train/cosmos3_action/humangen/generator_smoke/checkpoints/iter_000000002/model \
  --output ../outputs/humangen/infer/generator_smoke \
  --split test --samples 1 --steps 2 --seed 42 --condition-ablation
```

Reasoner 验收将模式与运行名对应替换。首次加载基础权重需要配置 `BASE_CHECKPOINT_PATH`、`WAN_VAE_PATH`、`COSMOS3_EDGE_PROCESSOR_PATH`。推理要求 checkpoint 上方存在训练生成的 `config.yaml`，会检查模式并恢复 FP32 主参数选项、精度、分辨率和动作宽度；其他自定义结构修改仍需同步推理配置。

推理输入会清零未来机器人像素和所有动作目标，保留人类示范与机器人首帧。Generator 输出先移除人类 latent 前缀，再单独解码机器人视频。输出包括 `action_*.npy`、`target_*.npy`、`robot_*.mp4`、`metrics.json`；消融模式记录移除示范内容后动作预测的变化。

该入口是有配对数据、已知预测长度的离线评估，不是在线机器人控制接口。冒烟通过只能证明数据读取、反向更新、DCP 保存／重载和采样链路可执行，不能证明收敛、ICL 泛化效果或闭环任务成功率。
