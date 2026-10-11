# 手游策略整合：实现、验证与运行

基线提交：`a329d98cbe62119679a254d71bea6446773541bc`。
仓库：`gbaloong/open-p2p`；GitHub将用户提供的`gbalabs/open-p2p`解析到这里。
开发分支：`integration/mobile-policy-foundation`。

## 一、按依赖顺序完成的代码

| 顺序 | 实现 | 验证边界 |
|---|---|---|
| 1 | 严格预训练加载、显式恢复/随机初始化、固定下载修订、单GPU CPUAdam/ZeRO-2入口、FFN激活检查点、原版推理基准工具 | 加载与配置逻辑有CPU测试；真实1.2B、GPU、DeepSpeed仍待验证 |
| 2 | 并行连续/离散手游动作头、OpenP2P共享核心适配器、时间戳数据集、手游监督训练入口 | 小型测试核心前向/反向、因果关系测试；不是实战能力证明 |
| 3 | 只保留最新指令、过期取消、多触点转换、实际确认时间、停止/重新启用、实际动作反馈 | 仅Trace模拟执行器，没有Android驱动 |
| 4 | 独立轻量视觉纠错网络、连续残差、按钮状态修正、纠错训练入口与控制器 | CPU反向传播、小型合成训练和模拟控制测试 |
| 5 | 伪标签置信度过滤、录制场次隔离、离线VQ潜在动作模型与训练入口、技能切分启发式 | 实验基线；不是完整LAPA，也不能直接从任意直播恢复精确触控 |
| 6 | 有容量/有效期限制的目标记忆、可见性区分、时间/目标条件 | 已有接口和单元测试；跨视角目标特征提取及对齐训练尚未实现 |

这些是独立实现的实验模块，借鉴OFT、A2C2、RTC、潜在动作与视觉目标方法。没有合并其他游戏的技能权重，没有继承论文中的速度或胜率。完整RTC动作补全、EXPO-FT强化学习、Dreamer、V-JEPA教师、ROCKET跨视角训练等仍是后续工作。

仅在获准环境中采集和测试。没有反作弊绕过、游戏内存读取或自动加入公开对局的功能。不要提交权重、录制素材、访问令牌。

## 二、源码中需要特别注意的事实

原训练入口默认构造随机模型。新入口要求三选一：预训练初始化、恢复我们自己的训练、显式从零开始。

原`train_stage3_finetune`函数与PC推理实现保留作对照。新训练入口是单进程/单GPU专用，不代替原多GPU复现实验。

原在线推理每步调用主Transformer两次：先预测动作，再把动作历史写入缓存。不能按一次前向估算耗时。新增手游适配器使用完整因果窗口，绕过原键鼠自回归解码器，使用并行动作头；**它尚不是优化后的手游KV-cache服务器**。

激活检查点只包裹FFN。直接包裹外层注意力模块会破坏原代码依赖`isinstance`的掩码和缓存路由。只检查点FFN不会消除全部激活显存。

## 三、先准备环境和检查代码

遵循原README准备Linux/WSL、Rust、FFmpeg和项目虚拟环境。以下CPU测试仅验证新增模块，不安装完整CUDA/Rust游戏运行链路：

```bash
python -m pytest -q tests/integration
python -m compileall -q elefant/integration elefant/mobile elefant/policy_model/train.py
python scripts/download_checkpoints.py 1200M --dry-run
```

新增GitHub Actions使用CPU PyTorch、NumPy和pytest，不下载1.2B。CI还会直接读取原始源码检查动作掩码，确认当前动作标签和未来图像不能进入当前决策。测试通过不等于真实1.2B/CUDA已通过。

## 四、下载1.2B权重到本地

在准备好的项目虚拟环境运行：

```bash
.venv/bin/python scripts/download_checkpoints.py 1200M
```

默认来源`elefantai/open-p2p`。脚本先解析修订SHA，再按同一修订下载，并记录`download_manifest.json`。权重保存在`checkpoints/1200M/`，不提交Git。

使用**代码仓库**的`config/policy_model/1200M.yaml`，不要使用下载包里已经弃用的配置。下载不等于训练。

加载默认使用`weights_only=True`，没有自动不安全回退。只有检查过来源并信任的pickle检查点才能显式加`--trust-checkpoint`；不要对陌生文件开启。

## 五、先测原版在目标显卡上的速度

将顺序图像准备为RGB uint8数组`[N,3,192,192]`，保存为NPY。该工具不会获取手机画面，不会执行游戏操作。

```bash
.venv/bin/python -m elefant.integration.benchmark \
  --config config/policy_model/1200M.yaml \
  --checkpoint checkpoints/1200M/checkpoint-step=00500000.ckpt \
  --frames datasets/benchmark_frames.npy \
  --warmup 256 --iterations 200 --compile \
  --output outputs/baseline-latency.json
```

测试保留连续历史，至少预热完整历史窗口，做CUDA同步，输出原始样本、p50/p95/p99和显存峰值。计时包含图像拷贝与PC策略计算，不包含手机采集、视频缩放、网络传输或游戏采纳。没有5060 Ti实测结果时，不填预测数字冒充成绩。

## 六、PC策略全参数卸载训练小测试

将DeepSpeed安装到**相同**项目环境，不要无意替换其CUDA版PyTorch：

```bash
uv pip install --python .venv/bin/python deepspeed
.venv/bin/python -m elefant.policy_model.train \
  --config config/policy_model/1200M.yaml \
  --init-checkpoint checkpoints/1200M/checkpoint-step=00500000.ckpt \
  --data_folder data_folder --cpu-offload --checkpoint-ffn \
  --sequence-length 16 --accumulate 4 --max-steps 2 --no_compile \
  --output outputs/pc-transfer-smoke
```

这是待在真实硬件验证的命令，不是已证明16GB显存能容纳的配置。缩短窗口会改变时间学习范围，不能说成完全相同的训练配方。CPU核心数也不能直接推出吞吐。

原数据模块有梯度累积限制。新入口仅在单进程条件下让数据模块维持累积参数1，由Trainer执行用户请求的真实梯度累积；多进程明确拒绝，避免直接移除原多GPU死锁保护。

PC迁移更新原策略参数；手游迁移更新共享视觉/时序核心与新动作模块，未使用的PC动作解码器和外部文本投影显式冻结并报告。不能把它们混为同一训练范围。

DeepSpeed保存的`final.ckpt`可能是分片**目录**。同一策略与精度下用`--resume outputs/.../final.ckpt`恢复。要作为新的权重初始化来源，应先使用已安装DeepSpeed提供的工具合并分片。初始化只读模型参数，不恢复优化器和步数。

## 七、手游数据与训练

JSONL每行一个独立片段；同一完整录制场次不能跨训练、验证、测试集：

```json
{"episode_id":"train-001","recording_id":"recording-A","split":"train","path":"train-001.npz","label_source":"logged"}
{"episode_id":"val-001","recording_id":"recording-B","split":"validation","path":"val-001.npz","label_source":"logged"}
```

NPZ路径相对于清单所在目录，禁止越界。必需数组：

- `frames[N,3,H,W]`：RGB uint8，尺寸与配置相符。
- `executed[N,4+B]`：每帧之后实际确认的输入，不仅是模型提议。
- `targets[N,4+B]`：期望动作；前四项是[-1,1]归一化移动X/Y、每控制步视角增量X/Y；其余B项为0/1按钮状态。
- `timestamps_ns[N]`：同一时钟域、非负、严格递增的整数时间戳。

可选`confidence[N]`、`goals[N,32]`、布尔`visible[N]`。伪标签必须给置信度，低置信度不参与监督；无标签视频不会被伪造为“正确动作全部为零”。日志、人工纠正、伪标签分别记录来源。

布局、灵敏度、控制周期应先固定并校准；当前结构没有实现任意配置自动泛化。路径和录制ID检查不能识别使用者故意把同一内容改成不同ID。

`executed[t]`属于观察t之后的动作。原掩码屏蔽当前和未来动作，只允许历史实际动作，不要额外错移一次。`--delay-frames`使用较旧图像/目标、当前监督和已发生的动作历史，显式提供观察年龄；这是延迟条件模仿学习，**不是完整RTC动作前缀补全**。

```bash
.venv/bin/python -m elefant.mobile.train \
  --config config/policy_model/1200M.yaml \
  --init-checkpoint checkpoints/1200M/checkpoint-step=00500000.ckpt \
  --manifest datasets/mobile/manifest.jsonl --buttons 8 \
  --sequence-length 16 --delay-frames 1 --accumulate 4 \
  --cpu-offload --checkpoint-ffn --max-steps 2 \
  --output outputs/mobile-smoke
```

具备真实权重、依赖、GPU与标注数据后，这个入口进行真实训练。它不会替你采集触控日志。压缩NPZ不支持真正的帧级内存映射，当前每个worker最多缓存一个片段，应控制片段大小。

## 八、快速纠错与执行

`FastCorrector`有独立最新画面视觉分支，输出初始为主策略动作本身，再学习有界连续修正及按钮logit修正。输入包括主计划年龄、控制间隔和实际历史动作。它不是已经训练好的A2C2权重。

```bash
.venv/bin/python -m elefant.mobile.train_corrector \
  --train datasets/correction/train.npz \
  --validation datasets/correction/validation.npz \
  --buttons 8 --steps 1000 --device cuda --output outputs/corrector
```

纠错NPZ含`frames[N,3,H,W]`、`plan_controls[N,4]`、`plan_logits[N,B]`、`executed[N,4+B]`、`targets[N,4+B]`、`timing[N,2]`、`weight[N]`、字符串`recording_ids[N]`。训练和验证的录制ID不能相交。计划必须来自因果回放/实际运行，不能借助未来帧生成。

`CorrectedController`将主计划、最新画面纠错、触控映射和执行器接起来。目前只有Trace后端。真机后端需实现坐标校准、多点事件、I/O超时、释放全部触点，以及同一时钟域的实际完成确认；设备确认仍不等于游戏已经处理。

视角操作第一步先按下建立锚点，不把DOWN当作已经滑动；接近边界时先释放再重新定位。当前是保守参考映射，不是优化完成的高速触控驱动。

真机部署必须有独立看门狗，即使主模型停顿仍能检查过期动作。阻塞的后端调用不能独占控制线程。同一GPU上两个模型也可能排队，不能把小网络单独运行速度当成双模型闭环速度。

## 九、离线录像与目标

`LatentActionVQ`学习离线视觉特征变化的量化表示，不生成精确触控日志，也不包含完整视频编码器/LAPA复现：

```bash
.venv/bin/python -m elefant.mobile.pretrain_video \
  --features datasets/video_features.npz --steps 1000 --device cuda \
  --output outputs/video-latents
```

输入为`features[N,D]`和有序`episode_ids[N]`，不会学习跨片段的相邻配对。仍需保留集评估以及潜在动作到触控的监督映射；当前没有自动写入主策略的隐式步骤。`skill_segments`只是误差变化启发式，必须预先屏蔽剪辑、加载画面等。

目标记忆区分最近看见和现在可见，具有有效期和容量上限；不可见目标不会冒充当前观察。跨视角视觉目标编码、对齐损失、地图级规划仍待实现。

## 方法与工具参考

- OpenP2P: https://github.com/elefant-ai/open-p2p
- OFT: https://arxiv.org/abs/2502.19645
- A2C2: https://arxiv.org/abs/2509.23224
- RTC: https://arxiv.org/abs/2506.07339
- LAPA: https://arxiv.org/abs/2410.11758
- ROCKET-2: https://arxiv.org/abs/2503.02505
- DeepSpeed: https://www.deepspeed.ai/tutorials/zero-offload/

不要用单元测试数量、合成训练损失或其他论文的硬件数字，代替手游实战和5060 Ti测量。
