# 三角洲手游方向：OpenP2P 1.2B二次开发

开发代码在本仓库，模型权重和录制数据保留在训练机本地。

先阅读 [整合说明与运行命令](doc/MOBILE_INTEGRATION.md)，再按 [当前下一步](doc/CURRENT_NEXT_ACTION.md) 验证。

本轮实现：显式预训练加载与恢复、单GPU卸载训练入口、手游动作与数据模型、触控时序模拟执行、轻量最新画面纠错、离线潜在动作学习、目标记忆与CPU测试。

**当前是实验性代码整合，不是训练完成的三角洲手游AI。** 尚未验证真实1.2B加载、5060 Ti速度、CUDA/DeepSpeed更新或Android真机控制。PC推理保持原版对照；手游在线KV缓存仍需实现。A2C2/OFT/RTC/LAPA/ROCKET是方法参考，不代表已完整复现这些研究。

训练入口现在必须显式选择 `--init-checkpoint`、`--resume` 或 `--from-scratch`，避免误从随机参数开始。
