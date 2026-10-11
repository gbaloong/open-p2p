# 本地 / 自托管 CI

本文件定义组件CI，不代表完整CUDA训练环境已经安装。模型功能开发暂缓，先让同一测试入口在本地、自托管runner和云端运行。

## 工作流

- `.github/workflows/self-hosted-ci.yml`：默认 `self-hosted, Linux, X64`，与参考工程的Linux自托管方式一致；指定分支push或手动触发。
- `.github/workflows/mobile-integration.yml`：保留GitHub托管的PR检查；不让外部PR自动执行在本地机器上。
- 两者运行同一个 `scripts/ci/run.py`，不各维护一套不同测试。
- 自托管checkout在本次运行专属的 `ci-work/<run-id>-<attempt>/`；只清理这个新建目录。没有全仓 `git clean -ffdx`、递归chown或自动sudo。
- 虚拟环境位于runner用户的 `~/.cache/open-p2p-ci/<profile>-<fingerprint>/`，与训练用 `.venv` 分离；缓存指纹包含Python版本、PyTorch和依赖清单，安装过程持有文件锁。

## 环境内容

Actions固定Python 3.11；脚本支持本地Linux/WSL的Python 3.11到3.13。

- CPU：PyTorch 2.10.0 CPU轮子、numpy 2.3.5、pytest 9.0.2、PyYAML 6.0.3。
- 可选CUDA：单独的PyTorch 2.10.0 cu128环境，用于小模型BF16前向/反向/参数更新检查。
- 不执行整个项目的 `uv sync`；不改系统Python、不安装驱动、不自动安装Rust/FFmpeg/DeepSpeed、不下载1200M权重、不训练游戏策略。
- CPU测试隐藏GPU；测试期间禁止Hugging Face隐式联网，关闭在线实验日志。

上游完整训练依赖要求与组件CI不同，不能把这里的绿色检查当作完整训练依赖已解决。

## 本地使用

在代码根目录，使用已有Python 3.11-3.13创建隔离环境：

```bash
python3 scripts/ci/bootstrap.py --profile cpu
```

命令末尾的JSON会给出精确 `python` 路径。使用该路径运行：

```bash
/实际输出的虚拟环境/bin/python scripts/ci/run.py --profile cpu
```

先预览、不安装任何依赖：

```bash
python3 scripts/ci/bootstrap.py --profile cpu --dry-run
```

可通过 `--cache-root /独立缓存目录` 或 `OPENP2P_CI_CACHE` 改变环境位置。不要让CPU缓存和训练环境共用同一虚拟环境。

缺少venv支持时，先由机器管理员为该Python配置venv能力。脚本会报错，不会擅自使用sudo或覆盖系统包。

## GitHub runner接入

YAML只能选择已注册且允许当前仓库使用的runner，不能注册、启动或授权runner。

已有组织runner时，保留原有注册，不要删除或重建其他项目runner。管理员需确认：该runner在线；对应runner组允许 `gbaloong/open-p2p`；标签匹配；runner账户有可写缓存目录并能下载官方依赖。仅在PSD仓库注册的repository runner，不会自动变成open-p2p的runner。

本地Code runner连接与GitHub Actions runner是两种连接，不能以其中一个在线证明另一个在线。

### CUDA检查

仅手动触发并选择 `cuda`；目标runner除上述通用标签外，必须有 `open-p2p-gpu`。只给已确认具备CUDA驱动和目标GPU的机器加这个标签。没有该标签会排队，而不会错误落到普通CPU服务器。

CUDA检查要求设备可用和BF16支持，不允许自动回退CPU后报成功。环境报告明确记录：没有加载1.2B，没有验证DeepSpeed，没有操作手机。

新工作流在开发分支首次提交可通过该分支push触发；手动入口通常须先存在于默认分支。不要为了显示手动按钮就提前合并尚待验收的全部模型改动。

## 安全与资源边界

自托管机器不是隔离沙箱。尤其公开仓库，禁止批准外部提交在长期运行的共享主机上执行；仅分支过滤不能替代组织runner组权限、外部贡献者审批和隔离账户。建议专用低权限runner，不存个人密钥和其他项目凭据，不授予免密sudo。工作流token只读且checkout不持久保存凭据。

自托管作业超时30分钟；同分支同profile取消旧运行；安装锁避免同时改同一缓存。工作流不上传权重、录像或整个机器环境，只上传此次测试生成的报告。

## 结果如何判定

`outputs/local-ci/run-<timestamp>/` 包含环境报告、各步骤日志、JUnit和 `summary.json`。

通过条件：环境小模型参数更新成功、pytest退出0、有真实执行的测试用例、零失败/错误/跳过、语法检查通过、模型下载dry-run通过。空测试、全部跳过、旧XML、超时、依赖失败均不得报绿。

- runner `queued` / 未分配runner：尚未验证本地环境，检查runner在线、标签和仓库访问范围。
- 任务已执行但失败：阅读实际失败步骤，不将安装失败等同于模型不可行。
- CPU绿色：只证明组件CI。
- CUDA smoke绿色：只证明小模型的CUDA计算路径。
- 真正1.2B训练和速度仍需独立验证。

本次本地脚本测试在助手工作区执行；用户服务器是否完成安装，以对应自托管作业实际状态为准。
