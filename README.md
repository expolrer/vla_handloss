# VLA Hand Loss

灵巧手 VLA 的手部专用建模消融：在相同 RGB 示范、关节映射、训练预算与部署协议下，比较统一动作头、手部残差分支、手部损失加权和两者组合。

当前实现是 **π0.5 JAX 全参微调权重的 Bench2Dex 在线评测系统**。输入为四视角 RGB、joint 和固定任务 prompt，**不使用深度或触觉**。ACT 对照与触觉扩展仍是后续计划，不冒充已实现或已获得成绩。

目录沿用 [Depth Process Model](https://github.com/expolrer/depth-process-model) 的配置、工作区、脚本、测试和文档分层；任务、权重、成功率与 RoboTwin 项目完全分开。

## 实验矩阵

| 条件 | 输出分支 | 手部权重 λ |
|---|---|---:|
| B0 统一基线 | 无 | 1 |
| B1 仅手部分支 | 40维手部非线性残差 | 1 |
| B2 仅额外手部损失 | 无 | 2 |
| B3 分支＋损失 | 同B1 | 2 |
| C0 参数量对照 | 全动作非线性残差，宽253 | 1 |
| C1 手臂/手部组平衡 | 无 | 装球0.30；拼图0.35 |
| C2 降低手部权重 | 无 | 0.5 |
| C3 提高手部权重 | 无 | 4 |

两个任务共16组，训练种子统一42，目标10,000更新、batch32、每任务固定50条canonical示范。单种子只能给出单次训练证据，训练更久不能替代多种子稳定性验证。

| 任务 | 机器人 | 主动动作维度 | 手部维度 |
|---|---|---:|---:|
| Ball Box Loading | 双UR5＋Wuji | 52 | 40 |
| Jigsaw Puzzle Assembly | 双iiwa7＋Sharpa | 54 | 40 |

拼图原始状态58维，经随权重保存的主动关节映射选为54维；不能简单截取前54维。每只手至少20个主动关节，双手共40维。

## 快速使用

```bash
# Python 3.11；完整依赖分属推理和仿真两个环境，不装进同一个venv。
python scripts/run.py matrix

# 复用已校验的运行时；只建立工作区链接，不复制或重写权重。
python scripts/run.py setup --runtime /path/to/prepared-runtime
python scripts/run.py doctor

# 预览不会占用GPU。只有明确加 --execute 才会运行。
python scripts/run.py queue --gpus 0
python scripts/run.py queue --gpus 0 --execute
python scripts/run.py status

# 单组控制链路检查，再进行正式50轮。
python scripts/run.py evaluate --run 27_ball_box_loading_B0_unified --gpu 0 --smoke
python scripts/run.py evaluate --run 27_ball_box_loading_B0_unified --gpu 0

python -m unittest discover -s tests -v
```

首次安装请阅读 [部署步骤](docs/DEPLOYMENT.md)。配置在 `config/`，本机覆盖只写入被Git忽略的 `workspaces/local.yaml`。自有代码以仓库根目录解析相对路径；权重、数据、环境、密码和日志均不推送GitHub。

## 评测与恢复

- 复用固定版本 Bench2Dex 的场景、物理控制、RGB相机、成功判定和policy server/client；每个模型先过一次独立控制smoke，smoke不混入正式分数。
- 默认每任务/模型50轮、800个policy steps、60步warmup、20步预测与执行chunk、稳定保持判定沿用官方任务定义。这是发布的π0.5部署YAML预算，不宣称等同于论文逐轨迹重算的1.5倍expert时限。
- episode从1开始，沿用官方 `namespace + task_id × 100000 + episode_index - 1`，而不是自行生成随机seed。native server每轮重置JAX RNG，所有条件使用同一训练prompt。
- SQLite WAL事务保存完成轮次；只从episode边界恢复。被打断的当前轮会重跑，既不臆造中途RNG状态，也不计为失败。协议、代码或权重指纹变动时禁止混用结果。
- `SUCCESS / TASK_FAILURE / RUNTIME_FAILURE / ACTION_EXECUTION_INVALID` 分开记录。CUDA错误、缺资产、无实际关节运动都不能伪装成普通0成功率。
- `none` 是anchor匹配，不代表空白干净背景。锚点里的背景、桌面纹理和杂物必须保留；这些锚点匹配训练示范，不能称作held-out测试集。
- 目前先排 `none` 通道；Cov/Inv/Full需要完整资产池及额外审计后才启用。拼图原始HDF5的 `meta/success=false` 保持原样；正式解释分数前仍需expert replay复核成功检查器，不能把这个字段当训练标签或模型结论。

## 工程布局

```text
config/                    任务、八种条件、预算、部署与中转模板
hand_pipeline/             模型适配、控制审计、记录、事务日志与动态队列
scripts/                   一键入口、离线环境安装、anchor资产准备、overlay安装
tests/                     配置一致性、恢复、校验门槛与动作执行有效性测试
workspaces/architectures/   与已训练权重对应的模型实现及机器人映射overlay
docs/                      研究计划、部署协议和记录保留要求
records/                   本机原始日志、配置快照和代码/环境指纹，不入Git
results/                   官方逐轮记录及验证后的统计，不入Git
checkpoints/               已完成SHA256校验的部署权重，不入Git
```

项目不会重启现有训练、自动改训练预算或删除训练状态。56上的既有训练继续使用原记录入口；所有后续训练必须保留原始loss数据、每次启动的完整配置、代码和环境指纹，且记录独立于checkpoint目录。

当前仓库不预填成功率。[研究设计](docs/RESEARCH_PLAN.md) · [评测协议](docs/EVALUATION_PROTOCOL.md) · [来源与许可](NOTICE)
