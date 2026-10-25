# 部署

## 两个隔离环境

- 推理：Python3.11、JAX0.5.3、Flax0.10.2、Orbax0.11.13，复用已有测试通过的依赖目录时保持只读，并在记录中保留其位置。删除被复用的基础环境会破坏这个venv。
- 仿真：Isaac Sim5.1.0、Isaac Lab2.3.2、PyTorch2.7.0+cu128、Gymnasium1.2.1。Isaac Sim wheel要求NumPy1.26.0，与Bench2Dex README的1.26.4建议冲突；仿真遵循wheel依赖，推理仍为1.26.4。
- 不把上游OpenPI pyproject里的RoboTwin/SAPIEN依赖混入Isaac仿真环境。
- Isaac Lab的Git release标签与Python distribution版本不是同一个编号：本次v2.3.2源码构建出的包版本为0.54.2。应安装这份源码/本地wheel，不能猜测 `pip install isaaclab==2.3.2`，后者会拿到不同元数据和Gymnasium约束。

在relay准备离线wheelhouse和IsaacLab源码，再上传评测服务器；`setup_environment.py` 不联网安装：

```bash
# 仅在下载中转机执行。传入固定v2.3.2的IsaacLab源码。
python3.11 scripts/prepare_simulation_wheels.py \
  --wheelhouse workspaces/wheelhouse --isaaclab workspaces/IsaacLab

# 以下在评测机执行，wheelhouse与源码已从中转机传入。
python3.11 scripts/setup_environment.py simulation \
  --wheelhouse workspaces/wheelhouse --isaaclab workspaces/IsaacLab

python3.11 scripts/setup_environment.py inference \
  --reuse-site-packages /path/to/tested/inference/site-packages
```

离线准备会将flatdict等源码依赖预先构建为wheel，并使用仍提供pkg_resources的构建依赖，避免评测机离线安装时重新联网或构建失败。依赖清单及SHA256保存在wheelhouse中。

## 工作区与模型

使用固定Bench2Dex版本后，运行 `python scripts/apply_model_overlay.py --workspace workspaces/Bench2Dex` 安装所训练的模型和机器人映射。原文件备份在records，不入Git。已有准备好的runtime使用README里的 `setup --runtime`，不必复制checkpoint或整个环境。

checkpoint必须包含 `params/`、`assets/dex2bench/norm_stats.json`、`deployment.json`、源端 `SHA256.json` 及目标最终 `VERIFIED.json`。不以文件名或大小代替内容校验。

`workspaces/architectures/pi05/overlay` 是从实际部署源代码快照提取的模型实现，`bench2dex_overlay/robots` 提供匹配训练的主动关节映射。权重不入Git，下载token、SSH密码与私钥也不入Git。

## 启动队列

```bash
python scripts/run.py queue --gpus 0 --execute
```

队列只对授权GPU编号检查是否空闲。环境、核心资产、anchor资源和最终权重未就绪时显示等待原因；这些缺失不能产生成功率。每组先独立smoke，再按相同协议正式评测。SIGTERM/SIGINT请求在episode边界停下；机器重启后重新运行同一命令即可从已提交轮次恢复。

服务器没有systemd时，不能把创建unit文件当作已安装服务。用进程管理器或会话运行队列，并检查 `state/queue.json`、records日志和实际GPU进程；重启后应重新确认它们。
