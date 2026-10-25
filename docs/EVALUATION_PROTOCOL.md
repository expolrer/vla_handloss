# 评测协议

## 固定项

官方Bench2Dex commit、场景YAML、robot配置、四RGB视角、主动关节映射、相机/控制频率、success dwell、action horizon20、固定训练prompt、每任务episode1–50和native RNG reset均保持一致。每组只替换匹配架构的10,000步权重和自己的训练统计量。

默认800个policy steps、warmup60、headless、early-stop稳定成功，来自发布的π0.5部署配置。native CLI单独运行时的默认值可能不同，因此不能省略覆盖参数。CPU准备不代表GPU渲染、模型加载和动作控制已通过。

## 成绩与有效性

使用原生 `per_episode.jsonl`，额外输出带协议指纹的 `validated_summary.json`。事务日志只统计实际完成的SUCCESS与TASK_FAILURE。没有完成的episode不凑分母；运行故障和动作未执行的轮次必须重新检查，而不是记为0。

smoke是独立命名空间。它验证四视角、joint、有限动作输出、实际qpos变化和官方结果记录，不等于验证模型能完成任务，也不替代expert replay的成功检查器审计。

## 恢复

同一job的协议、代码和checkpoint manifest SHA256必须相同。已经提交的episode跳过；中断的当前episode按原seed从初始场景重跑。官方server重置JAX RNG，不尝试从未知的中途随机数状态接着执行。

逐轮启动仿真client，模型server在同一job内保持运行。这比每轮重新加载模型更省时，但会有仿真初始化开销；当前优先保证错误隔离和恢复语义。

## 资产

通过100个origin anchor中的 `meta/scene_generalization_sample` 收集实际使用的背景、桌面纹理和clutter，不能扫描整个配置池后把所有资产都当成本轮依赖。下载只在relay执行，模型和数据不经本地转发。

根目录资源下载后还需验证USD传递依赖。只有源端、目标端资源校验和传递依赖检查完成后，才可写 `ANCHOR_ASSETS_VERIFIED.json`。`none` 也可能需要iTHOR背景，绝不能以缺资产自动回退的空场景冒充同口径评测。
