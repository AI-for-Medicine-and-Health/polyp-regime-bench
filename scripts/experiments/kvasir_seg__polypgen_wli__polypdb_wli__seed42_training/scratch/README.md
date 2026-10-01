# Scratch training

这里存放从随机初始化开始训练的脚本、配置和持久化调度器。

计划中的入口包括：

- `common/`：Scratch 专用单模型训练入口
- `configs/`：Scratch 条件配置
- `launchers/`：Scratch 调度器及服务启动脚本

产物统一写入：

`runs/training/kvasir_seg__polypgen_wli__polypdb_wli__seed42/scratch_base/`

Scratch 训练不得加载公共预训练权重；具体入口创建并通过预检后再启动。
