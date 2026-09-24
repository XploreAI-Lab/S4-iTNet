# S4-iTNet

[English](README.md)

S4-iTNet 是一个用于临床 EEG 癫痫类型分类的检测引导通道聚合框架。该项目包含数据预处理、模型训练、评估、可视化以及 CPU 冒烟测试代码。

## 环境安装

推荐使用 Python 3.9、PyTorch 2.1.2 和 CUDA 11.8。由于训练使用 PyKeOps，Linux 环境以及 C++ 编译器和 CUDA 开发工具包通常更适合 GPU 训练。

```bash
conda create -n s4_itnet python=3.9
conda activate s4_itnet
pip install torch==2.1.2 --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements-cuda.txt
```

如果只需要运行 CPU 冒烟测试，可以根据本机环境安装 `requirements.txt` 中的依赖。

## 数据准备

请按照数据集的使用条款，从 [TUSZ v2.0.6](https://isip.piconepress.com/projects/tuh_eeg/html/downloads.shtml) 获取数据。患者 EEG 数据不包含在本仓库中。

将 EDF 记录及其标注放置在以下结构中：

```text
edf/{train,dev,eval}/<patient>/...
```

本项目基于 TUSZ v2.0.6 构建固定的患者级训练、验证和测试划分。最终实验划分由 `preprocessing/patient_split.csv` 定义；官方 `train`、`dev` 和 `eval` 目录仅用于定位原始记录，不作为最终实验划分保留。

然后运行 10 秒窗口的数据构建流程：

```bash
python -m preprocessing.build_dataset \
  --input-root /path/to/tusz_v2.0.6 \
  --output-dir /path/to/processed_10s \
  --assignment-csv preprocessing/patient_split.csv \
  --duration-sec 10 --workers 1 \
  --overlap-ratio 0.40 --gnsz-stride-sec 6 \
  --absz-stride-sec 0.2 --ctsz-stride-sec 2
python data_audit.py --data-root /path/to/processed_10s
```

每个数据划分（`train`、`val`、`test`）包含：

| 文件 | 内容 |
|---|---|
| `<split>.npy` | EEG 窗口，形状为 `(N, 22, 2000)` |
| `<split>_multi_label.npy` | 癫痫类型标签，形状为 `(N,)` |
| `<split>_bi_label.npy` | 通道级二分类标签，形状为 `(N, 22)` |

构建程序还会生成 `window_manifest_array_order.csv`。类别编号为：`0=CFSZ`、`1=GNSZ`、`2=ABSZ`、`3=CTSZ`。更完整的数据格式和审计约束见 [docs/DATA.md](docs/DATA.md)。

原始 EEG 不应上传到代码仓库。数据划分、访问权限和使用条款应遵循 TUSZ 官方规定。

划分 CSV 包含窗口筛选前分配的患者（训练/验证/测试为 196/22/18 人），不等于最终保留样本涉及的人数。主实验发现的合格训练事件涉及 169 人、1,413 次事件；切分为满足条件的 10 秒窗口后，部分事件未留下窗口，最终涉及 167 人、1,401 次事件。最终验证/测试窗口分别涉及 21/16 人。

## 模型训练

```bash
python train.py \
  --data-root /path/to/processed_10s \
  --output runs/s4_itnet \
  --device cuda:0
```

默认配置位于 [configs/s4_itnet_10s.json](configs/s4_itnet_10s.json)。如需使用自定义配置，请通过 `--config` 指定，并为每次实验使用新的输出目录。

默认配置采用 10 秒窗口、22 个通道、4 个 S4 层和 4 个 iTransformer 编码器块。训练阶段的类别目标采样数为 `8000/4000/800/800`；验证集和测试集不进行训练集式重采样。


## 通道检测模型选择

同一次 `train.py` 两阶段训练中，Stage 1 按验证集通道 F1、通道准确率和 detection loss 保存 checkpoint，位于 `model/stage1_selection_checkpoints/`；Stage 2 按验证集分类指标保存 checkpoint，位于 `model/selection_checkpoints/`。两个目录均保存 `selection.json` 和验证集混淆矩阵。进入 Stage 2 的条件是验证集 detection loss 的 patience。

评估这次训练中按通道 F1 选出的 Stage 1 checkpoint：

```bash
python evaluate.py --data-root /path/to/processed_10s \
  --checkpoint runs/s4_itnet/model/stage1_selection_checkpoints/best_by_channel_f1.pth \
  --detection-only --device cuda:0 --output outputs/stage1_detection
```

通道指标汇总所有窗口与通道组合，sigmoid 概率大于 0.5 时判为阳性；常规评估也会输出通道检测指标。

报告的 Stage 1 测试通道准确率为 0.738516、通道 F1 为 0.776546（epoch 2，验证通道 F1 为 0.711083），Stage 2 测试通道 F1 为 0.6938；两者是同一次两阶段训练中分别选出的 checkpoint 在测试集上的结果。Stage 2 checkpoint 按验证集 weighted F1 选出。仓库目前仅提供 Stage 2 的权重；运行 `train.py` 会生成两个阶段各自的 checkpoint。

## 模型评估

```bash
python evaluate.py \
  --data-root /path/to/processed_10s \
  --checkpoint runs/s4_itnet/model/selection_checkpoints/best_by_weighted_f1.pth \
  --device cuda:0 \
  --output outputs/evaluation
```

仓库中提供的模型权重可通过以下路径使用：

```text
checkpoints/s4_itnet_10s_wf1.pth
```

评估脚本会保存分类指标、混淆矩阵、预测结果和通道权重。

## 通道权重可视化

使用测试集窗口绘制 EEG 波形和通道权重：

```bash
python visualize.py \
  --data-root /path/to/processed_10s \
  --checkpoint checkpoints/s4_itnet_10s_wf1.pth \
  --index 0 --device cuda:0 \
  --output outputs/channel_weights.png
```

下面给出一个通道权重可视化示例，其中包含 CFSZ 和 GNSZ 两类癫痫类型。红色波形和柱状条表示标注为 ictal 的通道，黑色波形和柱状条表示非 ictal 通道，灰色虚线表示等权重基准（`1/22`）。仓库中提供的数据样本是用于测试流程的合成数据，因此无法使用这些样本复现下面两张基于真实 EEG 的通道权重图。该示例图基于 TUSZ v2.0.6 记录生成；由于数据集的访问和使用条款，作者无法公开或再分发相应的真实 EEG 样本。若要生成同类图示，请按照 TUSZ 官方条款获取数据，并遵循上面的数据准备和可视化步骤。

图中两个面板分别对应以下测试集样本：

| 图中位置 | 类别 | 测试集数组索引 | 患者编号 | 原始记录 | 标注事件区间 | 10 秒窗口 | ictal 通道数 |
| --- | --- | ---: | --- | --- | --- | --- | ---: |
| 左图 | CFSZ | 49 | `aaaaarpv` | `aaaaarpv_s001_t003.edf` | 126.9927--214.8366 s | 190.9927--200.9927 s | 4/22 |
| 右图 | GNSZ | 577 | `aaaaatdt` | `aaaaatdt_s001_t001.edf` | 1.0000--1162.9329 s | 580.0000--590.0000 s | 10/22 |

![通道权重可视化示例](docs/channel_weights_example.png)

## 冒烟测试

仓库包含合成数据样例，可用于在 CPU 上检查基本流程。样例不包含患者数据。

```bash
python smoke_test.py --device cpu
python train.py --data-root dataset/smoke_demo --output outputs/train_smoke --device cpu --smoke
python -m pytest -q
```

## 许可证与致谢

本项目使用了 [S4](https://github.com/state-spaces/s4) 和 [iTransformer](https://github.com/thuml/iTransformer) 的相关组件。项目代码采用 [Apache-2.0](LICENSE) 许可证；第三方组件的许可信息见 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。
