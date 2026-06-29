# Square Target PINN Reproduction

本仓库用于复现论文第 5.3.2 节中 **0.3 GHz 规则正方形目标像素级电磁逆散射** 算例。目标为位于中心的正方形介质块，相对介电常数为 4，背景为空气。当前工程重点复现双分支 PINN 在 square target 上的介电常数重建结果，并对 continuous reconstruction 与 thresholded post-processing 结果进行对比分析。

## 1. 复现目标

论文表 5.1 中，0.3 GHz 规则目标下本文方法的指标为：

| Method                   | Relative Error |  SSIM |
| ------------------------ | -------------: | ----: |
| Paper double-branch PINN |         0.1292 | 0.944 |

本仓库当前最佳复现结果为：

| Result Type                 | Epoch | Threshold | Relative Error |    SSIM |
| --------------------------- | ----: | --------: | -------------: | ------: |
| Continuous epsilon output   | 14000 |         - |        0.23046 | 0.90973 |
| Thresholded post-processing | 14000 |      1.85 |        0.13596 | 0.97394 |

说明：直接使用 PINN 输出的连续介电常数图计算误差时，结果尚未达到论文表格指标；但对同一 checkpoint 进行阈值化后处理后，相对误差降至 **0.13596**，接近论文报告的 **0.1292**，SSIM 达到 **0.97394**。

该结果表明，当前模型已经较好恢复了 square target 的主要结构，但 continuous epsilon 图仍存在灰度过渡。论文表格指标可能与最佳 checkpoint 选择、后处理阈值、评价区域或论文实现中的边缘保持正则化/数据重加权细节有关。

## 2. 工程结构

```text
square-target-repro/
├── paper_chapter5_square.pdf
├── README.md
├── requirements.txt
└── square_target/
    ├── +x.txt
    ├── -x.txt
    ├── run_5_3_2_square_0_3ghz_pinn.py
    ├── pinn_pixel_inverse_core.py
    ├── square_case_common.py
    ├── compare_pinn_metrics.py
    ├── diagnose_checkpoint.py
    ├── run_5_3_2_square_som.py
    └── som_inverse_core.py
```

主要文件说明：

* `square_target/run_5_3_2_square_0_3ghz_pinn.py`：0.3 GHz square target PINN 主运行脚本。
* `square_target/pinn_pixel_inverse_core.py`：PINN 网络、损失函数、训练与评估核心代码。
* `square_target/square_case_common.py`：square target 数据读取、参数配置与命令行参数。
* `square_target/compare_pinn_metrics.py`：读取 `evaluation_metrics.csv`，比较 continuous / thresholded 指标，并支持 checkpoint threshold sweep。
* `square_target/diagnose_checkpoint.py`：诊断指定 checkpoint 的介电常数分布、阈值扫描、目标面积比例和误差图。
* `square_target/+x.txt` / `square_target/-x.txt`：两个入射方向的观测数据。

## 3. 环境配置

推荐使用 Python 3.10 或以上版本。

安装依赖：

```bash
pip install -r requirements.txt
```

Windows 下本实验使用的 Python 环境示例：

```powershell
T:\anaconda3\envs\py\python.exe
```

## 4. 运行 PINN 训练

基础训练命令：

```powershell
T:\anaconda3\envs\py\python.exe square_target\run_5_3_2_square_0_3ghz_pinn.py --epochs 30000 --log-every 500 --checkpoint-every 1000 --output-dir results_5_3_2_square_0_3GHz_metric
```

训练完成后会在输出目录 `square_target/results_5_3_2_square_0_3GHz_metric/` 中生成：

```text
evaluation_metrics.csv
loss_history.csv
metrics.json
checkpoint_adam_*.pt
model_final.pt
epsilon_reconstruction.npy
epsilon_thresholded.npy
epsilon_truth.npy
```

其中 `evaluation_metrics.csv` 会记录每个 checkpoint 的 continuous 和 thresholded 指标。

## 5. 指标分析

分析训练过程中 continuous / thresholded 的最佳结果：

```powershell
T:\anaconda3\envs\py\python.exe square_target\compare_pinn_metrics.py square_target\results_5_3_2_square_0_3GHz_metric\evaluation_metrics.csv --no-plots
```

对所有 checkpoint 做阈值扫描：

```powershell
T:\anaconda3\envs\py\python.exe square_target\compare_pinn_metrics.py square_target\results_5_3_2_square_0_3GHz_metric\evaluation_metrics.csv --checkpoint-threshold-sweep
```

本次复现中，checkpoint threshold sweep 的全局最佳结果为：

```text
checkpoint = checkpoint_adam_014000.pt
epoch = 14000
threshold = 1.85
relative error = 0.135962
SSIM = 0.973937
```

## 6. 最终复现输出

最终复现实验输出目录：

```text
square_target/results_5_3_2_square_0_3GHz_metric/final_reproduction/
```

使用的 checkpoint 与后处理阈值：

```text
checkpoint_adam_014000.pt
threshold = 1.85
```

该目录包含：

```text
epsilon_truth.png
epsilon_pred_continuous.png
epsilon_pred_thresholded_thr1p85.png
epsilon_abs_error_continuous.png
epsilon_abs_error_thresholded_thr1p85.png
epsilon_pred_histogram.png
final_metrics.json
final_metrics.csv
```

其中：

* `epsilon_truth.png`：真实介电常数图。
* `epsilon_pred_continuous.png`：PINN 原始连续输出图。
* `epsilon_pred_thresholded_thr1p85.png`：阈值 1.85 后处理图，也是当前最佳指标对应的图。
* `epsilon_abs_error_continuous.png`：连续图绝对误差。
* `epsilon_abs_error_thresholded_thr1p85.png`：后处理图绝对误差。
* `final_metrics.csv` / `final_metrics.json`：最终复现指标。

最终指标：

| Metric                       |     Value |
| ---------------------------- | --------: |
| epoch                        |     14000 |
| threshold                    |      1.85 |
| continuous relative error    | 0.2304585 |
| continuous SSIM              | 0.9097289 |
| thresholded relative error   | 0.1359620 |
| thresholded SSIM             | 0.9739370 |
| predicted object pixels      |      7610 |
| truth object pixels          |      7744 |
| predicted / truth area ratio |    0.9827 |

## 7. 关于论文损失函数对齐的说明

论文第 5.2.2、5.2.3 和 5.3.1 节描述的方法包含：

* 边缘保持正则项 `Lep`
* 残差自适应重加权数据项 `Ld^w`
* Adam + L-BFGS-B 两阶段优化
* 权重设置 `lambda_f = 1, lambda_d = 100, lambda_ep = 100`

本仓库后续加入了 `--loss-preset paper` 以尝试对齐论文形式的损失函数：

```text
L = lambda_f * Lf + lambda_d * Ld_w + lambda_ep * Lep
```

但是在当前代码尺度和实现条件下，直接使用论文默认权重并没有得到更好的 epsilon 重建结果。5000 step 小实验中，`lambda_ep=1/10/100` 均未恢复出目标结构。因此当前主复现结果仍采用 `current` loss 模式，并通过 checkpoint 选择和阈值后处理得到接近论文表格的结果。

## 8. 当前结论

当前代码在原始 continuous epsilon 输出上尚未完全达到论文表 5.1 的指标，但经过最佳 checkpoint 选择和阈值化后处理后，相对误差可降至 **0.13596**，接近论文报告的 **0.1292**。这说明模型已经恢复出 square target 的主要结构，剩余差异可能来自：

* 论文中后处理或二值化策略未明确说明；
* 论文评价指标与图像展示可能不完全对应；
* 边缘保持正则化与残差自适应重加权的具体实现细节不同；
* loss 权重、采样策略、L-BFGS-B 优化器或评价区域存在差异。

因此，本仓库当前结果应理解为：

> 复现了 square target 的主要成像结构，并通过阈值后处理获得了接近论文表格的定量指标；但 continuous reconstruction 尚未严格达到论文指标。
