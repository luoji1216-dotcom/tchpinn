# Austria 目标 0.3 GHz 优化版说明

FEM 文件第 8、9 列为 `FieldzRe` 和 `FieldzIm`。经方向拟合诊断，当前双分支 PINN 的 outgoing 边界约定下，奥地利目标数据应按如下方式读取：

```text
Ez = FieldzRe - i * FieldzIm
phase_sign = +1
```

诊断命令：

```powershell
python .\run_5_3_3_austria_0_3ghz_optimized.py --diagnose-imag-sign
```

此外，`+x.txt` 有 10021 行，而 `-x.txt`、`+y.txt`、`-y.txt` 各有 836 行。优化版默认设置：

```text
max_points_per_direction = 836
```

用于平衡四个入射方向的数据权重。

## 推荐运行

从工作区根目录运行：

默认训练轮数为 30000。快速验证可使用：

## 本次验证结果

8000 步短程验证结果目录：

```text
.\results_5_3_3_austria_0_3GHz_opt_v4_8000
```

结果指标：

```text
relative_error = 0.3540
SSIM = 0.7261
elapsed_s = 373.76
```

从图像上看，两个上圆、下方圆环主体与中心空洞均已恢复，但背景仍存在一定扩散。继续训练到默认 30000 步会进一步改善 PDE 残差与边界清晰度。 

## 噪声水平

优化版脚本也支持噪声测试
