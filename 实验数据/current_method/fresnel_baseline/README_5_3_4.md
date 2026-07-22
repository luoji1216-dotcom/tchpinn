# 5.3.4 实验数据成像验证

本文件夹使用 Institut Fresnel / IOP 官方实验数据库中的 `dielTM_dec8f.exp`，对应论文 5.3.4 中的偏心介质圆柱实验数据。官方数据说明给出的文件列含义为：

1. 发射天线位置编号，1 到 36，对应 0 到 350 度，步长 10 度。
2. 接收天线位置编号，角度步长 5 度。
3. 频率编号，`1` 到 `8` 分别对应 1 到 8 GHz。
4. 总场实部。
5. 总场虚部。
6. 入射场实部。
7. 入射场虚部。

默认实现使用 `总场 - 入射场` 作为散射场观测数据，采用双分支 PINN：场分支拟合多入射方向散射场，介电常数分支输出像素级相对介电常数，并联合数据误差、Helmholtz 残差、Sommerfeld 边界约束与 TV 正则训练。

## 数据下载

```powershell
python .\download_fresnel_data.py
```

已下载的数据会放在：

```text
.\fresnel_2001\dielTM_dec8f.exp
```

## 运行 5.3.4

```powershell
python .\run_5_3_4_fresnel_dieltm_pinn.py --device cuda
```

如果数据缺失：

```powershell
python .\run_5_3_4_fresnel_dieltm_pinn.py --download-missing --device cuda
```

默认频点为 `--frequency-index 1`，即 1 GHz。输出目录默认为：

```text
.\results_5_3_4_fresnel_dieltm_1GHz
```

其中会保存 `pinn_reconstruction.png`、`comparison.png`、`loss_curve.png`、`metrics.json`、`model_final.pt` 等结果。

## 备注

实验天线位于有限距离处，官方说明给出发射半径约 720 mm、接收半径约 760 mm。为了和第五章前面统一的双分支 PINN 平面波物理约束保持一致，代码中将每个发射位置近似为朝向成像中心的等效平面波，并用实测入射场拟合每个视角的复振幅。
