#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
散点 + 上/右边际柱状图（含预筛选与统计）：
- 先筛选：90 ≤ c_chain_atom_plddts_mean ≤ 100，0 ≤ netmhc_nM ≤ 10
- 主图坐标：X=90–100, Y=0–10
- 边际柱状图去掉刻度与数值，仅保留轴线
"""

import pandas as pd
import matplotlib.pyplot as plt
import logging
import os
from matplotlib import font_manager
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
from matplotlib import font_manager

# ============================================================================
# 基础设置预定义
# ============================================================================

# 字体设置
font_manager.fontManager.addfont('C:\\Windows\\Fonts\\ARIAL.TTF')
plt.rcParams['font.family'] = 'Arial'


# ----------------------------
# 文件配置
# ----------------------------
CSV_FILE = "c_chain_atom_plddts_alphavaccMCTSdenovo.csv"
OUTPUT_PNG = "c_chain_plddt_vs_netmhc_scatter_with_marginals_filtered_alphavaccMCTSdenovo_new.png"
LOG_FILE = "C_chain_plddt_vs_netmhc_with_marginals_filtered_alphavaccMCTSdenovo.log"

# CSV_FILE = "c_chain_atom_plddts_pepmlmdenovo.csv"
# OUTPUT_PNG = "c_chain_plddt_vs_netmhc_scatter_with_marginals_filtered_pepmlmdenovo_new.png"
# LOG_FILE = "C_chain_plddt_vs_netmhc_with_marginals_filtered_pepmlmdenovo.log"

# CSV_FILE = "c_chain_atom_plddts_pepPPOdenovo.csv"
# OUTPUT_PNG = "c_chain_plddt_vs_netmhc_scatter_with_marginals_filtered_pepPPOdenovo_new.png"
# LOG_FILE = "C_chain_plddt_vs_netmhc_with_marginals_filtered_pepPPOdenovo.log"

# ----------------------------
# 主要绘图参数（可按需修改）
# ----------------------------
POINT_COLOR = "#d9534f"   # 红色统一颜色，便于修改
# POINT_COLOR = "#1f77b4"  # 🔹蓝色自定义点颜色，方便后续自行修改
# POINT_COLOR = (244/255,179/255,74/255)   # 橙色
FIG_SIZE = (6, 6)

# 主图范围
X_RANGE = (90, 100)
Y_RANGE = (0, 10)

# 边际直方图 bin 尺寸
BIN_X = 0.5
BIN_Y = 0.5

# 主图刻度设置
XTICKS = list(range(90, 101, 2))   # 每 2 一个刻度
YTICKS = list(range(0, 11, 2))     # 每 2 一个刻度
TICK_LEN = 5
TICK_FONTSIZE = 21


# ----------------------------
# 日志配置
# ----------------------------
def setup_logging():
    logging.basicConfig(
        filename=LOG_FILE,
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter("%(levelname)s | %(message)s")
    console.setFormatter(formatter)
    logging.getLogger().addHandler(console)


def main():
    setup_logging()
    logging.info("=== 开始绘制：筛选 + 散点 + 边际柱状图 ===")

    if not os.path.exists(CSV_FILE):
        logging.error(f"未找到输入文件：{CSV_FILE}")
        return

    try:
        df = pd.read_csv(CSV_FILE)
    except Exception as e:
        logging.error(f"读取 CSV 文件失败: {e}")
        return

    required = ["c_chain_atom_plddts_mean", "netmhc_nM"]
    if not all(c in df.columns for c in required):
        logging.error("CSV 文件中缺少必要列：c_chain_atom_plddts_mean / netmhc_nM")
        return

    # 去缺失
    df = df.dropna(subset=required)

    # -------- 1) 预筛选并统计 --------
    total_n = len(df)
    df_filt = df[
        (df["c_chain_atom_plddts_mean"] >= 90) & (df["c_chain_atom_plddts_mean"] <= 100) &
        (df["netmhc_nM"] >= 0) & (df["netmhc_nM"] <= 10)
    ].copy()
    sel_n = len(df_filt)
    pct = (sel_n / total_n * 100.0) if total_n > 0 else 0.0

    msg = f"筛选后数据点: {sel_n}/{total_n} ({pct:.2f}%)  [条件：pLDDT 90–100 且 nM 0–10]"
    print(msg)
    logging.info(msg)

    # 若没有数据也继续生成空图，避免脚本中断
    x = df_filt["c_chain_atom_plddts_mean"].values if sel_n > 0 else np.array([])
    y = df_filt["netmhc_nM"].values if sel_n > 0 else np.array([])

    # 生成与坐标范围对齐的 bins
    x_bins = np.arange(X_RANGE[0], X_RANGE[1] + BIN_X, BIN_X)
    y_bins = np.arange(Y_RANGE[0], Y_RANGE[1] + BIN_Y, BIN_Y)

    # ----------------------------
    # 2) 主图：散点
    # ----------------------------
    fig, ax = plt.subplots(figsize=FIG_SIZE)

    ax.scatter(x, y, s=40, color=POINT_COLOR, alpha=0.7, edgecolors="none")

    ax.plot(
        [91, 99],     # X: 左到右
        [9, 1],     # Y: 上到下
        linestyle="--", 
        color="black", 
        linewidth=1.5, 
        alpha=0.6
    )

    ax.set_xlim(X_RANGE)
    ax.set_ylim(Y_RANGE)
    ax.set_xticks(XTICKS)
    ax.set_yticks(YTICKS)
    ax.tick_params(axis='both', which='major', length=TICK_LEN, labelsize=TICK_FONTSIZE)

    # 仅保留下/左脊线
    for spine in ['top', 'right']:
        ax.spines[spine].set_visible(False)

    # 无网格、无标题/标签
    ax.grid(False)
    ax.set_title("")
    ax.set_xlabel("")
    ax.set_ylabel("")

    # ----------------------------
    # 3) 边际柱状图：去掉“坐标”（刻度与数字），但保留坐标轴线
    # ----------------------------
    divider = make_axes_locatable(ax)
    ax_histx = divider.append_axes("top", 0.8, pad=0.12, sharex=ax)
    ax_histy = divider.append_axes("right", 0.8, pad=0.12, sharey=ax)

    # 上方直方图
    ax_histx.hist(x, bins=x_bins, color=POINT_COLOR, alpha=0.8, edgecolor="none")
    # 右侧直方图（水平）
    ax_histy.hist(y, bins=y_bins, orientation='horizontal', color=POINT_COLOR, alpha=0.8, edgecolor="none")

    # —— 上方直方图：仅保留下边框（坐标轴线），去掉刻度与数字
    for spine in ["left", "right", "top"]:
        ax_histx.spines[spine].set_visible(False)
    ax_histx.spines["bottom"].set_visible(True)  # 保留下边轴线
    ax_histx.tick_params(axis='both', which='both', length=0, labelbottom=False, labelleft=False, labelright=False, labeltop=False)
    ax_histx.set_ylabel("")
    ax_histx.grid(False)

    # —— 右侧直方图：仅保留左边框（坐标轴线），去掉刻度与数字
    for spine in ["right", "top", "bottom"]:
        ax_histy.spines[spine].set_visible(False)
    ax_histy.spines["left"].set_visible(True)  # 保留左边轴线
    ax_histy.tick_params(axis='both', which='both', length=0, labelbottom=False, labelleft=False, labelright=False, labeltop=False)
    ax_histy.set_xlabel("")
    ax_histy.grid(False)

    plt.tight_layout()
    plt.savefig(OUTPUT_PNG, dpi=300)
    plt.close()

    logging.info(f"已生成图像: {OUTPUT_PNG}")

    # -------- 4) 统计阈值内数据点的平均值与误差 --------
    if sel_n > 0:
        # 计算 c_chain_atom_plddts_mean 的统计信息
        x_mean = np.mean(x)
        x_std = np.std(x, ddof=1)  # 样本标准差
        x_sem = x_std / np.sqrt(sel_n)  # 标准误差

        # 计算 netmhc_nM 的统计信息
        y_mean = np.mean(y)
        y_std = np.std(y, ddof=1)  # 样本标准差
        y_sem = y_std / np.sqrt(sel_n)  # 标准误差

        # 输出统计信息
        stats_msg = f"""
{'='*60}
阈值范围内数据点统计信息 (n={sel_n}):
{'='*60}
c_chain_atom_plddts_mean (横轴 X):
  - 平均值: {x_mean:.4f}
  - 标准差: {x_std:.4f}
  - 标准误差: {x_sem:.4f}

netmhc_nM (纵轴 Y):
  - 平均值: {y_mean:.4f}
  - 标准差: {y_std:.4f}
  - 标准误差: {y_sem:.4f}
{'='*60}
"""
        print(stats_msg)
        logging.info(stats_msg)
    else:
        logging.warning("没有数据点在阈值范围内，无法计算统计信息")

    logging.info(f"日志文件: {LOG_FILE}")
    logging.info("=== 完成 ===")


if __name__ == "__main__":
    main()
