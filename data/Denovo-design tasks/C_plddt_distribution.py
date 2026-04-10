#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
读取 c_chain_atom_plddts.csv，统计所有 C 链 atom_pLDDT 的分布，
并绘制 80–100 范围内的直方图 + KDE 平滑曲线，标出均值与 ±1σ 区域。
"""

import json
import os
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ----------------------------
# 字体与图形全局设置（中文显示 & 负号）
# ----------------------------
plt.rcParams['font.sans-serif'] = ['SimHei']   # Windows: SimHei / Microsoft YaHei；macOS 可用 PingFang SC
plt.rcParams['axes.unicode_minus'] = False

# ----------------------------
# 配置部分
# ----------------------------
# 输入输出
# CSV_FILE   = "c_chain_atom_plddts_alphavaccMCTSdenovo.csv"
# OUTPUT_PNG = "c_chain_plddt_distribution_alphavaccMCTSdenovo.png"
CSV_FILE   = "c_chain_atom_plddts_pepmlmdenovo.csv"
OUTPUT_PNG = "c_chain_plddt_distribution_pepmlmdenovo.png"
LOG_FILE   = "C_chain_plddt_distribution.log"

# 绘图参数
X_RANGE   = (80, 100)   # 横坐标范围
BIN_WIDTH = 0.2         # bin 宽度（越小越细）
FIG_SIZE  = (10, 6)

# KDE 参数（None = 自动 Silverman 带宽；或手动设一个正数，如 0.5）
KDE_BW = None

# 纵轴范围（概率密度）。若不想固定，可置为 None 自动撑满
Y_LIM = (0, 1)


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


# ----------------------------
# 数据读取与解析
# ----------------------------
def load_plddt_values(csv_file: str, column: str = "c_chain_atom_plddts") -> np.ndarray:
    """从 CSV 读取 c_chain_atom_plddts 列，解析为一维 float 数组。"""
    df = pd.read_csv(csv_file)
    if column not in df.columns:
        raise KeyError(f"CSV 文件中未找到列 '{column}'")

    all_values = []
    for i, val in enumerate(df[column]):
        try:
            # 每行是一个 list 的 JSON 字符串
            arr = json.loads(val)
            if isinstance(arr, list):
                # 仅收集数字
                for x in arr:
                    if isinstance(x, (int, float)):
                        all_values.append(float(x))
        except Exception as e:
            logging.warning(f"第 {i} 行解析失败: {e}")

    if not all_values:
        raise ValueError("未提取到任何数值，请检查 CSV 内容。")

    return np.array(all_values, dtype=float)


# ----------------------------
# KDE（高斯核）—— 纯 NumPy 实现
# ----------------------------
def gaussian_kde_numpy(samples: np.ndarray, xs: np.ndarray, bw: float | None = None) -> tuple[np.ndarray, float]:
    """
    基于高斯核的核密度估计（KDE），纯 NumPy 实现。
    - samples: 样本数据（1D）
    - xs:      需要计算密度的网格点（1D）
    - bw:      带宽；None 则用 Silverman 经验带宽
    返回 (kde_y, h)，kde_y 为每个 xs 的密度，h 为使用的带宽。
    """
    n = len(samples)
    std = np.std(samples)
    # Silverman 经验带宽；若 std=0，则给一个很小的带宽避免除零
    silverman_bw = 1.06 * std * (n ** (-1/5)) if std > 0 else 1e-3
    h = bw if (bw is not None and bw > 0) else silverman_bw

    # 计算核密度：平均的高斯核之和（广播）
    u = (xs[None, :] - samples[:, None]) / h
    kde_y = np.exp(-0.5 * u * u).sum(axis=0) / (n * h * np.sqrt(2 * np.pi))
    return kde_y, h


# ----------------------------
# 主流程
# ----------------------------
def main():
    setup_logging()
    logging.info("=== 开始绘制 C 链 pLDDT 分布图（80–100 区间） ===")

    if not os.path.exists(CSV_FILE):
        logging.error(f"未找到输入文件：{CSV_FILE}")
        return

    try:
        values = load_plddt_values(CSV_FILE, "c_chain_atom_plddts")
    except Exception as e:
        logging.error(f"数据读取/解析失败：{e}")
        return

    # 基本统计
    mean_val = float(np.mean(values))
    std_val  = float(np.std(values))
    n_val    = len(values)
    logging.info(f"统计结果: 样本数={n_val}, 均值={mean_val:.3f}, 方差(σ)={std_val:.3f}")

    # 准备绘图
    bins = np.arange(X_RANGE[0], X_RANGE[1] + BIN_WIDTH, BIN_WIDTH)
    plt.figure(figsize=FIG_SIZE)

    # 直方图（密度归一，轮廓线）
    counts, bins, patches = plt.hist(
        values, bins=bins, density=True, histtype="step", linewidth=2, label="pLDDT 直方图（密度）"
    )

    # KDE 平滑曲线（在指定区间上计算）
    xs = np.linspace(X_RANGE[0], X_RANGE[1], 1200)
    kde_y, used_bw = gaussian_kde_numpy(values, xs, KDE_BW)
    plt.plot(xs, kde_y, linewidth=2, label=f"KDE（h={used_bw:.3f}）")

    # 均值线与 ±1σ 区域
    plt.axvline(mean_val, color="red", linestyle="--", linewidth=2, label=f"均值 = {mean_val:.2f}")
    plt.axvspan(mean_val - std_val, mean_val + std_val, color="gray", alpha=0.2, label=f"±1σ = {std_val:.2f}")

    # 坐标与标签
    plt.xlim(X_RANGE)
    if Y_LIM is not None:
        plt.ylim(*Y_LIM)
    plt.xlabel("C 链 pLDDT 值 (80–100)", fontsize=12)
    plt.ylabel("频率密度 (0–1)", fontsize=12)
    plt.title("C 链 pLDDT 分布曲线（放大 80–100）", fontsize=14)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()

    # 右上角文字标注
    # 为避免遮挡 KDE 峰值，这里将 y 放在轴顶端 90% 处（若未固定 Y_LIM，则取当前轴上限的 90%）
    ymax = plt.gca().get_ylim()[1]
    plt.text(mean_val + 0.5, ymax * 0.9, f"均值={mean_val:.2f}\nσ={std_val:.2f}\nN={n_val}", color="red", fontsize=11)

    plt.tight_layout()
    plt.savefig(OUTPUT_PNG, dpi=300)
    plt.close()

    logging.info(f"已生成分布图: {OUTPUT_PNG}")
    logging.info(f"详细日志写入: {LOG_FILE}")
    logging.info("=== 绘图完成 ===")


if __name__ == "__main__":
    main()
