#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
绘制散点图：
横坐标 = c_chain_atom_plddts_mean (88–100)
纵坐标 = netmhc_nM (0–30)
按照指定格式精简视觉样式。
"""

import pandas as pd
import matplotlib.pyplot as plt
import logging
import os
from matplotlib import font_manager

# ----------------------------
# 中文字体设置（防止乱码）
# ----------------------------
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# ----------------------------
# 文件配置
# ----------------------------
CSV_FILE = "c_chain_atom_plddts_alphavaccMCTSdenovo.csv"
OUTPUT_PNG = "c_chain_plddt_vs_netmhc_scatter_alphavaccMCTSdenovo.png"
# CSV_FILE = "c_chain_atom_plddts_pepmlmdenovo.csv"
# OUTPUT_PNG = "c_chain_plddt_vs_netmhc_scatter_pepmlmdenovo.png"
LOG_FILE = "C_chain_plddt_vs_netmhc.log"


# ----------------------------
# 绘图参数
# ----------------------------
# POINT_COLOR = "#1f77b4"  # 🔹自定义点颜色，方便后续自行修改
POINT_COLOR = "#BF092F"  #   自定义点颜色，方便后续自行修改
FIG_SIZE = (6, 5)
X_RANGE = (88, 98)
Y_RANGE = (0, 30)


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
# 主程序
# ----------------------------
def main():
    setup_logging()
    logging.info("=== 开始绘制精简版散点图 ===")

    if not os.path.exists(CSV_FILE):
        logging.error(f"未找到输入文件：{CSV_FILE}")
        return

    try:
        df = pd.read_csv(CSV_FILE)
    except Exception as e:
        logging.error(f"读取 CSV 文件失败: {e}")
        return

    if "c_chain_atom_plddts_mean" not in df.columns or "netmhc_nM" not in df.columns:
        logging.error("CSV 文件中缺少必要列。")
        return

    df = df.dropna(subset=["c_chain_atom_plddts_mean", "netmhc_nM"])

    # ----------------------------
    # 绘图部分
    # ----------------------------
    fig, ax = plt.subplots(figsize=FIG_SIZE)

    # 散点图
    ax.scatter(
        df["c_chain_atom_plddts_mean"],
        df["netmhc_nM"],
        s=40,
        color=POINT_COLOR,
        alpha=0.7,
        edgecolors="none"  # ✅ 去掉边框
    )

    # 坐标范围
    ax.set_xlim(X_RANGE)
    ax.set_ylim(Y_RANGE)

    # 设置刻度
    ax.set_xticks(range(88, 99, 2))
    ax.set_yticks(range(0, 31, 5))

    # 刻度样式：字体、线长
    ax.tick_params(axis='both', which='major', length=2, labelsize=18)

    # 去掉顶部和右侧边框，仅保留下和左
    for spine in ['top', 'right']:
        ax.spines[spine].set_visible(False)

    # ✅ 去掉网格
    ax.grid(False)

    # ✅ 去掉标签和标题
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_title("")

    # ✅ 不显示图例
    ax.legend().set_visible(False)

    plt.tight_layout()
    plt.savefig(OUTPUT_PNG, dpi=300)
    plt.close()

    logging.info(f"已生成精简散点图: {OUTPUT_PNG}")
    logging.info(f"日志文件: {LOG_FILE}")
    logging.info("=== 绘图完成 ===")


if __name__ == "__main__":
    main()
