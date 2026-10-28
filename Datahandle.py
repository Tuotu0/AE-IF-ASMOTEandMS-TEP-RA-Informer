"""2026-10-20 纯流量窗口版：19类；前置Normal:Attack=1.5:1。
保留读取→映射→融合→早期切分→采样→预处理→AE→IF→自适应过采样主链。
flow_uid仅作为关联旁表的元数据；预测使用同一个冻结的主融合处理器。
"""
import os
import sys
import re
import logging
import warnings
import time
import json
import threading
import hashlib
import random
import shutil
import gzip
import tempfile
import copy
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import psutil
from datetime import datetime
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, BatchSampler, RandomSampler, SequentialSampler
from sklearn.preprocessing import StandardScaler, RobustScaler, FunctionTransformer
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import pairwise_distances
from sklearn.ensemble import IsolationForest
from imblearn.over_sampling import SMOTE, KMeansSMOTE
from scipy import stats
from sklearn.exceptions import DataConversionWarning
import platform

print("当前运行目录：", os.getcwd())

# 配置警告抑制
warnings.filterwarnings("ignore", category=DataConversionWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

# -------------------------- 1. 配置初始化 --------------------------

# 配置matplotlib中文字体支持，防止乱码
def setup_chinese_font():
    """配置matplotlib中文字体"""
    system = platform.system()
    if system == "Windows":
        # Windows系统常用中文字体
        chinese_fonts = ['Microsoft YaHei', 'SimHei', 'SimSun', 'KaiTi']
    elif system == "Darwin":  # macOS
        chinese_fonts = ['PingFang SC', 'STHeiti', 'Heiti TC']
    else:  # Linux
        chinese_fonts = ['WenQuanYi Micro Hei', 'Droid Sans Fallback', 'Noto Sans CJK']
    
    for font in chinese_fonts:
        try:
            matplotlib.rcParams['font.sans-serif'] = [font] + matplotlib.rcParams['font.sans-serif']
            break
        except:
            continue
    
    # 解决负号显示问题
    matplotlib.rcParams['axes.unicode_minus'] = False
    
    # 设置默认字体大小
    matplotlib.rcParams['font.size'] = 10
    matplotlib.rcParams['axes.titlesize'] = 14
    matplotlib.rcParams['axes.labelsize'] = 12
    matplotlib.rcParams['xtick.labelsize'] = 10
    matplotlib.rcParams['ytick.labelsize'] = 10
    matplotlib.rcParams['legend.fontsize'] = 10
    matplotlib.rcParams['figure.dpi'] = 120
    matplotlib.rcParams['savefig.dpi'] = 350
    matplotlib.rcParams['axes.linewidth'] = 0.8
    matplotlib.rcParams['grid.linewidth'] = 0.5
    matplotlib.rcParams['grid.alpha'] = 0.25
    sns.set_theme(style="whitegrid", context="paper")

setup_chinese_font()

# 日志配置
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(process)d - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# 打印启动信息
print("=== 数据处理脚本启动 ===")
print(f"当前时间: {datetime.now()}")
print(f"Python版本: {sys.version}")


def clean_mangled_label(label: str) -> str:
    """清理标签中的乱码字符
    
    常见乱码模式：
    - � (替换字符) 
    - 编码错误的SQL注入等
    """
    if not isinstance(label, str):
        return label
    
    import re
    
    label = label.replace('\ufffd', ' ')
    label = label.replace('�', ' ')
    
    label = re.sub(r'\s+', ' ', label)
    
    label = label.strip()
    
    return label


# 分层采样函数


# 计时装饰器
def timer_decorator(func):
    """函数执行时间统计装饰器"""
    
    def wrapper(*args, **kwargs):
        start_time = time.time()
        rss_before = psutil.Process().memory_info().rss
        logger.info(f"开始执行：{func.__name__}")
        try:
            result = func(*args, **kwargs)
            elapsed_time = time.time() - start_time
            logger.info(f"完成执行：{func.__name__}，耗时：{elapsed_time:.2f}秒 ({elapsed_time / 60:.2f}分钟)")
            return result
        except Exception as e:
            elapsed_time = time.time() - start_time
            logger.error(f"执行失败：{func.__name__}，耗时：{elapsed_time:.2f}秒，错误：{str(e)}")
            raise
        finally:
            # 不重置GPU峰值、不做同步，避免测量改变执行方式；RSS为起止采样而非精确峰值。
            try:
                cfg = globals().get("config")
                if cfg is not None:
                    if not hasattr(cfg, "_performance_records"):
                        cfg._performance_records = []
                    cfg._performance_records.append({
                        "stage": func.__name__, "seconds": time.time() - start_time,
                        "rss_before_mb": rss_before / 1024**2,
                        "rss_after_mb": psutil.Process().memory_info().rss / 1024**2,
                        "cuda_peak_allocated_mb_since_process_start": (
                            torch.cuda.max_memory_allocated() / 1024**2
                            if torch.cuda.is_initialized() else None)})
            except Exception as exc:
                logger.warning("性能统计记录失败: %s", exc)
    
    return wrapper


def normalize_col(col):
    """统一字段标准化函数 - 兼容CICFlowMeter长/短字段
    
    标准化操作：下划线/连字符转空格，strip + lower + 压缩空格
    兼容：大小写差异、多空格、前后空格、tab、连续空白字符、CICFlowMeter长字段/短字段
    """
    col = str(col)
    col = col.replace('_', ' ')
    col = col.replace('-', ' ')
    col = col.strip().lower()
    col = re.sub(r'\s+', ' ', col)
    return col


def resolve_dataset_dir(folder_name: str, env_var: str, windows_default: str) -> str:
    """自动解析数据集目录，兼容 Windows 路径和 Linux/容器相对路径。

    优先级：
    1. 环境变量：CIC_IDS_2017_DIR / CIC_IDS_2018_DIR
    2. 当前运行目录下的 folder_name，例如 /root/CIC-IDS-2017
    3. 脚本所在目录下的 folder_name
    4. 原 Windows 默认路径（主要给 Windows 本地运行保留）
    5. 如果都不存在，返回当前运行目录下的 folder_name，便于报错信息直观
    """
    candidates = []

    env_path = os.environ.get(env_var)
    if env_path:
        candidates.append(env_path)

    cwd = os.getcwd()
    candidates.append(os.path.join(cwd, folder_name))

    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        candidates.append(os.path.join(script_dir, folder_name))
    except Exception:
        pass

    candidates.append(windows_default)

    # 去重并返回第一个真实存在的目录
    seen = set()
    for candidate in candidates:
        if not candidate:
            continue
        candidate = os.path.abspath(os.path.expanduser(candidate)) if not re.match(r"^[A-Za-z]:\\", str(candidate)) else candidate
        if candidate in seen:
            continue
        seen.add(candidate)
        if os.path.isdir(candidate):
            return candidate

    # 全部不存在时，返回最适合 Linux/容器运行的默认路径
    return os.path.join(os.getcwd(), folder_name)


# 配置类
class Config:
    # 数据集路径
    # Linux/容器中会自动识别当前目录或脚本目录下的 CIC-IDS-2017 / CIC-IDS-2018。
    # 也可以用环境变量覆盖：
    #   export CIC_IDS_2017_DIR=/root/CIC-IDS-2017
    #   export CIC_IDS_2018_DIR=/root/CIC-IDS-2018
    FOLDER_2017 = resolve_dataset_dir(
        "CIC-IDS-2017",
        "CIC_IDS_2017_DIR",
        r"/root/CIC-IDS-2017",
    )
    FOLDER_2018 = resolve_dataset_dir(
        "CIC-IDS-2018",
        "CIC_IDS_2018_DIR",
        r"/root/CIC-IDS-2018",
    )
    
    # 主输出文件夹（【修改说明】新名称，避免与原代码输出混淆）
    MAIN_OUTPUT_DIR = "/root/shared-nvme/timeseries_processing_results"
    
    def __init__(self):
        # 默认服务器路径不变；允许上传后用环境变量指定输出位置。
        self.MAIN_OUTPUT_DIR = os.environ.get("DATAHANDLE_OUTPUT_ROOT", self.MAIN_OUTPUT_DIR)
        # 生成带时间戳的输出路径
        self.TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        self.OUTPUT_DIR = os.path.join(self.MAIN_OUTPUT_DIR, f"run_{self.TIMESTAMP}")
        self.OUTPUT_TRAIN = os.path.join(self.OUTPUT_DIR, "train_dataset.csv")
        self.OUTPUT_TEST = os.path.join(self.OUTPUT_DIR, "test_dataset.csv")
        self.OUTPUT_BALANCED = os.path.join(self.OUTPUT_DIR, "balanced_dataset.csv")
        self.OUTPUT_VISUALIZATION = os.path.join(self.OUTPUT_DIR, "visualization")
        self.OUTPUT_VAL = os.path.join(self.OUTPUT_DIR, "val_dataset.csv")
        self._forecast_row_parts = []
        self.LATEST_TRAIN = os.path.join(self.MAIN_OUTPUT_DIR, "train_dataset.csv")
        self.LATEST_VAL = os.path.join(self.MAIN_OUTPUT_DIR, "val_dataset.csv")
        self.LATEST_TEST = os.path.join(self.MAIN_OUTPUT_DIR, "test_dataset.csv")
        self.LATEST_BALANCED = os.path.join(self.MAIN_OUTPUT_DIR, "balanced_dataset.csv")
        self.LATEST_SUMMARY = os.path.join(self.MAIN_OUTPUT_DIR, "latest_run_summary.txt")
        
        # 标准化 FEATURE_MAPPING 的所有 key（统一 normalize_col）
        self.FEATURE_MAPPING = {
            normalize_col(k): v
            for k, v in Config.FEATURE_MAPPING.items()
        }
    
    # 处理参数（120G内存服务器优化版：全量读取 + 以内存换速度）
    SEED = 42

    # 读取/合并阶段：不再按文件大小随机采样，直接全量读取；一次性concat，减少反复concat开销
    FULL_READ_CSV = True
    READ_MEMORY_MAP = True
    CSV_READ_WORKERS = 4  # 同时处理的CSV文件数；限制并发，避免大文件造成内存峰值过高

    # 读取/列处理加速：避免在大表上执行不必要的整表复制和逐行字符串扫描
    SKIP_MALFORMED_ROW_SCAN = True
    FAST_NUMERIC_CONVERSION_SAMPLE = 10000
    # 大表性能优化：映射阶段只抽样判断object列是否“看起来像数值”，不立即全列to_numeric。
    # 真正的数值转换延迟到Train/Val/Test切分后，由对应特征处理函数按需完成。
    # 这样可避免2018千万级数据在映射前和特征分类阶段被重复全表扫描。
    DEFER_LARGE_OBJECT_NUMERIC_CONVERSION = True
    DEFER_NUMERIC_TO_TRAIN = True  # 映射阶段不根据全量分布改变数值/文本字段
    PRE_MAPPING_FULL_NUMERIC_CONVERSION_MAX_ROWS = 2_000_000
    FEATURE_CLASSIFY_SAMPLE_SIZE = 50000

    # 全局样本上限：仅用于SMOTE/AE前的安全阀，不影响CSV全量读取和初始合并
    # 120G内存下，200万行 * 百级特征在pandas/NumPy中通常仍有余量；若服务器还有空余可继续上调
    MAX_SAMPLES = 2_000_000

    # RobustScaler服务器优化：统计量仍然只来自训练集；超大训练集仅用训练集随机子集估计中位数/IQR，避免全量分位数排序造成长时间卡顿/峰值OOM。
    # 这不会引入验证集/测试集信息，因此仍满足无数据泄露要求。
    ROBUST_SCALER_FIT_MAX_SAMPLES = 2_000_000
    ROBUST_SCALER_TRANSFORM_CHUNK_SIZE = 250_000

    # 内存阈值：高于0.92只报警和释放缓存，不再自动砍半采样；高于0.97才触发强保护
    MAX_MEMORY_USAGE = 0.92
    CRITICAL_MEMORY_USAGE = 0.97

    TRAIN_TEST_SPLIT = 0.8
    VAL_SPLIT = 0.1
    TEST_SPLIT = 0.1
    # 预测只采用文件内部原始行顺序，每窗恰好500条有效真实流量。
    FORECAST_SEQ_LEN = 48
    FORECAST_LABEL_LEN = 24
    FORECAST_PRED_LEN = 12
    FORECAST_BRIDGE_OOF_FOLDS = 3
    FORECAST_SPLIT_SOLVER_SECONDS = 60
    IO_CHUNK_ROWS = 100000
    MEMORY_BUDGET_FRACTION = 0.90
    # 仅控制计算/缓存粒度，不改变样本、窗口定义、模型或采样规则。
    MMD_BLOCK_SIZE = 1024
    OPTIMIZATION_REVISION = "flow_windows_20261020_v1"
    EXPORT_FUSED_FORECAST = True
    FORECAST_ROWS_PER_WINDOW = 500
    FORECAST_EXPORT_CHUNK_SIZE = 100000
    # 分类/预测共用在fit前确定的片段内顺序划分；不重新定义融合或特征处理步骤。
    FORECAST_UNIFIED_SPLIT = True
    # 用户选择：优先训练类别覆盖，按完整窗口扩展连续训练前缀；比例按实际值审计。
    FORECAST_TRAIN_COVERAGE_POLICY = "global_ordered"
    # 审查纠错版：需要改变实验定义的配置在说明文档中单独列明。
    AMBIGUOUS_LABEL_POLICY = "separate_unknown"  # 用户决定：分别新增DDoS_Unknown/DoS_Unknown
    SOURCE_LABEL_OVERRIDES = {"2017": {}, "2018": {}}
    DUPLICATE_SPLIT_MODE = "group"  # group / audit；由用户确认
    MIXED_RESAMPLING_MODE = "inherit"  # inherit / smotenc；由用户确认
    AE_INPUT_SCALE_MODE = "robust"  # robust / standard；由用户确认
    FEATURE_UNIT_MULTIPLIERS = {"2017": {}, "2018": {}}  # 已核实的单位换算，默认不猜测
    FEATURE_TYPE_OVERRIDES = {}  # canonical列名 -> continuous/discrete/port
    STRICT_INPUT_FILES = True
    DISK_FREE_MIN_GB = 2.0
    OUTPUT_SIZE_SAFETY_FACTOR = 1.5
    BALANCED_ALIAS_MODE = "hardlink"  # hardlink节省磁盘；copy生成可独立编辑的副本
    EXPORT_HEAVY_VISUALS = True
    DISTANCE_ANALYSIS_ENABLED = True
    DISTANCE_SAMPLE_SIZE = 5000  # 仅诊断统计；不删除任何训练样本
    DETERMINISTIC_TORCH = False  # 显式可选；固定随机种子不等于跨硬件逐位一致
    # 分布距离/可视化采样：相比原来的5k/10k明显放大，但避免MMD二次矩阵无意义膨胀
    MMD_SAMPLE_SIZE = 15000
    WASSERSTEIN_SAMPLE_SIZE = 50000
    PLOT_SAMPLE_SIZE = 20000

    # AutoEncoder：服务器增强版。
    # AE_EPOCHS设置为上限，不再固定跑满；Early Stopping会在验证集不再改善时自动停止。
    AE_BATCH_SIZE = 4096
    AE_EPOCHS = 100
    AE_LATENT_MAX = 32
    AE_LEARNING_RATE = 0.001
    AE_WEIGHT_DECAY = 1e-5
    AE_VAL_SPLIT = 0.1
    AE_EARLY_STOPPING_PATIENCE = 12
    AE_MIN_DELTA = 1e-4
    AE_LR_SCHEDULER_PATIENCE = 5
    AE_LR_SCHEDULER_FACTOR = 0.5
    AE_DROPOUT = 0.05
    AE_GRAD_CLIP_NORM = 5.0
    AE_USE_LAYER_NORM = True
    AE_EXPORT_TRAINING_CURVE = True

    # 前置采样真实来源目标：2018:2017=0.7:0.3；此参数始终表示2017占比。
    SOURCE_BALANCE_TARGET = 0.3
    SOURCE_BALANCE_TOLERANCE = 0.05
    # 数据源平衡只能是软约束，不能以破坏类别先验为代价。
    # 前置联合采样目标：Normal:全部Attack=1.5:1；不设攻击总量上限。
    NORMAL_MAX_ATTACK_RATIO = 1.5
    VISUALIZATION_DPI = 350

    # 平衡/异常检测参数
    IF_CONTAMINATION = 0.01
    KMEANS_SMOTE_CLUSTERS = 10
    SMOTE_MIN_REQUIRED = 5

    # 论文版自适应采样策略：默认不再依赖每个类别的大整数硬编码目标。
    # 思路：根据当前类别分布自动生成过采样目标，并限制最大增强倍率，避免极小类被人工合成样本淹没。
    BALANCING_TARGET_MODE = "adaptive"  # adaptive / manual
    ADAPTIVE_BALANCE_TARGET_TOTAL = 650_000
    # 小类优先过采样策略：只对少数攻击类做明显增强，大类默认不增强。
    # 说明：PROTECTED_CLASSES只用于“不删成0”，不再表示“必须增强”。
    ADAPTIVE_ATTACK_MIN_TARGET = 2_000          # 小类的基础补强目标
    ADAPTIVE_TINY_CLASS_TARGET = 100            # <SMOTE_MIN_REQUIRED 的极小类随机复制目标上限
    ADAPTIVE_SMALL_CLASS_THRESHOLD = 2_000      # <2000：小类，优先增强
    ADAPTIVE_MEDIUM_CLASS_THRESHOLD = 5_000     # 2000~4999：中等类，只允许轻微增强
    ADAPTIVE_SMALL_CLASS_MAX_MULTIPLIER = 4.0   # 小类最高增强倍率，避免8倍增强造成误报激增
    ADAPTIVE_MEDIUM_CLASS_MAX_MULTIPLIER = 1.2  # 中等类最高增强倍率
    ADAPTIVE_LARGE_CLASS_MAX_MULTIPLIER = 1.2   # 大类若允许增强，最高也只到1.2倍
    ADAPTIVE_DISABLE_LARGE_CLASS_OVERSAMPLING = True  # >=5000的大类默认不做SMOTE增强
    ADAPTIVE_TINY_CLASS_MAX_MULTIPLIER = 20.0
    ADAPTIVE_MAX_SYNTHETIC_RATIO = 0.60
    # 温和过采样约束：最终攻击总量不应反超Normal，单个攻击类也不应被补得过大
    ADAPTIVE_MAX_ATTACK_TO_NORMAL_RATIO = 0.70
    ADAPTIVE_MAX_SINGLE_ATTACK_TO_NORMAL_RATIO = 0.18
    # 当当前攻击总量已超过上限时，仍允许“极小类/小类”使用一个很小的兜底预算做增强。
    # 这样可以避免 DDoS/DoS 大类占满攻击总量后，SQL/XSS/LOIC_UDP/Heartbleed 等小类完全得不到增强。
    ADAPTIVE_ALLOW_SMALL_CLASS_OVERSAMPLING_OVER_ATTACK_CAP = True
    ADAPTIVE_SMALL_CLASS_OVER_CAP_EXTRA_BUDGET = 10_000
    ADAPTIVE_SMALL_CLASS_OVER_CAP_NORMAL_RATIO = 0.10
    EXPORT_SAMPLING_AUDIT_PLOTS = True

    # 论文增强可视化：记录各处理阶段的小样本快照和统计摘要，生成前后对比/比例/变化率图
    EXPORT_COMPREHENSIVE_STAGE_VISUALS = True
    VIZ_STAGE_SAMPLE_SIZE = 50000
    VIZ_FEATURE_TOP_N = 12

    # 标签强约束：最终及中间所有Label必须严格落在 STANDARD_LABELS 的配置中的标准类别内

    # 读取阶段速度优化：保持严格配置中的标准类别约束，但用唯一标签映射+向量化过滤，减少逐行Python开销
    LABEL_AUDIT_TOP_N = 200
    OPTIMIZE_OBJECT_CATEGORY_ON_READ = False

    
    # 攻击类保护机制 - 只要原始合并数据中出现过，后续任何阶段都不允许被删成0
    # 注意：Normal可以按比例下采样；16个攻击类全部强制保护。
    PROTECTED_CLASSES = [
        "FTP_BruteForce",
        "SSH_BruteForce",
        "DoS_Hulk",
        "DoS_GoldenEye",
        "DoS_Slowloris",
        "DoS_SlowHTTPTest",
        "DDoS_LOIC_HTTP",
        "DDoS_LOIC_UDP",
        "DDoS_HOIC",
        "Web_SQL_Injection",
        "Web_XSS",
        "Web_BruteForce",
        "Infiltration",
        "Botnet",
        "Heartbleed",
        "PortScan",
    ]
    
    # 异常检测阈值 - 样本数大于此值才执行IF/LOF
    
    # 手动采样策略 - 各攻击类别目标数量
    # 基于CSE-CIC-IDS2017/2018官方流量占比调整
    SAMPLING_STRATEGY_MANUAL = {
        # 120G内存服务器版：整体约107.6万目标样本，约为原策略4倍
        "Normal": 200000,
        "DoS_Hulk": 120000,
        "DoS_GoldenEye": 80000,
        "DoS_Slowloris": 60000,
        "DoS_SlowHTTPTest": 60000,
        "DDoS_HOIC": 120000,
        "DDoS_LOIC_HTTP": 80000,
        "DDoS_LOIC_UDP": 80000,
        "PortScan": 100000,
        "FTP_BruteForce": 32000,
        "SSH_BruteForce": 32000,
        "Web_SQL_Injection": 20000,
        "Web_XSS": 20000,
        "Web_BruteForce": 20000,
        "Infiltration": 20000,
        "Botnet": 20000,
        "Heartbleed": 12000,
    }
    
    # 特征映射表 - 基于CSE-CIC-IDS2017/2018官方映射表
    # 数据集来源: https://www.unb.ca/cic/datasets/ids-2017.html
    #              https://www.unb.ca/cic/datasets/ids-2018.html
    # 映射规则: cse_cic_ids_2017_2018_feature_mapping.csv
    # 标准化规则: 所有key使用小写+标准空格格式(下划线转空格，全小写)
    FEATURE_MAPPING = {
        "source port": "src_port",
        "src port": "src_port",
        "srcport": "src_port",
        "protocol": "protocol",
        # ==================== 流基本信息 ====================
        "flow duration": "flow_duration",
        "flow duration": "flow_duration",
        "flowdur": "flow_duration",
        
        # ==================== 端口信息 ====================
        "destination port": "dst_port",
        "dst port": "dst_port",
        "dstport": "dst_port",
        
        # ==================== 数据包计数 ====================
        "total fwd packets": "total_fwd_packets",
        "tot fwd pkts": "total_fwd_packets",
        "totfwdpkts": "total_fwd_packets",
        
        "total backward packets": "total_bwd_packets",
        "tot bwd pkts": "total_bwd_packets",
        "totbwdpkts": "total_bwd_packets",
        
        # ==================== 总长度统计 ====================
        "total length of fwd packets": "total_length_fwd_packets",
        "totlen fwd pkts": "total_length_fwd_packets",
        
        "total length of bwd packets": "total_length_bwd_packets",
        "totlen bwd pkts": "total_length_bwd_packets",
        
        # ==================== Fwd数据包长度统计 ====================
        "fwd packet length max": "fwd_pkt_len_max",
        "fwd pkt len max": "fwd_pkt_len_max",
        "fwdpktlenmax": "fwd_pkt_len_max",
        
        "fwd packet length min": "fwd_pkt_len_min",
        "fwd pkt len min": "fwd_pkt_len_min",
        "fwdpktlenmin": "fwd_pkt_len_min",
        
        "fwd packet length mean": "fwd_pkt_len_mean",
        "fwd pkt len mean": "fwd_pkt_len_mean",
        "fwdpktlenmean": "fwd_pkt_len_mean",
        
        "fwd packet length std": "fwd_pkt_len_std",
        "fwd pkt len std": "fwd_pkt_len_std",
        "fwdpktlenstd": "fwd_pkt_len_std",
        
        # ==================== Bwd数据包长度统计 ====================
        "bwd packet length max": "bwd_pkt_len_max",
        "bwd pkt len max": "bwd_pkt_len_max",
        "bwdpktlenmax": "bwd_pkt_len_max",
        
        "bwd packet length min": "bwd_pkt_len_min",
        "bwd pkt len min": "bwd_pkt_len_min",
        "bwdpktlenmin": "bwd_pkt_len_min",
        
        "bwd packet length mean": "bwd_pkt_len_mean",
        "bwd pkt len mean": "bwd_pkt_len_mean",
        "bwdpktlenmean": "bwd_pkt_len_mean",
        
        "bwd packet length std": "bwd_pkt_len_std",
        "bwd pkt len std": "bwd_pkt_len_std",
        "bwdpktlenstd": "bwd_pkt_len_std",
        
        # ==================== 速率统计 ====================
        "flow bytes/s": "flow_bytes_per_s",
        "flow byts/s": "flow_bytes_per_s",
        "flowbyts/s": "flow_bytes_per_s",
        
        "flow packets/s": "flow_pkts_per_s",
        "flow pkts/s": "flow_pkts_per_s",
        "flowpkts/s": "flow_pkts_per_s",
        
        # ==================== IAT统计 ====================
        "flow iat mean": "flow_iat_mean",
        "flow iat std": "flow_iat_std",
        "flow iat max": "flow_iat_max",
        "flow iat min": "flow_iat_min",
        
        "fwd iat total": "fwd_iat_total",
        "fwd iat tot": "fwd_iat_total",
        
        "fwd iat mean": "fwd_iat_mean",
        
        "fwd iat std": "fwd_iat_std",
        
        "fwd iat max": "fwd_iat_max",
        
        "fwd iat min": "fwd_iat_min",
        
        "bwd iat total": "bwd_iat_total",
        "bwd iat tot": "bwd_iat_total",
        
        "bwd iat mean": "bwd_iat_mean",
        
        "bwd iat std": "bwd_iat_std",
        
        "bwd iat max": "bwd_iat_max",
        
        "bwd iat min": "bwd_iat_min",
        
        # ==================== 头部信息 ====================
        "fwd header length": "fwd_header_length",
        "fwd header len": "fwd_header_length",
        
        "bwd header length": "bwd_header_length",
        "bwd header len": "bwd_header_length",
        
        # ==================== 方向速率 ====================
        "fwd packets/s": "fwd_pkts_per_s",
        "fwd pkts/s": "fwd_pkts_per_s",
        "fwdpkts/s": "fwd_pkts_per_s",
        
        "bwd packets/s": "bwd_pkts_per_s",
        "bwd pkts/s": "bwd_pkts_per_s",
        "bwdpkts/s": "bwd_pkts_per_s",
        
        # ==================== 包长度统计（通用） ====================
        "max packet length": "max_pkt_len",
        "pkt len max": "max_pkt_len",
        "pktlenmax": "max_pkt_len",
        
        "min packet length": "min_pkt_len",
        "pkt len min": "min_pkt_len",
        "pktlenmin": "min_pkt_len",
        
        "packet length mean": "pkt_len_mean",
        "pkt len mean": "pkt_len_mean",
        "pktlenmean": "pkt_len_mean",
        
        "packet length std": "pkt_len_std",
        "pkt len std": "pkt_len_std",
        "pktlenstd": "pkt_len_std",
        
        "packet length variance": "pkt_len_var",
        "pkt len var": "pkt_len_var",
        "pktlenvar": "pkt_len_var",
        
        # ==================== 平均大小 ====================
        "average packet size": "avg_packet_size",
        "pkt size avg": "avg_packet_size",
        "pkt size avg": "avg_packet_size",
        
        "avg fwd segment size": "avg_fwd_segment_size",
        "fwd seg size avg": "avg_fwd_segment_size",
        
        "avg bwd segment size": "avg_bwd_segment_size",
        "bwd seg size avg": "avg_bwd_segment_size",
        
        # ==================== 上下行比率 ====================
        "down/up ratio": "down_up_ratio",
        "down/upratio": "down_up_ratio",
        
        # ==================== 子流信息 ====================
        "subflow fwd packets": "subflow_fwd_packets",
        "subflow fwd pkts": "subflow_fwd_packets",
        
        "subflow fwd bytes": "subflow_fwd_bytes",
        "subflow fwd byts": "subflow_fwd_bytes",
        
        "subflow bwd packets": "subflow_bwd_packets",
        "subflow bwd pkts": "subflow_bwd_packets",
        
        "subflow bwd bytes": "subflow_bwd_bytes",
        "subflow bwd byts": "subflow_bwd_bytes",
        
        # ==================== 标志位统计 ====================
        "fin flag count": "fin_flag_count",
        "fin flag cnt": "fin_flag_count",
        "fin flag cnt": "fin_flag_count",
        
        "syn flag count": "syn_flag_count",
        "syn flag cnt": "syn_flag_count",
        
        "rst flag count": "rst_flag_count",
        "rst flag cnt": "rst_flag_count",
        
        "psh flag count": "psh_flag_count",
        "psh flag cnt": "psh_flag_count",
        
        "ack flag count": "ack_flag_count",
        "ack flag cnt": "ack_flag_count",
        
        "urg flag count": "urg_flag_count",
        "urg flag cnt": "urg_flag_count",
        
        "ece flag count": "ece_flag_count",
        "ece flag cnt": "ece_flag_count",
        
        "cwe flag count": "cwe_flag_count",
        "cwe flag cnt": "cwe_flag_count",
        
        "fwd psh flags": "fwd_psh_flags",
        
        "bwd psh flags": "bwd_psh_flags",
        
        "fwd urg flags": "fwd_urg_flags",
        
        "bwd urg flags": "bwd_urg_flags",
        
        # ==================== 初始化窗口 ====================
        "init_win_bytes_forward": "init_fwd_win_bytes",
        "init fwd win byts": "init_fwd_win_bytes",
        
        "init_win_bytes_backward": "init_bwd_win_bytes",
        "init bwd win byts": "init_bwd_win_bytes",
        
        # ==================== 有效数据 ====================
        "act_data_pkt_fwd": "act_data_pkt_fwd",
        "fwd act data pkts": "act_data_pkt_fwd",
        
        "min_seg_size_forward": "min_seg_size_forward",
        "fwd seg size min": "min_seg_size_forward",
        
        # ==================== 活跃时间统计 ====================
        "active mean": "active_mean",
        "active std": "active_std",
        "active max": "active_max",
        "active min": "active_min",
        
        # ==================== 空闲时间统计 ====================
        "idle mean": "idle_mean",
        "idle std": "idle_std",
        "idle max": "idle_max",
        "idle min": "idle_min",
        
        # ==================== Bulk统计 ====================
        "fwd avg bytes/bulk": "avg_fwd_bytes_bulk",
        "fwd byts/b avg": "avg_fwd_bytes_bulk",
        
        "bwd avg bytes/bulk": "avg_bwd_bytes_bulk",
        "bwd byts/b avg": "avg_bwd_bytes_bulk",
        
        "fwd avg packets/bulk": "avg_fwd_packets_bulk",
        "fwd pkts/b avg": "avg_fwd_packets_bulk",
        
        "bwd avg packets/bulk": "avg_bwd_packets_bulk",
        "bwd pkts/b avg": "avg_bwd_packets_bulk",
        
        "fwd avg bulk rate": "avg_fwd_bulk_rate",
        "fwd blk rate avg": "avg_fwd_bulk_rate",
        
        "bwd avg bulk rate": "avg_bwd_bulk_rate",
        "bwd blk rate avg": "avg_bwd_bulk_rate",
        
        # ==================== 标签列（保留） ====================
        "label": "label",
    }
    
    # 需要删除的特征列表（对应 mapping_class=D）
    DROP_FEATURES = [
        "src ip",
        "src ip address",
        "dst ip",
        "dst ip address",
        "destination ip",
        "destination ip address",
        "flow id",
        "timestamp",
        "external ip",
    ]
    
    # ==================== 统一标准标签体系 ====================
    # CSE-CIC-IDS2017/2018 统一后的 17 类标准语义标签体系
    # 1类正常流量 + 全部攻击类别攻击流量
    STANDARD_LABELS = [
        "Normal",
        "FTP_BruteForce",
        "SSH_BruteForce",
        "DoS_Hulk",
        "DoS_GoldenEye",
        "DoS_Slowloris",
        "DoS_SlowHTTPTest",
        "DDoS_LOIC_HTTP",
        "DDoS_LOIC_UDP",
        "DDoS_HOIC",
        "Web_SQL_Injection",
        "Web_XSS",
        "Web_BruteForce",
        "Infiltration",
        "Botnet",
        "Heartbleed",
        "PortScan"
    ]
    REQUIRED_BASE_LABELS = tuple(STANDARD_LABELS)
    STANDARD_LABELS = STANDARD_LABELS + ["DDoS_Unknown", "DoS_Unknown"]
    PROTECTED_CLASSES = PROTECTED_CLASSES + ["DDoS_Unknown", "DoS_Unknown"]
    
    # 标签映射表 - 基于CSE-CIC-IDS2017/2018细分类映射
    LABEL_MAPPING = {
        # ==================== 正常流量 ====================
        "Normal": "Normal",
        "BENIGN": "Normal",
        "Benign": "Normal",
        "benign": "Normal",
        "BENIGN ": "Normal",
        "Normal ": "Normal",
        
        # ==================== FTP暴力破解 ====================
        "FTP-Patator": "FTP_BruteForce",
        "FTP Brute Force": "FTP_BruteForce",
        "FTP-BruteForce": "FTP_BruteForce",
        "FTP Patator": "FTP_BruteForce",
        "ftp-patator": "FTP_BruteForce",
        "FTPBruteForce": "FTP_BruteForce",
        
        # ==================== SSH暴力破解 ====================
        "SSH-Patator": "SSH_BruteForce",
        "SSH Bruteforce": "SSH_BruteForce",
        "SSH-BruteForce": "SSH_BruteForce",
        "SSH Patator": "SSH_BruteForce",
        "ssh-patator": "SSH_BruteForce",
        "SSHBruteForce": "SSH_BruteForce",
        
        # ==================== DoS Hulk ====================
        "DoS Hulk": "DoS_Hulk",
        "DoS- Hulk": "DoS_Hulk",
        "DoS attacks-Hulk": "DoS_Hulk",
        "DoS-Hulk": "DoS_Hulk",
        "DoS_attacks-Hulk": "DoS_Hulk",
        "dos hulk": "DoS_Hulk",
        "DOS Hulk": "DoS_Hulk",
        "DoS_Hulk": "DoS_Hulk",
        "dos_hulk": "DoS_Hulk",
        "DOS-Hulk": "DoS_Hulk",
        
        # ==================== DoS GoldenEye ====================
        "DoS GoldenEye": "DoS_GoldenEye",
        "DoS-GoldenEye": "DoS_GoldenEye",
        "DoS attacks-GoldenEye": "DoS_GoldenEye",
        "DoS_attacks-GoldenEye": "DoS_GoldenEye",
        "dos goldeneye": "DoS_GoldenEye",
        "DOS GoldenEye": "DoS_GoldenEye",
        "DoS_GoldenEye": "DoS_GoldenEye",
        "dos_goldeneye": "DoS_GoldenEye",
        "DOS-GoldenEye": "DoS_GoldenEye",
        
        # ==================== DoS Slowloris ====================
        "DoS Slowloris": "DoS_Slowloris",
        "DoS-Slowloris": "DoS_Slowloris",
        "DoS attacks-Slowloris": "DoS_Slowloris",
        "DoS Slow-loris": "DoS_Slowloris",
        "DoS_attacks-Slowloris": "DoS_Slowloris",
        "dos slowloris": "DoS_Slowloris",
        "DOS Slowloris": "DoS_Slowloris",
        "DoS_Slowloris": "DoS_Slowloris",
        "dos_slowloris": "DoS_Slowloris",
        "DOS-Slowloris": "DoS_Slowloris",
        
        # ==================== DoS SlowHTTPTest ====================
        "DoS Slowhttptest": "DoS_SlowHTTPTest",
        "DoS-Slowhttptest": "DoS_SlowHTTPTest",
        "DoS attacks-SlowHTTPTest": "DoS_SlowHTTPTest",
        "DoS SlowHTTPTest": "DoS_SlowHTTPTest",
        "DoS-SlowHTTPTest": "DoS_SlowHTTPTest",
        "DoS_attacks-SlowHTTPTest": "DoS_SlowHTTPTest",
        "dos slowhttptest": "DoS_SlowHTTPTest",
        "DOS Slowhttptest": "DoS_SlowHTTPTest",
        "DoS_SlowHTTPTest": "DoS_SlowHTTPTest",
        "dos_slowhttptest": "DoS_SlowHTTPTest",
        "DOS-Slowhttptest": "DoS_SlowHTTPTest",
        
        # ==================== DDoS LOIC HTTP ====================
        "DDoS-LOIC-HTTP": "DDoS_LOIC_HTTP",
        "DDoS LOIC HTTP": "DDoS_LOIC_HTTP",
        "DDOS attack-LOIC-HTTP": "DDoS_LOIC_HTTP",
        "DDoS attack-LOIC-HTTP": "DDoS_LOIC_HTTP",
        "ddos loic http": "DDoS_LOIC_HTTP",
        "DDoS_LOIC_HTTP": "DDoS_LOIC_HTTP",
        "ddos-loic-http": "DDoS_LOIC_HTTP",
        "DDOS_LOIC_HTTP": "DDoS_LOIC_HTTP",
        "ddos_loic_http": "DDoS_LOIC_HTTP",
        
        # ==================== DDoS LOIC UDP ====================
        "DDoS LOIC UDP": "DDoS_LOIC_UDP",
        "DDoS-LOIC-UDP": "DDoS_LOIC_UDP",
        "DDOS attack-LOIC-UDP": "DDoS_LOIC_UDP",
        "DDoS attack-LOIC-UDP": "DDoS_LOIC_UDP",
        "ddos loic udp": "DDoS_LOIC_UDP",
        "DDoS_LOIC_UDP": "DDoS_LOIC_UDP",
        "ddos-loic-udp": "DDoS_LOIC_UDP",
        "DDOS_LOIC_UDP": "DDoS_LOIC_UDP",
        "ddos_loic_udp": "DDoS_LOIC_UDP",
        
        # ==================== DDoS HOIC ====================
        "DDoS HOIC": "DDoS_HOIC",
        "DDoS-HOIC": "DDoS_HOIC",
        "DDoS attack-HOIC": "DDoS_HOIC",
        "DDOS attack-HOIC": "DDoS_HOIC",
        "ddos hoic": "DDoS_HOIC",
        "DDoS_HOIC": "DDoS_HOIC",
        "ddos-hoic": "DDoS_HOIC",
        "DDOS_HOIC": "DDoS_HOIC",
        "ddos_hoic": "DDoS_HOIC",
        
        # ==================== DDoS通用变体 ====================
        "DDoS": "DDoS_Unknown",
        "DDOS": "DDoS_Unknown",
        "ddos": "DDoS_Unknown",
        "DDos": "DDoS_Unknown",
        "DoS": "DoS_Unknown",
        "dos": "DoS_Unknown",
        "DOS": "DoS_Unknown",
        
        # ==================== Web SQL注入 ====================
        "SQL Injection": "Web_SQL_Injection",
        "Sql Injection": "Web_SQL_Injection",
        "sql injection": "Web_SQL_Injection",
        "SQL_Injection": "Web_SQL_Injection",
        "Web Attack - Sql Injection": "Web_SQL_Injection",
        "Web Attack - SQL Injection": "Web_SQL_Injection",
        "Web Attack Sql Injection": "Web_SQL_Injection",
        "Web Attack - SQLInjection": "Web_SQL_Injection",
        "web sql injection": "Web_SQL_Injection",
        "Web_SQL_Injection": "Web_SQL_Injection",
        "sql_injection": "Web_SQL_Injection",
        
        # ==================== Web XSS ====================
        "XSS": "Web_XSS",
        "xss": "Web_XSS",
        "XSS ": "Web_XSS",
        "Cross-site Scripting": "Web_XSS",
        "Web Attack - XSS": "Web_XSS",
        "Web Attack XSS": "Web_XSS",
        "web xss": "Web_XSS",
        "Web_XSS": "Web_XSS",
        "xss_attack": "Web_XSS",
        
        # ==================== Web暴力破解 ====================
        "Web Attack - Brute Force": "Web_BruteForce",
        "Web Attack BruteForce": "Web_BruteForce",
        "Web BruteForce": "Web_BruteForce",
        "Web Brute Force": "Web_BruteForce",
        "Web Attack - BruteForce": "Web_BruteForce",
        "web brute force": "Web_BruteForce",
        "Web_BruteForce": "Web_BruteForce",
        "Brute Force-Web": "Web_BruteForce",
        "BruteForce-Web": "Web_BruteForce",
        "brute_force-web": "Web_BruteForce",
        
        # ==================== Web通用攻击 ====================
        "Web Attack": "Web_BruteForce",
        "Web-attack": "Web_BruteForce",
        "web attack": "Web_BruteForce",
        "web_attack": "Web_BruteForce",
        
        # ==================== Infiltration ====================
        "Infiltration": "Infiltration",
        "Infiltration ": "Infiltration",
        "infiltration": "Infiltration",
        "INFILTRATION": "Infiltration",
        "Infiltrating": "Infiltration",
        "Infiltration Attack": "Infiltration",
        "infiteration": "Infiltration",
        "Infiteration": "Infiltration",
        "infilteration": "Infiltration",
        "Infilteration": "Infiltration",
        "Infiltration - Dropbox Download": "Infiltration",
        "Dropbox Download": "Infiltration",
        "Infiltration - NMAP PortScan": "Infiltration",
        "NMAP PortScan": "Infiltration",
        "Infiltration_Dropbox_Download": "Infiltration",
        "INFILTRATION - Dropbox Download": "Infiltration",
        "INFILTRATION - NMAP PortScan": "Infiltration",
        
        # ==================== Botnet ====================
        "Bot": "Botnet",
        "Botnet": "Botnet",
        "Bot ": "Botnet",
        "Botnet ": "Botnet",
        "bot": "Botnet",
        "BOT": "Botnet",
        "Botnet Traffic": "Botnet",
        "Botnet Attack": "Botnet",
        "Botnet_ARES": "Botnet",
        
        # ==================== Heartbleed ====================
        "Heartbleed": "Heartbleed",
        "Heartbleed ": "Heartbleed",
        "heartbleed": "Heartbleed",
        "HEARTBLEED": "Heartbleed",
        "Heartbleed Attack": "Heartbleed",
        "heartbleed_attack": "Heartbleed",
        
        # ==================== PortScan ====================
        "PortScan": "PortScan",
        "Port Scan": "PortScan",
        "Port-Scan": "PortScan",
        "portscan": "PortScan",
        "PORTSCAN": "PortScan",
        "Portscan": "PortScan",
        "Nmap": "PortScan",
        "Sweep": "PortScan",
        "port_scan": "PortScan",
    }


# 创建配置实例
config = Config()


def get_active_config() -> Config:
    """返回当前主流程使用的Config，避免函数内部重复生成新的run目录。"""
    cfg = globals().get("config", None)
    if cfg is not None and hasattr(cfg, "OUTPUT_DIR"):
        return cfg
    return Config()


def validate_config(cfg: Config) -> None:
    """服务器版参数一致性检查，防止采样目标、切分比例、内存阈值互相打架。"""
    if bool(getattr(cfg, "FORECAST_UNIFIED_SPLIT", False)) != bool(getattr(cfg, "EXPORT_FUSED_FORECAST", False)):
        raise ValueError("FORECAST_UNIFIED_SPLIT与EXPORT_FUSED_FORECAST需要同时启用或同时关闭，避免缺少共享划分/顺序旁表")
    if cfg.FORECAST_TRAIN_COVERAGE_POLICY != "global_ordered":
        raise ValueError("本版仅支持全局顺序划分global_ordered")
    if min(cfg.FORECAST_ROWS_PER_WINDOW,cfg.FORECAST_SEQ_LEN,cfg.FORECAST_LABEL_LEN,cfg.FORECAST_PRED_LEN,cfg.IO_CHUNK_ROWS)<1:
        raise ValueError("流量窗口、预测长度和块大小必须为正数")
    if cfg.FORECAST_LABEL_LEN>cfg.FORECAST_SEQ_LEN or cfg.FORECAST_SEQ_LEN<4 or cfg.FORECAST_BRIDGE_OOF_FOLDS<2:
        raise ValueError("要求历史长度>=4、解码历史<=输入历史、桥接折数>=2")
    if not 0.5<=cfg.MEMORY_BUDGET_FRACTION<cfg.CRITICAL_MEMORY_USAGE:
        raise ValueError("内存预算比例必须介于0.5与强保护阈值之间")
    try:
        from scipy.optimize import milp
    except ImportError as exc:
        raise RuntimeError("全局划分需要支持scipy.optimize.milp的SciPy版本（>=1.9）；环境预检已在读取数据前停止") from exc
    split_sum = cfg.TRAIN_TEST_SPLIT + cfg.VAL_SPLIT + cfg.TEST_SPLIT
    if abs(split_sum - 1.0) > 1e-6:
        raise ValueError(f"训练/验证/测试比例之和必须为1，当前为 {split_sum:.6f}")

    if str(getattr(cfg, "BALANCING_TARGET_MODE", "adaptive")).lower() == "manual":
        target_total = int(sum(cfg.SAMPLING_STRATEGY_MANUAL.values()))
        if cfg.MAX_SAMPLES < target_total:
            logger.warning(
                f"MAX_SAMPLES={cfg.MAX_SAMPLES:,} 小于手动SMOTE目标总和 {target_total:,}；"
                "建议继续上调MAX_SAMPLES或降低SAMPLING_STRATEGY_MANUAL。"
            )
    else:
        target_total = int(getattr(cfg, "ADAPTIVE_BALANCE_TARGET_TOTAL", cfg.MAX_SAMPLES))
        if cfg.MAX_SAMPLES < target_total:
            logger.warning(
                f"MAX_SAMPLES={cfg.MAX_SAMPLES:,} 小于自适应采样目标总量 {target_total:,}；"
                "实际目标会被MAX_SAMPLES截断。"
            )

    if not (0 < cfg.MAX_MEMORY_USAGE < cfg.CRITICAL_MEMORY_USAGE <= 0.99):
        raise ValueError("内存阈值应满足 0 < MAX_MEMORY_USAGE < CRITICAL_MEMORY_USAGE <= 0.99")
    for source,features in cfg.FEATURE_UNIT_MULTIPLIERS.items():
        for feature,multiplier in features.items():
            if not np.isfinite(float(multiplier)) or float(multiplier)<=0:
                raise ValueError(f"单位换算倍率必须为有限正数: {source}/{feature}")

    if cfg.MMD_SAMPLE_SIZE > 20000:
        logger.warning("MMD_SAMPLE_SIZE超过20000会产生非常大的二次距离矩阵，耗时和内存都会明显上升。")

    logger.info(
        "服务器版配置检查通过: "
        f"FULL_READ_CSV={cfg.FULL_READ_CSV}, MAX_SAMPLES={cfg.MAX_SAMPLES:,}, "
        f"采样目标模式={getattr(cfg, 'BALANCING_TARGET_MODE', 'adaptive')}, 目标总量={target_total:,}, MMD_SAMPLE_SIZE={cfg.MMD_SAMPLE_SIZE:,}, "
        f"AE_BATCH_SIZE={cfg.AE_BATCH_SIZE:,}"
    )


def is_normal_label_value(label: Any) -> bool:
    """兼容CIC-IDS不同版本中的良性流量标签。"""
    return str(label).strip().lower() in {"normal", "benign", "benign_traffic"}


def is_invalid_label_value(label: Any) -> bool:
    """识别不应进入采样/绘图统计的异常标签值。"""
    text = str(label).strip().lower()
    return text in {"", "nan", "none", "null", "label", "class", "unknown"}


def _normalize_label_lookup_key(label: Any) -> str:
    """生成标签查表key：大小写、空白、连字符、下划线、乱码统一归一化。"""
    if pd.isna(label):
        return ""
    text = clean_mangled_label(str(label))
    text = text.replace("\ufeff", " ").replace("\u200b", " ").replace("\xa0", " ")
    text = text.replace("–", "-").replace("—", "-")
    text = text.strip().lower()
    text = re.sub(r"[_\-/]+", " ", text)
    text = re.sub(r"[^a-z0-9]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _standard_label_set() -> set:
    return set(str(x) for x in get_active_config().STANDARD_LABELS)


_LABEL_LOOKUP_CACHE: Optional[Dict[str, str]] = None


def _get_label_lookup_cache() -> Dict[str, str]:
    """构建一次配置中的标准类别标签查表缓存，避免每一行重复重建字典。"""
    global _LABEL_LOOKUP_CACHE
    if _LABEL_LOOKUP_CACHE is not None:
        return _LABEL_LOOKUP_CACHE

    cfg = get_active_config()
    std_set = set(str(x) for x in cfg.STANDARD_LABELS)
    lookup: Dict[str, str] = {}
    for std in cfg.STANDARD_LABELS:
        lookup[_normalize_label_lookup_key(std)] = std
        lookup[str(std).strip()] = std
    for src, dst in cfg.LABEL_MAPPING.items():
        if dst in std_set:
            lookup[_normalize_label_lookup_key(src)] = dst
            lookup[str(src).strip()] = dst
    _LABEL_LOOKUP_CACHE = lookup
    return lookup


def clean_label_series_fast(series: pd.Series) -> pd.Series:
    """向量化清理标签字符串，替代逐行clean_mangled_label。"""
    s = series.astype("string").fillna("")
    s = s.str.replace("\ufeff", " ", regex=False)
    s = s.str.replace("\u200b", " ", regex=False)
    s = s.str.replace("\xa0", " ", regex=False)
    s = s.str.replace("\ufffd", " ", regex=False)
    s = s.str.replace("�", " ", regex=False)
    s = s.str.replace(r"\s+", " ", regex=True).str.strip()
    return s.astype(str)


def map_traffic_label_series(labels: pd.Series, source=None) -> pd.Series:
    """快速批量映射标签：只对唯一标签调用canonicalize，避免百万行逐行慢映射。"""
    if labels is None or len(labels) == 0:
        return pd.Series(dtype=object)
    if labels.dtype == object and all(isinstance(v,str) and v in get_active_config().STANDARD_LABELS
                                      for v in pd.unique(labels)):
        return labels
    cleaned = clean_label_series_fast(labels)
    unique_labels = pd.unique(cleaned)
    mapping = {raw: canonicalize_traffic_label(raw, source=source) for raw in unique_labels}
    return cleaned.map(mapping)


def canonicalize_traffic_label(label: Any, source=None) -> Optional[str]:
    """严格映射为配置中的标准类别标准流量标签；无法映射时返回None。

    速度优化：标签查表字典使用全局缓存，不再对每一行重建lookup。
    """
    cfg = get_active_config()
    if pd.isna(label):
        return None

    raw = clean_mangled_label(str(label)).strip()
    if raw == "" or is_invalid_label_value(raw):
        return None

    ambiguity_key = _normalize_label_lookup_key(raw)
    if ambiguity_key in {"ddos", "ddos attack", "ddos attacks", "dos", "dos attack", "dos attacks", "dos ddos", "ddos dos"}:
        source_key = str(source).split("_")[0] if source is not None else "unknown"
        overrides = getattr(cfg, "SOURCE_LABEL_OVERRIDES", {}).get(source_key, {})
        resolved = overrides.get(raw, overrides.get(ambiguity_key))
        if resolved is not None:
            if resolved not in cfg.STANDARD_LABELS:
                raise ValueError(f"来源{source_key}歧义标签{raw!r}的人工映射不属于配置中的标准类别: {resolved}")
            return resolved
        if ambiguity_key in {"dos ddos","ddos dos"}:
            raise ValueError(f"来源{source_key}标签{raw!r}未区分DoS/DDoS，必须人工核实后配置归属")
        if getattr(cfg, "AMBIGUOUS_LABEL_POLICY", "error") == "separate_unknown":
            return "DoS_Unknown" if ambiguity_key.startswith("dos") else "DDoS_Unknown"
        if getattr(cfg, "AMBIGUOUS_LABEL_POLICY", "error") != "legacy_assumption":
            raise ValueError(f"来源{source_key}标签{raw!r}缺少攻击子类信息；请核实后设置"
                             "Config.SOURCE_LABEL_OVERRIDES。不能自动猜测HOIC/LOIC。")
        logger.warning("[人工归类假设] 来源%s的%s按用户确认的旧规则归入DDoS_HOIC", source_key, raw)
        return "DDoS_HOIC"

    std_set = _standard_label_set()

    # 1) 已经是标准标签，直接通过
    if raw in std_set:
        return raw

    # 2) 原始LABEL_MAPPING精确匹配
    mapped = cfg.LABEL_MAPPING.get(raw) or cfg.LABEL_MAPPING.get(raw.strip())
    if mapped in std_set:
        return mapped

    # 3) 缓存归一化查表
    key = _normalize_label_lookup_key(raw)
    if key == "" or is_invalid_label_value(key):
        return None

    lookup = _get_label_lookup_cache()
    if key in lookup:
        return lookup[key]

    # 4) 规则兜底：只输出配置中的标准类别之一；匹配不到就丢弃
    compact = key.replace(" ", "")
    if key in {"normal", "benign", "benign traffic"}:
        return "Normal"
    if "ftp" in key and ("patator" in key or "brute" in key):
        return "FTP_BruteForce"
    if "ssh" in key and ("patator" in key or "brute" in key):
        return "SSH_BruteForce"
    if "goldeneye" in compact:
        return "DoS_GoldenEye"
    if "slowloris" in compact or "slow loris" in key:
        return "DoS_Slowloris"
    if "slowhttptest" in compact or "slow http test" in key or "slowhttp" in compact:
        return "DoS_SlowHTTPTest"
    if "hulk" in key and ("dos" in key or "ddos" not in key):
        return "DoS_Hulk"
    if "loic" in key and "http" in key:
        return "DDoS_LOIC_HTTP"
    if "loic" in key and "udp" in key:
        return "DDoS_LOIC_UDP"
    if "hoic" in key or key in {"ddos", "ddos attack", "dos"}:
        return "DDoS_HOIC"
    if "sql" in key and ("injection" in key or "inject" in key):
        return "Web_SQL_Injection"
    if "xss" in key or "cross site scripting" in key:
        return "Web_XSS"
    if "web" in key and "brute" in key:
        return "Web_BruteForce"
    if "web attack" in key:
        return "Web_BruteForce"
    if "infil" in key or "dropbox" in key:
        return "Infiltration"
    if "botnet" in key or key == "bot" or key.startswith("bot "):
        return "Botnet"
    if "heartbleed" in key:
        return "Heartbleed"
    if "portscan" in compact or "port scan" in key or key in {"nmap", "sweep"}:
        return "PortScan"

    return None


def enforce_standard_labels(df: pd.DataFrame, stage_name: str = "unknown", label_col: str = "Label",
                            raise_if_empty: bool = True, copy: bool = True) -> pd.DataFrame:
    """强制DataFrame标签只保留配置中的标准类别标准标签。

    速度优化：先清理Series，再只对唯一标签做映射；审计文件只写聚合统计，不写逐行明细。
    """
    cfg = get_active_config()
    if df is None or len(df) == 0:
        return df
    if label_col not in df.columns:
        raise ValueError(f"[{stage_name}] 缺少标签列 {label_col}")

    # 默认仍返回独立副本；只有调用者明确拥有该表时才允许copy=False。
    # 对已规范化的object标签只检查唯一值，避免每阶段再清洗数百万个字符串。
    unique_labels = pd.unique(df[label_col])
    if df[label_col].dtype == object and all(
            isinstance(value, str) and value in cfg.STANDARD_LABELS for value in unique_labels):
        out = df.copy() if copy else df
        out.reset_index(drop=True, inplace=True)
        return out
    raw_labels = clean_label_series_fast(df[label_col])
    mapped = map_traffic_label_series(raw_labels)
    valid_mask = mapped.isin(cfg.STANDARD_LABELS)
    invalid_count = int((~valid_mask).sum())

    if invalid_count > 0:
        try:
            audit_dir = os.path.join(cfg.OUTPUT_DIR, "label_audit")
            os.makedirs(audit_dir, exist_ok=True)
            audit_summary = (
                pd.DataFrame({"raw_label": raw_labels.loc[~valid_mask], "mapped_label": mapped.loc[~valid_mask]})
                .groupby(["raw_label", "mapped_label"], dropna=False)
                .size()
                .reset_index(name="count")
                .sort_values("count", ascending=False)
                .head(int(getattr(cfg, "LABEL_AUDIT_TOP_N", 200)))
            )
            safe_stage = re.sub(r"[^A-Za-z0-9_\-]+", "_", str(stage_name))[:120]
            audit_summary.to_csv(
                os.path.join(audit_dir, f"invalid_labels_{safe_stage}.csv"),
                index=False, encoding="utf-8-sig"
            )
        except Exception as exc:
            logger.warning(f"[{stage_name}] 标签审计文件保存失败: {exc}")
        logger.warning(f"[{stage_name}] 删除 {invalid_count:,} 行非配置中的标准类别标准标签样本。")
        print(f"[标签强约束] {stage_name}: 删除 {invalid_count:,} 行非配置中的标准类别标准标签样本")

    out = df.loc[valid_mask.values].copy() if invalid_count else (df.copy() if copy else df)
    out[label_col] = mapped.loc[valid_mask].astype(str).values
    out.reset_index(drop=True, inplace=True)

    remaining = sorted(set(out[label_col].dropna().astype(str)) - _standard_label_set())
    if remaining:
        raise ValueError(f"[{stage_name}] 标签强约束失败，仍存在非标准标签: {remaining}")
    if raise_if_empty and len(out) == 0:
        raise ValueError(f"[{stage_name}] 严格标签过滤后数据为空，请检查原始Label映射。")

    return out


def assert_standard_labels_only(df: pd.DataFrame, stage_name: str = "unknown", label_col: str = "Label") -> None:
    """保存/切分/窗口化前的最后安全断言。"""
    if df is None or len(df) == 0 or label_col not in df.columns:
        return
    extra = sorted(set(df[label_col].dropna().astype(str)) - _standard_label_set())
    if extra:
        raise ValueError(f"[{stage_name}] 检测到非配置中的标准类别标准标签: {extra}")


def get_present_attack_labels(df: Optional[pd.DataFrame], label_col: str = "Label") -> set:
    """返回当前DataFrame中出现过的标准攻击类标签，不包含Normal。"""
    cfg = get_active_config()
    attack_labels = set(cfg.STANDARD_LABELS) - {"Normal"}
    if df is None or len(df) == 0 or label_col not in df.columns:
        return set()

    mapped = map_traffic_label_series(df[label_col])
    present = set(mapped.dropna().astype(str))
    return present & attack_labels


def assert_no_class_disappears(before_df: Optional[pd.DataFrame],
                               after_df: Optional[pd.DataFrame],
                               stage_name: str,
                               label_col: str = "Label",
                               before_attack_labels: Optional[set] = None) -> None:
    """防止任何已出现的标准攻击类别在某个处理阶段后变成0。

    只检查16个攻击类，Normal仍允许被下采样。before_attack_labels用于避免为了断言而复制大表。
    """
    cfg = get_active_config()
    attack_labels = set(cfg.STANDARD_LABELS) - {"Normal"}

    if before_attack_labels is None:
        before_labels = get_present_attack_labels(before_df, label_col=label_col)
    else:
        before_labels = set(before_attack_labels) & attack_labels

    after_labels = get_present_attack_labels(after_df, label_col=label_col)
    disappeared = sorted(before_labels - after_labels)

    if disappeared:
        before_counts = pd.Series(dtype=int)
        after_counts = pd.Series(dtype=int)
        try:
            if before_df is not None and label_col in before_df.columns:
                before_counts = map_traffic_label_series(before_df[label_col]).value_counts()
            if after_df is not None and label_col in after_df.columns:
                after_counts = map_traffic_label_series(after_df[label_col]).value_counts()

            audit_dir = os.path.join(cfg.OUTPUT_DIR, "balance_debug_logs", "class_disappearance_checks")
            os.makedirs(audit_dir, exist_ok=True)
            safe_stage = re.sub(r"[^A-Za-z0-9_\-]+", "_", str(stage_name))[:120]
            audit_rows = []
            for label in disappeared:
                audit_rows.append({
                    "stage": stage_name,
                    "Label": label,
                    "before_count": int(before_counts.get(label, -1)) if len(before_counts) else -1,
                    "after_count": int(after_counts.get(label, 0)) if len(after_counts) else 0,
                })
            pd.DataFrame(audit_rows).to_csv(
                os.path.join(audit_dir, f"disappeared_classes_{safe_stage}.csv"),
                index=False,
                encoding="utf-8-sig"
            )
        except Exception as exc:
            logger.warning(f"[{stage_name}] 类别消失审计文件保存失败: {exc}")

        raise ValueError(
            f"[{stage_name}] 以下攻击类别在处理后消失: {disappeared}。"
            "请检查该阶段的下采样、标签过滤、IF过滤或数据源平衡逻辑。"
        )


def export_label_count_audit(df: Optional[pd.DataFrame], stage_name: str, label_col: str = "Label") -> None:
    """导出关键阶段配置中的标准类别标签数量，便于定位某个类别在哪一步被削减。"""
    try:
        cfg = get_active_config()
        if df is None or len(df) == 0 or label_col not in df.columns:
            return
        mapped = map_traffic_label_series(df[label_col])
        counts = mapped[mapped.isin(cfg.STANDARD_LABELS)].value_counts()
        rows = []
        for label in cfg.STANDARD_LABELS:
            rows.append({"Label": label, "Count": int(counts.get(label, 0))})
        audit_dir = os.path.join(cfg.OUTPUT_DIR, "label_count_audit")
        os.makedirs(audit_dir, exist_ok=True)
        safe_stage = re.sub(r"[^A-Za-z0-9_\-]+", "_", str(stage_name))[:120]
        pd.DataFrame(rows).to_csv(
            os.path.join(audit_dir, f"label_counts_{safe_stage}.csv"),
            index=False,
            encoding="utf-8-sig"
        )
    except Exception as exc:
        logger.warning(f"[{stage_name}] 标签数量审计导出失败: {exc}")


def _label_candidate_columns(df: pd.DataFrame) -> List[str]:
    """返回可能的标签列，包含 pandas 自动生成的 Label.1 / Class.1 等重复列。"""
    candidates = []
    label_tokens = {"label", "class", "attacktype", "attack_type", "label_class", "attack", "type", "category"}
    for col in df.columns:
        raw = str(col).strip()
        lower = raw.lower()
        base = re.sub(r"\.\d+$", "", lower).strip()
        normalized = re.sub(r"[^a-z0-9]+", "_", base).strip("_")
        if base in label_tokens or normalized in label_tokens:
            candidates.append(col)
        elif any(tok in normalized.split("_") for tok in ["label", "attack", "class", "category"]):
            candidates.append(col)
    # 保持原列顺序去重
    seen = set()
    ordered = []
    for col in candidates:
        if col not in seen:
            seen.add(col)
            ordered.append(col)
    return ordered


def _choose_best_label_column(df: pd.DataFrame, dataset_name: str = "unknown") -> str:
    """从多个候选标签列中选择可映射配置中的标准类别最多、攻击样本最多的一列。

    修复点：旧逻辑会先删除 Label.1 这类 pandas 重名列；如果真实攻击标签落在重复标签列中，
    Web_SQL_Injection 等小类会在读取阶段被误删或严重少算。
    """
    candidates = _label_candidate_columns(df)
    if not candidates:
        raise ValueError(f"{dataset_name}数据集未找到任何标签列！")

    scored = []
    for col in candidates:
        try:
            cleaned = clean_label_series_fast(df[col])
            mapped = map_traffic_label_series(cleaned, source=dataset_name)
            valid_mask = mapped.isin(get_active_config().STANDARD_LABELS)
            valid_count = int(valid_mask.sum())
            attack_count = int((valid_mask & ~mapped.map(is_normal_label_value)).sum())
            unique_attack_count = int(mapped[valid_mask & ~mapped.map(is_normal_label_value)].nunique())
            sql_count = int((mapped == "Web_SQL_Injection").sum())
            heartbleed_count = int((mapped == "Heartbleed").sum())
            normal_count = int((mapped == "Normal").sum())
            # 优先选择：有效配置中的标准类别多、攻击样本多、攻击类别覆盖多；SQL/Heartbleed计数仅用于平局诊断。
            score = valid_count + attack_count * 3 + unique_attack_count * 1000 + sql_count * 2 + heartbleed_count * 2
            scored.append({
                "column": col,
                "valid_count": valid_count,
                "normal_count": normal_count,
                "attack_count": attack_count,
                "unique_attack_count": unique_attack_count,
                "Web_SQL_Injection": sql_count,
                "Heartbleed": heartbleed_count,
                "score": score,
            })
        except Exception as exc:
            scored.append({
                "column": col,
                "valid_count": 0,
                "normal_count": 0,
                "attack_count": 0,
                "unique_attack_count": 0,
                "Web_SQL_Injection": 0,
                "Heartbleed": 0,
                "score": -1,
                "error": str(exc),
            })

    scored_df = pd.DataFrame(scored).sort_values("score", ascending=False)
    try:
        audit_dir = os.path.join(get_active_config().OUTPUT_DIR, "label_audit")
        os.makedirs(audit_dir, exist_ok=True)
        safe_name = re.sub(r"[^A-Za-z0-9_\-]+", "_", str(dataset_name))[:120]
        scored_df.to_csv(os.path.join(audit_dir, f"label_column_candidates_{safe_name}.csv"), index=False, encoding="utf-8-sig")
    except Exception as exc:
        logger.warning(f"[{dataset_name}] 标签列候选审计保存失败: {exc}")

    best_col = scored_df.iloc[0]["column"]
    if len(scored_df) > 1:
        print(f"  [标签列选择] 候选列评分已保存，选择 '{best_col}' 作为Label列")
        for _, row in scored_df.head(5).iterrows():
            print(
                f"    {row['column']}: valid={int(row['valid_count'])}, "
                f"attack={int(row['attack_count'])}, unique_attack={int(row['unique_attack_count'])}, "
                f"SQL={int(row['Web_SQL_Injection'])}, Heartbleed={int(row['Heartbleed'])}"
            )
    return best_col

def ordered_standard_labels(labels: Optional[List[Any]] = None) -> List[str]:
    """按照固定配置中的标准类别标签顺序排序；非配置中的标准类别标签不再进入图表或统计。"""
    if labels is None:
        return list(get_active_config().STANDARD_LABELS)
    canonical = []
    for label in labels:
        mapped = canonicalize_traffic_label(label)
        if mapped is not None:
            canonical.append(mapped)
    present = set(canonical)
    return [label for label in get_active_config().STANDARD_LABELS if label in present]


def build_adaptive_sampling_targets(label_counts: pd.Series,
                                    include_normal: bool = False,
                                    debug_log=None) -> Dict[str, int]:
    """根据当前数据分布生成“小类优先、大类克制”的过采样目标。

    核心原则：
    1. 只生成过采样目标，不在SMOTE阶段下采样。
    2. Normal默认不参与SMOTE过采样，Normal比例由前置Normal下采样控制。
    3. 极小类/小类优先获得新增预算；大类攻击默认不增强。
    4. 中等类最多轻微增强；如果开启大类增强，大类最高也只允许1.2倍。
    5. PROTECTED_CLASSES只表示“不能删成0”，不再意味着要被强行补到高目标。
    6. 输出adaptive_sampling_targets.csv，明确每个类别的分层、目标和是否被跳过。
    """
    cfg = get_active_config()
    counts = pd.Series(label_counts).dropna().astype(int)
    mapped_counts = {}
    for raw_label, count in counts.items():
        mapped_label = canonicalize_traffic_label(raw_label)
        if mapped_label is not None:
            mapped_counts[mapped_label] = mapped_counts.get(mapped_label, 0) + int(count)
    counts = pd.Series(mapped_counts, dtype=int)
    counts = counts.reindex([l for l in get_active_config().STANDARD_LABELS if l in counts.index]).dropna().astype(int)
    if counts.empty:
        return {}

    # manual模式保持原有语义，不在这里额外改动。
    if str(getattr(cfg, "BALANCING_TARGET_MODE", "adaptive")).lower() == "manual":
        manual = getattr(cfg, "SAMPLING_STRATEGY_MANUAL", {})
        return {str(k): int(v) for k, v in manual.items() if k in counts.index and int(v) > int(counts.get(k, 0))}

    attack_labels = [str(l) for l in counts.index if not is_normal_label_value(l)]
    if not attack_labels:
        return {}

    normal_current = int(counts.get("Normal", 0))
    attack_counts = counts.reindex(attack_labels).fillna(0).astype(int)
    current_attack_total = int(attack_counts.sum())

    max_attack_to_normal_ratio = float(getattr(cfg, "ADAPTIVE_MAX_ATTACK_TO_NORMAL_RATIO", 0.70))
    max_single_attack_to_normal_ratio = float(getattr(cfg, "ADAPTIVE_MAX_SINGLE_ATTACK_TO_NORMAL_RATIO", 0.18))

    if normal_current > 0:
        ratio_attack_budget = int(np.floor(normal_current * max_attack_to_normal_ratio))
        # 严格攻击增长预算：用于约束整体新增攻击样本。
        # 注意：如果当前攻击总量已经超过比例上限，后面会给 tiny/small 类一个很小的兜底增强预算，
        # 避免大类占满攻击总量后小类完全无法增强。
        strict_attack_growth_budget = max(0, ratio_attack_budget - current_attack_total)
    else:
        ratio_attack_budget = None
        # 无Normal时使用总量软上限兜底，但仍然走小类优先规则。
        total_current = int(counts.sum())
        target_total = int(min(getattr(cfg, "MAX_SAMPLES", total_current),
                               max(total_current, getattr(cfg, "ADAPTIVE_BALANCE_TARGET_TOTAL", total_current))))
        strict_attack_growth_budget = max(0, target_total - total_current)

    allowed_extra_total = int(strict_attack_growth_budget)
    attack_total_over_ratio_cap = bool(
        normal_current > 0 and ratio_attack_budget is not None and current_attack_total >= ratio_attack_budget
    )

    smote_min_required = int(getattr(cfg, "SMOTE_MIN_REQUIRED", 5))
    tiny_target = int(getattr(cfg, "ADAPTIVE_TINY_CLASS_TARGET", 100))
    tiny_multiplier = float(getattr(cfg, "ADAPTIVE_TINY_CLASS_MAX_MULTIPLIER", 20.0))

    small_threshold = int(getattr(cfg, "ADAPTIVE_SMALL_CLASS_THRESHOLD", 2000))
    medium_threshold = int(getattr(cfg, "ADAPTIVE_MEDIUM_CLASS_THRESHOLD", 5000))
    small_floor = int(getattr(cfg, "ADAPTIVE_ATTACK_MIN_TARGET", 2000))
    small_multiplier = float(getattr(cfg, "ADAPTIVE_SMALL_CLASS_MAX_MULTIPLIER", 4.0))
    medium_multiplier = float(getattr(cfg, "ADAPTIVE_MEDIUM_CLASS_MAX_MULTIPLIER", 1.2))
    large_multiplier = float(getattr(cfg, "ADAPTIVE_LARGE_CLASS_MAX_MULTIPLIER", 1.2))
    disable_large_oversampling = bool(getattr(cfg, "ADAPTIVE_DISABLE_LARGE_CLASS_OVERSAMPLING", True))

    max_synth_ratio = float(getattr(cfg, "ADAPTIVE_MAX_SYNTHETIC_RATIO", 0.60))
    ratio_cap_multiplier = 1.0 / max(1.0 - max_synth_ratio, 1e-6)

    preliminary_rows = []
    for label in attack_labels:
        current = int(attack_counts.get(label, 0))
        if current <= 0:
            continue

        single_ratio_cap = None
        if normal_current > 0 and max_single_attack_to_normal_ratio > 0:
            single_ratio_cap = int(np.floor(normal_current * max_single_attack_to_normal_ratio))

        if current < smote_min_required:
            tier = "tiny"
            priority = 0
            method_hint = "random_copy_tiny"
            desired_floor = max(current, min(tiny_target, int(np.ceil(current * tiny_multiplier))))
            multiplier_cap = max(current, int(np.ceil(current * tiny_multiplier)))
            reason = "极小类，优先随机复制到温和目标"
        elif current < small_threshold:
            tier = "small"
            priority = 1
            method_hint = "smote_small_priority"
            desired_floor = max(current, small_floor)
            # 小类需要真正补强，不再受 ADAPTIVE_MAX_SYNTHETIC_RATIO 的 2.5x 软上限限制；
            # 整体新增量由小类兜底预算/攻击总量预算控制，避免总样本爆炸。
            multiplier_cap = max(current, int(np.ceil(current * small_multiplier)))
            reason = "小类，优先SMOTE增强"
        elif current < medium_threshold:
            tier = "medium"
            priority = 2
            method_hint = "smote_medium_light"
            # 中等类不再强行补到small_floor，只允许轻微增加。
            desired_floor = current
            multiplier_cap = max(current, int(np.ceil(current * min(medium_multiplier, ratio_cap_multiplier))))
            reason = "中等类，只允许轻微增强"
        else:
            tier = "large"
            priority = 3
            if disable_large_oversampling:
                method_hint = "skip_large_enough"
                desired_floor = current
                multiplier_cap = current
                reason = "大类数量已足够，跳过SMOTE增强"
            else:
                method_hint = "smote_large_capped_1_2x"
                desired_floor = current
                multiplier_cap = max(current, int(np.ceil(current * min(large_multiplier, ratio_cap_multiplier))))
                reason = "大类最多轻微增强，倍率不超过配置上限"

        cap = multiplier_cap
        if single_ratio_cap is not None:
            cap = min(cap, max(current, single_ratio_cap))

        # tiny/small有明确补强目标；medium/large只在允许时按倍率给很小增量。
        if tier in {"tiny", "small"}:
            preliminary_target = max(current, min(max(desired_floor, current), cap))
        elif tier == "medium":
            preliminary_target = max(current, cap)
        else:
            preliminary_target = max(current, cap)

        desired_add = max(preliminary_target - current, 0)

        preliminary_rows.append({
            "Label": label,
            "Original_Count": current,
            "Class_Tier": tier,
            "Priority": int(priority),
            "Small_Class_Threshold": int(small_threshold),
            "Medium_Class_Threshold": int(medium_threshold),
            "Desired_Floor": int(desired_floor),
            "Multiplier_Cap": int(multiplier_cap),
            "Single_Normal_Ratio_Cap": int(single_ratio_cap) if single_ratio_cap is not None else "NA",
            "Cap": int(cap),
            "Preliminary_Target": int(preliminary_target),
            "Desired_Add": int(desired_add),
            "Method_Hint": method_hint,
            "Skip_Reason": reason,
            "Protected": label in getattr(cfg, "PROTECTED_CLASSES", []),
        })

    # 如果当前攻击总量已经超过 Normal 比例上限，严格预算会变成0。
    # 但这通常是 DDoS/DoS/PortScan 等大类太多造成的，不应该连带阻止极小类/小类补强。
    # 因此只给 tiny/small 类一个小的兜底预算；medium/large 仍然不能使用这个预算。
    small_over_cap_extra_budget = 0
    if allowed_extra_total <= 0 and attack_total_over_ratio_cap and bool(getattr(cfg, "ADAPTIVE_ALLOW_SMALL_CLASS_OVERSAMPLING_OVER_ATTACK_CAP", True)):
        tiny_small_need = int(sum(
            int(row.get("Desired_Add", 0))
            for row in preliminary_rows
            if str(row.get("Class_Tier", "")) in {"tiny", "small"}
        ))
        configured_budget = int(getattr(cfg, "ADAPTIVE_SMALL_CLASS_OVER_CAP_EXTRA_BUDGET", 10000))
        normal_ratio_budget = configured_budget
        if normal_current > 0:
            normal_ratio_budget = int(np.floor(normal_current * float(getattr(cfg, "ADAPTIVE_SMALL_CLASS_OVER_CAP_NORMAL_RATIO", 0.10))))
        small_over_cap_extra_budget = max(0, min(tiny_small_need, configured_budget, normal_ratio_budget))
        allowed_extra_total = int(small_over_cap_extra_budget)

    # 小类优先分配新增预算：tiny -> small -> medium -> large，同层按当前样本数从少到多。
    additions_by_label: Dict[str, int] = {}
    remaining_budget = int(allowed_extra_total)
    for row in sorted(preliminary_rows, key=lambda r: (int(r["Priority"]), int(r["Original_Count"]), str(r["Label"]))):
        label = str(row["Label"])
        desired_add = int(row["Desired_Add"])
        if desired_add <= 0 or remaining_budget <= 0:
            additions_by_label[label] = 0
            continue
        add = min(desired_add, remaining_budget)
        additions_by_label[label] = int(add)
        remaining_budget -= int(add)

    targets: Dict[str, int] = {}
    audit_rows = []
    for row in preliminary_rows:
        label = str(row["Label"])
        current = int(row["Original_Count"])
        generated = int(additions_by_label.get(label, 0))
        final_target = min(current + generated, int(row["Preliminary_Target"]))

        if final_target > current:
            targets[label] = int(final_target)

        row_out = dict(row)
        row_out.update({
            "Strict_Attack_Growth_Budget": int(strict_attack_growth_budget),
            "Small_Over_Cap_Extra_Budget": int(small_over_cap_extra_budget),
            "Allowed_Attack_Growth_Budget": int(allowed_extra_total),
            "Unused_Attack_Growth_Budget": int(remaining_budget),
            "Attack_Total_Already_Over_Ratio_Cap": bool(attack_total_over_ratio_cap),
            "Attack_Normal_Ratio_Cap": float(max_attack_to_normal_ratio),
            "Single_Attack_Normal_Ratio_Cap": float(max_single_attack_to_normal_ratio),
            "Large_Class_Oversampling_Disabled": bool(disable_large_oversampling),
            "Final_Target": int(final_target),
            "Generated_Target": int(max(final_target - current, 0)),
            "Target_Multiplier": float(final_target / max(current, 1)),
        })
        audit_rows.append(row_out)

    try:
        audit_dir = os.path.join(cfg.OUTPUT_DIR, "balance_debug_logs")
        os.makedirs(audit_dir, exist_ok=True)
        pd.DataFrame(audit_rows).to_csv(os.path.join(audit_dir, "adaptive_sampling_targets.csv"), index=False, encoding="utf-8-sig")
    except Exception as exc:
        logger.warning(f"自适应采样目标审计表保存失败: {exc}")

    if debug_log is not None:
        debug_log("\n【自适应采样目标 - 小类优先，大类克制】")
        debug_log("-" * 80)
        ratio_budget_text = f"{ratio_attack_budget:,}" if ratio_attack_budget is not None else "按总量软上限"
        debug_log(
            f"Normal当前: {normal_current:,}; 当前攻击总量: {current_attack_total:,}; "
            f"攻击总量上限: {ratio_budget_text}; 严格新增预算: {strict_attack_growth_budget:,}; "
            f"小类兜底预算: {small_over_cap_extra_budget:,}; 实际允许新增: {allowed_extra_total:,}; "
            f"未使用新增预算: {remaining_budget:,}"
        )
        debug_log(
            f"小类阈值: <{small_threshold:,}; 中等类阈值: <{medium_threshold:,}; "
            f"小类最高倍率: x{small_multiplier:.2f}; 中等类最高倍率: x{medium_multiplier:.2f}; "
            f"大类最高倍率: x{large_multiplier:.2f}; 大类跳过增强: {disable_large_oversampling}"
        )
        for row in audit_rows:
            debug_log(
                f"  {row['Label']:<30} {row['Original_Count']:>8} -> {row['Final_Target']:>8} "
                f"(tier={row['Class_Tier']}, x{row['Target_Multiplier']:.2f}, "
                f"+{row['Generated_Target']:,}, {row['Method_Hint']}, {row['Skip_Reason']})"
            )

    return targets


def setup_thesis_plot_style() -> None:
    """统一EI论文图表风格：印刷友好、色盲友好、矢量字体可编辑。"""
    sns.set_theme(style="ticks", context="paper")
    matplotlib.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": matplotlib.rcParams.get("font.sans-serif", []) + ["Noto Sans CJK SC", "SimHei", "Microsoft YaHei", "DejaVu Sans"],
        "axes.unicode_minus": False,
        "figure.dpi": 180,
        "savefig.dpi": int(getattr(config, "VISUALIZATION_DPI", 600)) if "config" in globals() else 600,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.edgecolor": "#333333",
        "axes.titleweight": "semibold",
        "axes.titlesize": 11.5,
        "axes.labelsize": 9.5,
        "xtick.labelsize": 8.2,
        "ytick.labelsize": 8.2,
        "legend.fontsize": 8.0,
        "legend.title_fontsize": 8.2,
        "axes.linewidth": 0.75,
        "grid.linewidth": 0.45,
        "grid.alpha": 0.22,
        "grid.linestyle": "--",
        "lines.linewidth": 1.6,
        "patch.edgecolor": "#333333",
        "patch.linewidth": 0.45,
        "figure.constrained_layout.use": False,
    })


EI_COLORS = {
    "blue": "#3B6EA8",
    "orange": "#E07B39",
    "green": "#4C956C",
    "red": "#C84C4C",
    "purple": "#8064A2",
    "cyan": "#4E9AA6",
    "gray": "#7A7A7A",
    "light_blue": "#9EC1E6",
}
EI_SEQUENCE = [
    EI_COLORS["blue"], EI_COLORS["orange"], EI_COLORS["green"],
    EI_COLORS["red"], EI_COLORS["purple"], EI_COLORS["cyan"],
    "#D6A84B", "#6F8FAF", "#B56576", "#6A994E", "#8D6E63",
    "#5E548E", "#2A9D8F", "#BC6C25", "#577590", "#A44A3F", "#778DA9", "#6B705C", "#9B5DE5",
]


def save_thesis_figure(fig, save_path: str, close: bool = True) -> None:
    """安全保存论文可视化图片"""
    try:
        setup_thesis_plot_style()

        if fig is None:
            logger.warning(f"跳过保存：fig为空 {save_path}")
            return

        save_dir = os.path.dirname(save_path)
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)

        root, ext = os.path.splitext(save_path)
        if not ext:
            save_path = root + ".png"
            root = root

        # 防止空白图
        if len(fig.axes) == 0:
            logger.warning(f"跳过保存：无坐标轴 {save_path}")
            return

        fig.tight_layout(pad=0.8)

        fig.savefig(
            save_path,
            dpi=getattr(config, "VISUALIZATION_DPI", 350),
            bbox_inches="tight",
            facecolor="white",
            pad_inches=0.04,
            metadata={"Creator": "AE-IF-ASMOTE visualization pipeline"}
        )

        for extra_ext in [".pdf", ".svg"]:
            try:
                fig.savefig(
                    root + extra_ext,
                    bbox_inches="tight",
                    facecolor="white",
                    pad_inches=0.04,
                    metadata={"Creator": "AE-IF-ASMOTE visualization pipeline"}
                )
            except Exception as e:
                logger.warning(f"附加格式保存失败 {extra_ext}: {e}")

        logger.info(f"可视化保存成功: {save_path}")

    except Exception as e:
        logger.warning(
            f"可视化保存失败: {save_path}, 原因: {e}",
            exc_info=True
        )

    finally:
        if close:
            plt.close(fig)


def _format_plot_value(value: float, kind: str = "count") -> str:
    """用于图内精确标注；坐标轴可缩写，但数据标签不缩写。"""
    if not np.isfinite(value):
        return ""
    if kind == "percent":
        return f"{value:.1f}%"
    if kind == "signed_percent":
        return f"{value:+.2f} pp"
    if kind == "float":
        return f"{value:.4f}"
    return f"{int(round(value)):,}"


def annotate_horizontal_bars(ax, values=None, offset_ratio: float = 0.015, fontsize: int = 8,
                             kind: str = "count") -> None:
    """向后兼容的横向柱形标注接口。"""
    for patch in ax.patches:
        value = float(patch.get_width())
        if not np.isfinite(value):
            continue
        x_value = patch.get_x() + value
        if value == 0 and ax.get_xscale() == "log":
            x_value = ax.get_xlim()[0]
        ax.annotate(_format_plot_value(value, kind),
                    (x_value, patch.get_y() + patch.get_height() / 2),
                    xytext=(3, 0), textcoords="offset points", va="center", ha="left",
                    fontsize=fontsize, color="#222222", clip_on=False)


def add_panel_label(ax, label: str) -> None:
    ax.text(-0.08, 1.04, label, transform=ax.transAxes, fontsize=10.5,
            fontweight="bold", va="bottom", ha="left")


def style_axis(ax, grid_axis: str = "y") -> None:
    ax.grid(axis=grid_axis, zorder=0)
    ax.set_axisbelow(True)
    ax.tick_params(direction="out", length=3, width=0.7)
    sns.despine(ax=ax, top=True, right=True)


def plot_sampling_audit(original_counts: pd.Series,
                        after_smote_counts: pd.Series,
                        final_counts: pd.Series,
                        target_counts: Dict[str, int],
                        save_dir: str) -> None:
    """输出采样审计图，展示原始/目标/SMOTE后/最终数量以及合成倍率。"""
    try:
        setup_thesis_plot_style()
        labels = ordered_standard_labels(list(set(original_counts.index) | set(after_smote_counts.index) | set(final_counts.index) | set(target_counts.keys())))
        if not labels:
            return
        rows = []
        for label in labels:
            orig = int(original_counts.get(label, 0))
            target = int(target_counts.get(label, orig))
            smote = int(after_smote_counts.get(label, orig))
            final = int(final_counts.get(label, smote))
            rows.append({"Label": label, "Original": orig, "Target": target, "After_SMOTE": smote, "Final": final,
                         "Synthetic_Ratio": max(final - orig, 0) / max(final, 1)})
        audit = pd.DataFrame(rows)
        audit.to_csv(os.path.join(save_dir, "sampling_audit_summary.csv"), index=False, encoding="utf-8-sig")

        plot_df = audit.melt(id_vars="Label", value_vars=["Original", "Target", "After_SMOTE", "Final"],
                             var_name="Stage", value_name="Count")
        fig, ax = plt.subplots(figsize=(10.8, max(5.3, 0.38 * len(labels))))
        palette = {"Original": EI_COLORS["blue"], "Target": EI_COLORS["purple"],
                   "After_SMOTE": EI_COLORS["orange"], "Final": EI_COLORS["green"]}
        sns.barplot(data=plot_df, y="Label", x="Count", hue="Stage", ax=ax, palette=palette,
                    edgecolor="white", linewidth=0.35)
        ax.set_xscale("log")
        ax.set_xlabel("Number of samples (log scale)")
        ax.set_ylabel("")
        ax.set_title("Sampling audit across four class-size stages", pad=10)
        annotate_horizontal_bars(ax, fontsize=6.4)
        ax.margins(x=0.18)
        style_axis(ax, "x")
        ax.legend(title="Stage", loc="lower right", frameon=False, ncol=2)
        save_thesis_figure(fig, os.path.join(save_dir, "sampling_audit_comparison.png"))

        fig, ax = plt.subplots(figsize=(9.3, max(5.0, 0.34 * len(labels))))
        audit_sorted = audit.sort_values("Synthetic_Ratio", ascending=True)
        bars = ax.barh(audit_sorted["Label"], audit_sorted["Synthetic_Ratio"] * 100,
                       color=EI_COLORS["red"], edgecolor="white", linewidth=0.4, zorder=3)
        ax.axvline(float(getattr(config, "ADAPTIVE_MAX_SYNTHETIC_RATIO", 0.85)) * 100,
                   color="#333333", linestyle="--", linewidth=1.1, label="Configured upper bound")
        ax.set_xlabel("Synthetic sample proportion in final class (%)")
        ax.set_ylabel("")
        ax.set_title("Synthetic sample proportion by class", pad=10)
        annotate_horizontal_bars(ax, fontsize=7.2, kind="percent")
        ax.set_xlim(0, max(100.0, float((audit_sorted["Synthetic_Ratio"] * 100).max()) * 1.16))
        style_axis(ax, "x")
        ax.legend(frameon=False, loc="lower right")
        save_thesis_figure(fig, os.path.join(save_dir, "sampling_synthetic_ratio.png"))
    except Exception as exc:
        logger.warning(f"采样审计图绘制失败: {exc}")


# ========================== 论文增强可视化：阶段快照与综合报告 ==========================
def _viz_safe_value_counts(series: Optional[pd.Series]) -> pd.Series:
    """安全统计类别数量：先严格映射，再只统计配置中的标准类别标准标签。"""
    if series is None or len(series) == 0:
        return pd.Series(dtype=int)
    unique_mapping = {value: canonicalize_traffic_label(value) for value in pd.unique(series)}
    mapped = series.map(unique_mapping).dropna().astype(str)
    mapped = mapped[mapped.isin(get_active_config().STANDARD_LABELS)]
    if len(mapped) == 0:
        return pd.Series(dtype=int)
    return mapped.value_counts()


def _viz_numeric_columns(df: pd.DataFrame, max_cols: Optional[int] = None) -> List[str]:
    """选择用于阶段快照/分布对比的数值特征，优先保留方差较大的列。"""
    if df is None or len(df) == 0:
        return []
    cols = [c for c in df.columns if c not in ["Label", "dataset_source"] and pd.api.types.is_numeric_dtype(df[c])]
    if not cols:
        return []
    if max_cols is None:
        max_cols = int(getattr(config, "VIZ_FEATURE_TOP_N", 12))
    if len(cols) <= max_cols:
        return cols
    try:
        # 用小样本估计方差，避免对超大表做全量方差计算
        n = min(len(df), int(getattr(config, "VIZ_STAGE_SAMPLE_SIZE", 50000)))
        sample = df.iloc[np.random.default_rng(getattr(config,"SEED",42)).choice(len(df),n,replace=False)][cols] if len(df)>n else df[cols]
        var_rank = sample.replace([np.inf, -np.inf], np.nan).var(numeric_only=True).sort_values(ascending=False)
        return [c for c in var_rank.index.tolist() if c in cols][:max_cols]
    except Exception:
        return cols[:max_cols]


def make_stage_snapshot(stage_name: str,
                        df: Optional[pd.DataFrame],
                        feature_cols: Optional[List[str]] = None,
                        max_sample: Optional[int] = None) -> Dict[str, Any]:
    """生成阶段快照：只保存统计摘要和小样本，不复制完整DataFrame。"""
    cfg = get_active_config()
    if max_sample is None:
        max_sample = int(getattr(cfg, "VIZ_STAGE_SAMPLE_SIZE", 50000))
    snap: Dict[str, Any] = {
        "stage": stage_name,
        "rows": 0,
        "cols": 0,
        "numeric_feature_count": 0,
        "label_counts": pd.Series(dtype=int),
        "label_props": pd.Series(dtype=float),
        "normal_attack_counts": pd.Series(dtype=int),
        "normal_attack_props": pd.Series(dtype=float),
        "source_counts": pd.Series(dtype=int),
        "source_props": pd.Series(dtype=float),
        "label_source_counts": pd.DataFrame(),
        "missing_rate": 0.0,
        "inf_rate": 0.0,
        "zero_rate": 0.0,
        "feature_cols": [],
        "feature_stats": pd.DataFrame(),
        "sample_df": pd.DataFrame(),
    }
    if df is None or len(df) == 0:
        return snap

    snap["rows"] = int(len(df))
    snap["cols"] = int(len(df.columns))
    numeric_cols_all = [c for c in df.columns if c not in ["Label", "dataset_source"] and pd.api.types.is_numeric_dtype(df[c])]
    snap["numeric_feature_count"] = int(len(numeric_cols_all))

    if "Label" in df.columns:
        label_counts = _viz_safe_value_counts(df["Label"])
        ordered_labels = ordered_standard_labels(label_counts.index.tolist())
        label_counts = label_counts.reindex(ordered_labels).dropna().astype(int)
        snap["label_counts"] = label_counts
        snap["label_props"] = label_counts / max(label_counts.sum(), 1)
        normal_count = int(label_counts[[l for l in label_counts.index if is_normal_label_value(l)]].sum()) if len(label_counts) else 0
        attack_count = int(label_counts.sum() - normal_count)
        na_counts = pd.Series({"Normal": normal_count, "Attack": attack_count})
        snap["normal_attack_counts"] = na_counts
        snap["normal_attack_props"] = na_counts / max(na_counts.sum(), 1)

    if "dataset_source" in df.columns:
        src_counts = df["dataset_source"].dropna().astype(str).str.strip().value_counts()
        snap["source_counts"] = src_counts
        snap["source_props"] = src_counts / max(src_counts.sum(), 1)
        if "Label" in df.columns:
            try:
                tmp = df[["Label", "dataset_source"]].copy()
                tmp["Label"] = tmp["Label"].astype(str).str.strip()
                tmp = tmp[~tmp["Label"].map(is_invalid_label_value)]
                snap["label_source_counts"] = pd.crosstab(tmp["Label"], tmp["dataset_source"])
            except Exception:
                pass

    # 缺失/无穷/零值比例，体现清洗、填补、标准化阶段效果
    try:
        if numeric_cols_all:
            n_for_quality = min(len(df), max_sample)
            q_df = df.iloc[np.random.default_rng(cfg.SEED).choice(len(df), n_for_quality, replace=False)][numeric_cols_all] if len(df) > n_for_quality else df[numeric_cols_all]
            arr = q_df.to_numpy(dtype=np.float64, copy=False)
            snap["missing_rate"] = float(np.isnan(arr).mean()) if arr.size else 0.0
            snap["inf_rate"] = float(np.isinf(arr).mean()) if arr.size else 0.0
            finite = np.isfinite(arr)
            snap["zero_rate"] = float((arr[finite] == 0).mean()) if finite.any() else 0.0
    except Exception:
        pass

    if feature_cols is None:
        selected_features = _viz_numeric_columns(df, max_cols=int(getattr(cfg, "VIZ_FEATURE_TOP_N", 12)))
    else:
        selected_features = [c for c in feature_cols if c in df.columns and pd.api.types.is_numeric_dtype(df[c])]
        if len(selected_features) > int(getattr(cfg, "VIZ_FEATURE_TOP_N", 12)):
            selected_features = selected_features[:int(getattr(cfg, "VIZ_FEATURE_TOP_N", 12))]
        if not selected_features:
            selected_features = _viz_numeric_columns(df, max_cols=int(getattr(cfg, "VIZ_FEATURE_TOP_N", 12)))
    snap["feature_cols"] = selected_features

    sample_cols = []
    for c in ["Label", "dataset_source"] + selected_features:
        if c in df.columns and c not in sample_cols:
            sample_cols.append(c)
    try:
        n_sample = min(len(df), max_sample)
        sample_df = df.iloc[np.random.default_rng(cfg.SEED).choice(len(df), n_sample, replace=False)][sample_cols] if len(df) > n_sample else df[sample_cols].copy()
        snap["sample_df"] = sample_df.reset_index(drop=True)
    except Exception:
        snap["sample_df"] = pd.DataFrame()

    try:
        if selected_features:
            stats_rows = []
            for col in selected_features:
                s = pd.to_numeric(snap["sample_df"][col], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
                if len(s) == 0:
                    continue
                stats_rows.append({
                    "stage": stage_name,
                    "feature": col,
                    "mean": float(s.mean()),
                    "std": float(s.std()) if len(s) > 1 else 0.0,
                    "median": float(s.median()),
                    "q25": float(s.quantile(0.25)),
                    "q75": float(s.quantile(0.75)),
                    "min": float(s.min()),
                    "max": float(s.max()),
                })
            snap["feature_stats"] = pd.DataFrame(stats_rows)
            snap["feature_stats_basis"] = "sampled_training_or_stage_rows_not_full_population"
            snap["feature_stats_sample_size"] = len(snap["sample_df"])
    except Exception:
        pass

    return snap


def _stage_ordered_items(stage_snapshots: Dict[str, Dict[str, Any]]):
    return [(k, stage_snapshots[k]) for k in sorted(stage_snapshots.keys())]


def _stage_display_name(stage: str) -> str:
    return str(stage).replace("_", " ")


def export_stage_snapshot_tables(stage_snapshots: Dict[str, Dict[str, Any]], save_dir: str) -> None:
    """导出所有阶段的统计表，便于论文附录或方法章节引用。"""
    os.makedirs(save_dir, exist_ok=True)
    summary_rows = []
    label_prop_rows = []
    source_prop_rows = []
    feature_stats_rows = []
    for stage, snap in _stage_ordered_items(stage_snapshots):
        summary_rows.append({
            "Stage": stage,
            "Rows": snap.get("rows", 0),
            "Columns": snap.get("cols", 0),
            "Numeric_Features": snap.get("numeric_feature_count", 0),
            "Missing_Rate": snap.get("missing_rate", 0.0),
            "Inf_Rate": snap.get("inf_rate", 0.0),
            "Zero_Rate": snap.get("zero_rate", 0.0),
        })
        props = snap.get("label_props", pd.Series(dtype=float))
        for label, value in props.items():
            label_prop_rows.append({"Stage": stage, "Label": label, "Proportion": float(value)})
        src_props = snap.get("source_props", pd.Series(dtype=float))
        for src, value in src_props.items():
            source_prop_rows.append({"Stage": stage, "Source": src, "Proportion": float(value)})
        fstats = snap.get("feature_stats", pd.DataFrame())
        if isinstance(fstats, pd.DataFrame) and not fstats.empty:
            feature_stats_rows.extend(fstats.to_dict("records"))

    pd.DataFrame(summary_rows).to_csv(os.path.join(save_dir, "stage_overview_summary.csv"), index=False, encoding="utf-8-sig")
    pd.DataFrame(label_prop_rows).to_csv(os.path.join(save_dir, "stage_label_proportions.csv"), index=False, encoding="utf-8-sig")
    pd.DataFrame(source_prop_rows).to_csv(os.path.join(save_dir, "stage_source_proportions.csv"), index=False, encoding="utf-8-sig")
    if feature_stats_rows:
        pd.DataFrame(feature_stats_rows).to_csv(os.path.join(save_dir, "stage_feature_statistics.csv"), index=False, encoding="utf-8-sig")


def plot_stage_overview_dashboard(stage_snapshots: Dict[str, Dict[str, Any]], save_dir: str) -> None:
    """阶段总览：行数、特征数、缺失率、无穷值比例。"""
    items = _stage_ordered_items(stage_snapshots)
    if not items:
        return
    df = pd.DataFrame([
        {
            "Stage": _stage_display_name(stage),
            "Rows": snap.get("rows", 0),
            "Numeric_Features": snap.get("numeric_feature_count", 0),
            "Missing_Rate": snap.get("missing_rate", 0.0) * 100,
            "Inf_Rate": snap.get("inf_rate", 0.0) * 100,
        }
        for stage, snap in items
    ])
    setup_thesis_plot_style()
    fig, axes = plt.subplots(2, 2, figsize=(13.2, max(7.2, 0.44 * len(df))))
    axes = axes.ravel()
    sns.barplot(data=df, y="Stage", x="Rows", ax=axes[0], color=EI_COLORS["blue"], edgecolor="white", linewidth=0.35)
    axes[0].set_xscale("log")
    axes[0].set_title("Rows retained across processing stages")
    axes[0].set_xlabel("Rows (log scale)")
    axes[0].set_ylabel("")

    sns.barplot(data=df, y="Stage", x="Numeric_Features", ax=axes[1], color=EI_COLORS["cyan"], edgecolor="white", linewidth=0.35)
    axes[1].set_title("Numeric feature count across stages")
    axes[1].set_xlabel("Number of numeric features")
    axes[1].set_ylabel("")

    sns.barplot(data=df, y="Stage", x="Missing_Rate", ax=axes[2], color=EI_COLORS["orange"], edgecolor="white", linewidth=0.35)
    axes[2].set_title("Missing value ratio across stages")
    axes[2].set_xlabel("Missing values (%)")
    axes[2].set_ylabel("")

    sns.barplot(data=df, y="Stage", x="Inf_Rate", ax=axes[3], color=EI_COLORS["red"], edgecolor="white", linewidth=0.35)
    axes[3].set_title("Infinite value ratio across stages")
    axes[3].set_xlabel("Infinite values (%)")
    axes[3].set_ylabel("")
    for idx, (ax, kind) in enumerate(zip(axes, ["count", "count", "percent", "percent"])):
        annotate_horizontal_bars(ax, fontsize=6.8, kind=kind)
        ax.margins(x=0.18)
        style_axis(ax, "x")
        add_panel_label(ax, f"({chr(97 + idx)})")
    fig.suptitle("Processing pipeline overview", fontsize=13, fontweight="semibold", y=1.01)
    fig.tight_layout()
    save_thesis_figure(fig, os.path.join(save_dir, "A_stage_overview_dashboard.png"))


def plot_normal_attack_proportion_flow(stage_snapshots: Dict[str, Dict[str, Any]], save_dir: str) -> None:
    """Normal/Attack占比随处理流程变化，强调比例而非单纯数量。"""
    rows = []
    for stage, snap in _stage_ordered_items(stage_snapshots):
        props = snap.get("normal_attack_props", pd.Series(dtype=float))
        if props.empty:
            continue
        for label in ["Normal", "Attack"]:
            rows.append({"Stage": _stage_display_name(stage), "Type": label, "Proportion": float(props.get(label, 0.0))})
    if not rows:
        return
    df = pd.DataFrame(rows)
    pivot = df.pivot_table(index="Stage", columns="Type", values="Proportion", fill_value=0)
    pivot = pivot[[c for c in ["Normal", "Attack"] if c in pivot.columns]]
    setup_thesis_plot_style()
    fig, ax = plt.subplots(figsize=(10.5, max(5.0, 0.38 * len(pivot))))
    left = np.zeros(len(pivot))
    colors = {"Normal": EI_COLORS["blue"], "Attack": EI_COLORS["red"]}
    y = np.arange(len(pivot))
    for col in pivot.columns:
        vals = pivot[col].values * 100
        ax.barh(y, vals, left=left, label=col, color=colors.get(col), edgecolor="white", linewidth=0.5)
        for i, v in enumerate(vals):
            if v >= 3:
                ax.text(left[i] + v / 2, i, f"{v:.1f}%", va="center", ha="center", fontsize=7.2,
                        color="white" if col == "Attack" else "#111111", fontweight="semibold")
            elif v > 0:
                ax.annotate(f"{v:.1f}%", (left[i] + v, i), xytext=(3, 0), textcoords="offset points",
                            va="center", ha="left", fontsize=6.6, clip_on=False)
        left += vals
    ax.set_yticks(y)
    ax.set_yticklabels(pivot.index)
    ax.set_xlim(0, 100)
    ax.set_xlabel("Proportion (%)")
    ax.set_ylabel("")
    ax.set_title("Normal and attack traffic proportions across processing stages")
    ax.legend(loc="lower right", frameon=False, ncol=2)
    style_axis(ax, "x")
    save_thesis_figure(fig, os.path.join(save_dir, "B_normal_attack_proportion_flow.png"))


def plot_label_proportion_heatmap(stage_snapshots: Dict[str, Dict[str, Any]], save_dir: str) -> None:
    """多类别占比热力图：一眼看出每一步类别结构如何变化。"""
    labels_all = []
    for _, snap in _stage_ordered_items(stage_snapshots):
        labels_all.extend(list(snap.get("label_props", pd.Series(dtype=float)).index))
    labels = ordered_standard_labels(labels_all)
    if not labels:
        return
    stages = []
    matrix = []
    for stage, snap in _stage_ordered_items(stage_snapshots):
        props = snap.get("label_props", pd.Series(dtype=float))
        if props.empty:
            continue
        stages.append(_stage_display_name(stage))
        matrix.append([float(props.get(label, 0.0)) * 100 for label in labels])
    if not matrix:
        return
    heat = pd.DataFrame(matrix, index=stages, columns=labels)
    setup_thesis_plot_style()
    fig, ax = plt.subplots(figsize=(max(11.5, 0.72 * len(labels)), max(5.8, 0.44 * len(stages))))
    sns.heatmap(heat, ax=ax, cmap="Blues", linewidths=0.5, linecolor="white",
                annot=True, fmt=".1f", annot_kws={"fontsize": 6.5},
                cbar_kws={"label": "Class proportion (%)", "shrink": 0.82})
    ax.set_title("Class proportions across processing stages", pad=10)
    ax.set_xlabel("Class label")
    ax.set_ylabel("Processing stage")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=35, ha="right")
    save_thesis_figure(fig, os.path.join(save_dir, "C_class_proportion_heatmap.png"))


def plot_label_count_heatmap(stage_snapshots: Dict[str, Dict[str, Any]], save_dir: str) -> None:
    """类别数量热力图：使用log10(count+1)，辅助展示小类变化。"""
    labels_all = []
    for _, snap in _stage_ordered_items(stage_snapshots):
        labels_all.extend(list(snap.get("label_counts", pd.Series(dtype=int)).index))
    labels = ordered_standard_labels(labels_all)
    if not labels:
        return
    stages = []
    matrix = []
    count_matrix = []
    for stage, snap in _stage_ordered_items(stage_snapshots):
        counts = snap.get("label_counts", pd.Series(dtype=int))
        if counts.empty:
            continue
        stages.append(_stage_display_name(stage))
        raw_counts = [int(counts.get(label, 0)) for label in labels]
        count_matrix.append(raw_counts)
        matrix.append([np.log10(float(v) + 1.0) for v in raw_counts])
    if not matrix:
        return
    heat = pd.DataFrame(matrix, index=stages, columns=labels)
    annotations = pd.DataFrame(count_matrix, index=stages, columns=labels).map(lambda x: f"{int(x):,}")
    setup_thesis_plot_style()
    fig, ax = plt.subplots(figsize=(max(12.5, 0.80 * len(labels)), max(6.0, 0.46 * len(stages))))
    sns.heatmap(heat, ax=ax, cmap="YlGnBu", linewidths=0.5, linecolor="white",
                annot=annotations, fmt="", annot_kws={"fontsize": 6.0},
                cbar_kws={"label": "Color scale: log10(count + 1)", "shrink": 0.82})
    ax.set_title("Exact class counts across processing stages", pad=10)
    ax.set_xlabel("Class label")
    ax.set_ylabel("Processing stage")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=35, ha="right")
    save_thesis_figure(fig, os.path.join(save_dir, "D_class_count_log_heatmap.png"))


def plot_before_after_class_delta(stage_snapshots: Dict[str, Dict[str, Any]], save_dir: str) -> None:
    """最关键的一眼图：基准阶段与最终平衡阶段的类别占比变化。"""
    items = [(s, snap) for s, snap in _stage_ordered_items(stage_snapshots) if not snap.get("label_counts", pd.Series(dtype=int)).empty]
    if len(items) < 2:
        return
    # 优先选合并/标准化后的真实数据作为before，选SMOTE后/最终平衡作为after
    before_candidates = [x for x in items if "Merged" in x[0] or "RobustScaled" in x[0] or "BeforeNormal" in x[0]]
    after_candidates = [x for x in items if "AfterSMOTE" in x[0] or "FinalBalanced" in x[0] or "AfterIF" in x[0]]
    before_stage, before_snap = before_candidates[0] if before_candidates else items[0]
    after_stage, after_snap = after_candidates[-1] if after_candidates else items[-1]
    before_props = before_snap.get("label_props", pd.Series(dtype=float))
    after_props = after_snap.get("label_props", pd.Series(dtype=float))
    labels = ordered_standard_labels(list(set(before_props.index) | set(after_props.index)))
    rows = []
    for label in labels:
        b = float(before_props.get(label, 0.0)) * 100
        a = float(after_props.get(label, 0.0)) * 100
        rows.append({"Label": label, "Before": b, "After": a, "Delta": a - b})
    df = pd.DataFrame(rows).sort_values("Delta")
    setup_thesis_plot_style()
    fig, ax = plt.subplots(figsize=(10.2, max(5.2, 0.36 * len(df))))
    colors = [EI_COLORS["green"] if v >= 0 else EI_COLORS["red"] for v in df["Delta"]]
    ax.barh(df["Label"], df["Delta"], color=colors, edgecolor="white", linewidth=0.4, zorder=3)
    ax.axvline(0, color="#333333", linewidth=0.9)
    ax.set_xlabel("Change in class proportion (percentage points)")
    ax.set_ylabel("")
    ax.set_title(f"Class proportion change: {_stage_display_name(before_stage)} to {_stage_display_name(after_stage)}")
    for patch, value in zip(ax.patches, df["Delta"]):
        ax.annotate(_format_plot_value(value, "signed_percent"),
                    (patch.get_x() + patch.get_width(), patch.get_y() + patch.get_height()/2),
                    xytext=(4 if value >= 0 else -4, 0), textcoords="offset points",
                    ha="left" if value >= 0 else "right", va="center", fontsize=6.8, clip_on=False)
    ax.margins(x=0.22)
    style_axis(ax, "x")
    save_thesis_figure(fig, os.path.join(save_dir, "E_before_after_class_proportion_delta.png"))

    long_df = df.melt(id_vars="Label", value_vars=["Before", "After"], var_name="Stage", value_name="Proportion")
    fig, ax = plt.subplots(figsize=(10.2, max(5.2, 0.36 * len(df))))
    sns.barplot(data=long_df, y="Label", x="Proportion", hue="Stage", ax=ax,
                palette={"Before": EI_COLORS["light_blue"], "After": EI_COLORS["green"]}, edgecolor="white", linewidth=0.35)
    ax.set_xlabel("Class proportion (%)")
    ax.set_ylabel("")
    ax.set_title("Class composition before and after processing")
    annotate_horizontal_bars(ax, fontsize=6.6, kind="percent")
    ax.margins(x=0.18)
    style_axis(ax, "x")
    ax.legend(frameon=False, title="", ncol=2)
    save_thesis_figure(fig, os.path.join(save_dir, "F_before_after_class_proportion_bars.png"))


def plot_source_proportion_flow(stage_snapshots: Dict[str, Dict[str, Any]], save_dir: str) -> None:
    """2017/2018来源占比随流程变化，体现数据源平衡功能。"""
    rows = []
    all_sources = set()
    for stage, snap in _stage_ordered_items(stage_snapshots):
        props = snap.get("source_props", pd.Series(dtype=float))
        if props.empty:
            continue
        all_sources.update(props.index.astype(str).tolist())
        for source, val in props.items():
            rows.append({"Stage": _stage_display_name(stage), "Source": str(source), "Proportion": float(val) * 100})
    if not rows:
        return
    df = pd.DataFrame(rows)
    pivot = df.pivot_table(index="Stage", columns="Source", values="Proportion", fill_value=0)
    setup_thesis_plot_style()
    fig, ax = plt.subplots(figsize=(10.5, max(5.0, 0.38 * len(pivot))))
    left = np.zeros(len(pivot))
    palette = EI_SEQUENCE[:len(pivot.columns)]
    y = np.arange(len(pivot))
    for i, col in enumerate(pivot.columns):
        vals = pivot[col].values
        ax.barh(y, vals, left=left, label=str(col), color=palette[i], edgecolor="white", linewidth=0.5)
        for j, v in enumerate(vals):
            if v >= 4:
                ax.text(left[j] + v / 2, j, f"{v:.1f}%", va="center", ha="center", fontsize=7.2,
                        color="white" if i == 0 else "#111111", fontweight="semibold")
            elif v > 0:
                ax.annotate(f"{v:.1f}%", (left[j] + v, j), xytext=(3, 0), textcoords="offset points",
                            va="center", fontsize=6.5, clip_on=False)
        left += vals
    ax.set_yticks(y)
    ax.set_yticklabels(pivot.index)
    ax.set_xlim(0, 100)
    ax.set_xlabel("Dataset source proportion (%)")
    ax.set_ylabel("")
    ax.set_title("Dataset-source proportions across processing stages")
    ax.legend(title="Source", frameon=False, loc="lower right", ncol=min(3, len(pivot.columns)))
    style_axis(ax, "x")
    save_thesis_figure(fig, os.path.join(save_dir, "G_dataset_source_proportion_flow.png"))


def plot_latest_label_source_composition(stage_snapshots: Dict[str, Dict[str, Any]], save_dir: str) -> None:
    """最终阶段每类流量来自2017/2018的比例热力图。"""
    items = [(stage, snap) for stage, snap in _stage_ordered_items(stage_snapshots) if isinstance(snap.get("label_source_counts"), pd.DataFrame) and not snap.get("label_source_counts").empty]
    if not items:
        return
    stage, snap = items[-1]
    table = snap.get("label_source_counts").copy()
    labels = ordered_standard_labels(table.index.tolist())
    table = table.reindex(labels).fillna(0)
    prop = table.div(table.sum(axis=1).replace(0, np.nan), axis=0).fillna(0) * 100
    setup_thesis_plot_style()
    fig, ax = plt.subplots(figsize=(max(7.0, 1.4 * len(prop.columns)), max(5.8, 0.36 * len(prop))))
    sns.heatmap(prop, annot=True, fmt=".1f", cmap="Blues", linewidths=0.5, linecolor="white",
                annot_kws={"fontsize": 7.2}, cbar_kws={"label": "Within-class source proportion (%)", "shrink": 0.82}, ax=ax)
    ax.set_title(f"Within-class dataset-source composition: {_stage_display_name(stage)}", pad=10)
    ax.set_xlabel("Dataset source")
    ax.set_ylabel("Class label")
    save_thesis_figure(fig, os.path.join(save_dir, "H_latest_label_source_composition.png"))


def plot_feature_distribution_shift(stage_snapshots: Dict[str, Dict[str, Any]], save_dir: str) -> None:
    """关键数值特征前后分布对比，体现log/scale/AE前后的分布变化。"""
    items = [(stage, snap) for stage, snap in _stage_ordered_items(stage_snapshots) if isinstance(snap.get("sample_df"), pd.DataFrame) and not snap.get("sample_df").empty]
    if len(items) < 2:
        return
    before_candidates = [x for x in items if "Merged" in x[0] or "LogTransform" in x[0] or "RobustScaled" in x[0]]
    after_candidates = [x for x in items if "AfterSMOTE" in x[0] or "AfterIF" in x[0] or "FinalBalanced" in x[0]]
    before_stage, before_snap = before_candidates[0] if before_candidates else items[0]
    after_stage, after_snap = after_candidates[-1] if after_candidates else items[-1]
    bdf = before_snap.get("sample_df", pd.DataFrame())
    adf = after_snap.get("sample_df", pd.DataFrame())
    common_features = [c for c in before_snap.get("feature_cols", []) if c in after_snap.get("feature_cols", []) and c in bdf.columns and c in adf.columns]
    # 如果AE之后没有共同原始特征，就对各自代表性特征画统计箱线图，避免空图
    setup_thesis_plot_style()
    if common_features:
        features = common_features[:min(6, len(common_features))]
        n_cols = 2
        n_rows = int(np.ceil(len(features) / n_cols))
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, max(4.2, 3.0 * n_rows)))
        axes = np.array(axes).reshape(-1)
        for ax, feature in zip(axes, features):
            b = pd.to_numeric(bdf[feature], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
            a = pd.to_numeric(adf[feature], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
            if len(b) == 0 or len(a) == 0:
                ax.set_visible(False)
                continue
            # 极端长尾用分位数截断做展示，不影响原始数据
            combined = pd.concat([b, a])
            lo, hi = combined.quantile([0.01, 0.99])
            b = b.clip(lo, hi)
            a = a.clip(lo, hi)
            sns.kdeplot(b, ax=ax, label=f"{_stage_display_name(before_stage)} (n={len(b):,})",
                        color=EI_COLORS["blue"], fill=True, alpha=0.18, linewidth=1.6)
            sns.kdeplot(a, ax=ax, label=f"{_stage_display_name(after_stage)} (n={len(a):,})",
                        color=EI_COLORS["orange"], fill=True, alpha=0.16, linewidth=1.6)
            ax.set_title(feature)
            ax.set_xlabel("Feature value (1%-99% clipped for display)")
            ax.set_ylabel("Density")
            style_axis(ax, "y")
            ax.legend(frameon=False, fontsize=7.0)
        for ax in axes[len(features):]:
            ax.set_visible(False)
        fig.suptitle("Representative feature-distribution shift before and after processing",
                     fontsize=12.5, fontweight="semibold", y=1.01)
        fig.tight_layout()
        save_thesis_figure(fig, os.path.join(save_dir, "I_feature_distribution_shift_before_after.png"))
    else:
        # AE替换后特征名不一致时，使用各阶段特征统计的箱线图展示尺度压缩效果
        stats_rows = []
        for stage, snap in [(before_stage, before_snap), (after_stage, after_snap)]:
            fs = snap.get("feature_stats", pd.DataFrame())
            if isinstance(fs, pd.DataFrame) and not fs.empty:
                fs = fs.copy()
                fs["Stage"] = _stage_display_name(stage)
                stats_rows.append(fs)
        if not stats_rows:
            return
        stats = pd.concat(stats_rows, ignore_index=True)
        fig, ax = plt.subplots(figsize=(9.5, 5.5))
        sns.boxplot(data=stats, x="Stage", y="std", ax=ax, color=EI_COLORS["cyan"],
                    width=0.5, linewidth=0.8, fliersize=2.5)
        ax.set_title("Feature standard-deviation summary before and after processing")
        ax.set_xlabel("")
        ax.set_ylabel("Feature standard deviation")
        medians = stats.groupby("Stage", sort=False)["std"].median()
        for i, (_, value) in enumerate(medians.items()):
            ax.annotate(f"Median = {value:.3g}", (i, value), xytext=(0, 9), textcoords="offset points",
                        ha="center", fontsize=7.0)
        style_axis(ax, "y")
        save_thesis_figure(fig, os.path.join(save_dir, "I_feature_scale_summary_before_after.png"))


def plot_split_label_proportions(df_train: Optional[pd.DataFrame],
                                 df_val: Optional[pd.DataFrame],
                                 df_test: Optional[pd.DataFrame],
                                 save_dir: str) -> None:
    """训练/验证/测试集标签占比对比，验证分层切分是否保持分布。"""
    frames = {"Train": df_train, "Validation": df_val, "Test": df_test}
    labels_all = []
    for df in frames.values():
        if df is not None and len(df) > 0 and "Label" in df.columns:
            labels_all.extend(_viz_safe_value_counts(df["Label"]).index.tolist())
    labels = ordered_standard_labels(labels_all)
    if not labels:
        return
    rows = []
    for split, df in frames.items():
        if df is None or len(df) == 0 or "Label" not in df.columns:
            continue
        counts = _viz_safe_value_counts(df["Label"])
        props = counts / max(counts.sum(), 1)
        for label in labels:
            rows.append({"Split": split, "Label": label, "Proportion": float(props.get(label, 0.0)) * 100})
    if not rows:
        return
    plot_df = pd.DataFrame(rows)
    heat = plot_df.pivot_table(index="Label", columns="Split", values="Proportion", fill_value=0).reindex(labels)
    setup_thesis_plot_style()
    fig, ax = plt.subplots(figsize=(7.5, max(5.5, 0.34 * len(labels))))
    sns.heatmap(heat, annot=True, fmt=".2f", cmap="Blues", linewidths=0.5, linecolor="white",
                annot_kws={"fontsize": 7.0}, cbar_kws={"label": "Class proportion (%)", "shrink": 0.82}, ax=ax)
    ax.set_title("Class-proportion consistency across train/validation/test splits", pad=10)
    ax.set_xlabel("Dataset split")
    ax.set_ylabel("Class label")
    save_thesis_figure(fig, os.path.join(save_dir, "J_train_val_test_class_proportion_heatmap.png"))


def generate_comprehensive_visualization_report(stage_snapshots: Dict[str, Dict[str, Any]],
                                                save_dir: str,
                                                df_train: Optional[pd.DataFrame] = None,
                                                df_val: Optional[pd.DataFrame] = None,
                                                df_test: Optional[pd.DataFrame] = None) -> None:
    """统一生成论文增强可视化报告。"""
    if not getattr(get_active_config(), "EXPORT_COMPREHENSIVE_STAGE_VISUALS", True):
        return
    report_dir = os.path.join(save_dir, "comprehensive_stage_report")
    os.makedirs(report_dir, exist_ok=True)
    tasks = [
        (
            "stage_table",
            lambda: export_stage_snapshot_tables(
                stage_snapshots, report_dir
            )
        ),

        (
            "stage_dashboard",
            lambda: plot_stage_overview_dashboard(
                stage_snapshots, report_dir
            )
        ),

        (
            "normal_attack_flow",
            lambda: plot_normal_attack_proportion_flow(
                stage_snapshots, report_dir
            )
        ),

        (
            "label_prop_heatmap",
            lambda: plot_label_proportion_heatmap(
                stage_snapshots, report_dir
            )
        ),

        (
            "label_count_heatmap",
            lambda: plot_label_count_heatmap(
                stage_snapshots, report_dir
            )
        ),

        (
            "class_delta",
            lambda: plot_before_after_class_delta(
                stage_snapshots, report_dir
            )
        ),

        (
            "source_flow",
            lambda: plot_source_proportion_flow(
                stage_snapshots, report_dir
            )
        ),

        (
            "source_composition",
            lambda: plot_latest_label_source_composition(
                stage_snapshots, report_dir
            )
        ),

        (
            "feature_shift",
            lambda: plot_feature_distribution_shift(
                stage_snapshots, report_dir
            )
        ),

        (
            "split_prop",
            lambda: plot_split_label_proportions(
                df_train,
                df_val,
                df_test,
                report_dir
            )
        ),

    ]


    for name, func in tasks:

        try:
            func()
            logger.info(
                f"[综合可视化] {name} 完成"
            )

        except Exception as exc:
            logger.warning(
                f"[综合可视化] {name}失败，跳过该图: {exc}",
                exc_info=True
            )


    logger.info(
        f"综合阶段可视化报告已生成: {report_dir}"
    )

    print(
        f"[综合可视化] 已生成阶段报告目录: {report_dir}"
    )


def plan_pre_ae_normal_sampling(normal_counts: Dict[str, int], attack_counts: Dict[str, int], cfg) -> Dict[str, Any]:
    """联合规划Normal总量和来源配额；攻击计数固定，不靠删攻击凑来源比例。

    两个真实来源时，在Normal保留总量固定的可行区间内，选择最接近
    SOURCE_BALANCE_TARGET（2017占比）的整数配额。无法同时满足两个目标时
    返回最近可行解和未达标原因，而不是伪造目标来源比例。
    """
    sources = sorted(set(normal_counts) | set(attack_counts))
    normals = {s: int(normal_counts.get(s, 0)) for s in sources}
    attacks = {s: int(attack_counts.get(s, 0)) for s in sources}
    if any(v < 0 for v in list(normals.values()) + list(attacks.values())):
        raise ValueError("前置采样计数不能为负数")
    ratio = float(getattr(cfg, "NORMAL_MAX_ATTACK_RATIO", 5.0))
    target = float(getattr(cfg, "SOURCE_BALANCE_TARGET", 0.3))
    tolerance = float(getattr(cfg, "SOURCE_BALANCE_TOLERANCE", 0.05))
    if not np.isfinite(ratio) or ratio <= 0:
        raise ValueError("NORMAL_MAX_ATTACK_RATIO必须为正数")
    if not 0 < target < 1 or not 0 <= tolerance < 1:
        raise ValueError("数据来源目标占比或容差无效")
    n_normal, n_attack = sum(normals.values()), sum(attacks.values())
    requested_normal = int(np.ceil(n_attack * ratio)) if n_attack else n_normal
    keep_normal = min(n_normal, requested_normal)
    active_sources = {s for s in sources if normals[s] + attacks[s] > 0}
    applicable = active_sources == {"2017", "2018"}
    quotas = dict(normals)
    share_range = None

    if applicable:
        # N17+N18=K，且0<=N17<=原N17、0<=N18<=原N18。
        # (A17+N17)/(A17+A18+K)决定采样后的2017来源占比。
        lower = max(0, keep_normal - normals["2018"])
        upper = min(normals["2017"], keep_normal)
        total_after = n_attack + keep_normal
        desired_2017 = target * total_after - attacks["2017"]
        # 等距离时多留一个2017 Normal，优先削减2018冗余Normal。
        keep_2017 = min(upper, max(lower, int(np.floor(desired_2017 + 0.5))))
        quotas["2017"], quotas["2018"] = keep_2017, keep_normal - keep_2017
        if total_after:
            share_range = [(attacks["2017"] + lower) / total_after,
                           (attacks["2017"] + upper) / total_after]
    elif n_normal and keep_normal < n_normal:
        # 单来源/未知来源仍能安全压缩Normal，但不宣称完成2017/2018平衡。
        raw = {s: normals[s] * keep_normal / n_normal for s in sources}
        quotas = {s: int(np.floor(raw[s])) for s in sources}
        remainder = keep_normal - sum(quotas.values())
        for s in sorted(sources, key=lambda s: (-(raw[s] - quotas[s]), s)):
            if remainder <= 0:
                break
            if quotas[s] < normals[s]:
                quotas[s] += 1
                remainder -= 1

    expected_sources = {s: attacks[s] + quotas[s] for s in sources}
    total_after = sum(expected_sources.values())
    achieved_share = expected_sources.get("2017", 0) / total_after if total_after else None
    goal_met = (abs(achieved_share - target) <= tolerance + 1e-12) if applicable else None
    reasons = []
    if n_attack == 0:
        reasons.append("no_attack_keep_all_normal")
    elif n_normal < requested_normal:
        reasons.append("insufficient_real_normal_keep_all")
    if not applicable:
        reasons.append("source_balance_requires_2017_and_2018_only")
    elif not goal_met:
        reasons.append("source_target_infeasible_with_fixed_normal_budget_and_all_attacks_kept")
    if sum(quotas.values()) != keep_normal or any(not 0 <= quotas[s] <= normals[s] for s in sources):
        raise RuntimeError("前置Normal采样配额不守恒")
    reason_messages = {
        "no_attack_keep_all_normal": "没有攻击样本，不清空Normal，保留全部真实样本",
        "insufficient_real_normal_keep_all": "真实Normal不足攻击总量的目标倍数，全部保留，不合成Normal也不删除攻击",
        "source_balance_requires_2017_and_2018_only": "来源不完整或含未知来源，不伪造2017/2018来源平衡",
        "source_target_infeasible_with_fixed_normal_budget_and_all_attacks_kept":
            "固定Normal目标数量并保留全部攻击时，来源目标不可达，采用最近可行配额",
    }
    return {
        "sampling_stage": "classification_train_after_split_before_preprocessing_and_AE",
        "normal_to_attack_target": ratio,
        "normal_before": n_normal, "attack_before": n_attack,
        "requested_normal": requested_normal, "normal_after": keep_normal,
        "normal_to_attack_after": keep_normal / n_attack if n_attack else None,
        "normal_goal_met": keep_normal == requested_normal if n_attack else None,
        "normal_removed": n_normal - keep_normal,
        "attack_after": n_attack, "all_attacks_retained": True,
        "normal_keep_by_source": quotas,
        "normal_before_by_source": normals, "attack_by_source": attacks,
        "source_counts_before": {s: normals[s] + attacks[s] for s in sources},
        "source_counts_after": expected_sources,
        "source_2017_target_share": target, "source_share_tolerance": tolerance,
        "source_2017_after_share": achieved_share,
        "source_2017_feasible_share_range": share_range,
        "source_balance_applicable": applicable, "source_goal_met": goal_met,
        "limitations": reasons,
        "limitation_messages": [reason_messages[reason] for reason in reasons],
    }


@timer_decorator
def sample_classification_train_before_ae(df: pd.DataFrame, cfg=None) -> pd.DataFrame:
    """只删除分类训练集Normal：先联合定配额，再按来源一次随机抽样。

    不复制/合成Normal，不截断攻击，不修改val/test或读取时采集的时序窗口。
    不恢复旧阶段5的安全截断、全量聚类和临时BorderlineSMOTE。
    """
    cfg = get_active_config() if cfg is None else cfg
    if "Label" not in df.columns:
        raise ValueError("前置采样需要Label列")
    labels = df["Label"]
    normal_values = [v for v in labels.unique() if is_normal_label_value(v)]
    normal_mask = labels.isin(normal_values).to_numpy()
    sources = (df["dataset_source"].astype(str).to_numpy()
               if "dataset_source" in df.columns else np.full(len(df), "unknown", dtype=object))
    normal_counts = pd.Series(sources[normal_mask], dtype=object).value_counts().to_dict()
    attack_counts = pd.Series(sources[~normal_mask], dtype=object).value_counts().to_dict()
    plan = plan_pre_ae_normal_sampling(normal_counts, attack_counts, cfg)
    keep_mask = ~normal_mask
    rng = np.random.default_rng(int(cfg.SEED))
    for source, keep_count in plan["normal_keep_by_source"].items():
        positions = np.flatnonzero(normal_mask & (sources == source))
        if keep_count >= len(positions):
            keep_mask[positions] = True
        elif keep_count > 0:
            # 抽取保留/删除两者中更小的一侧，降低接近全保留时的随机索引开销。
            if len(positions) - keep_count < keep_count:
                keep_mask[positions] = True
                keep_mask[rng.choice(positions, size=len(positions) - keep_count, replace=False)] = False
            else:
                keep_mask[rng.choice(positions, size=keep_count, replace=False)] = True
    kept_positions = np.flatnonzero(keep_mask)
    result = df if len(kept_positions) == len(df) else df.take(kept_positions)
    # 位置级检查：不仅防止攻击类别消失，还必须保留每一条攻击样本。
    attacks_retained = bool(np.all(keep_mask[~normal_mask]))
    if not attacks_retained or len(result) != plan["attack_before"] + plan["normal_after"]:
        raise RuntimeError("前置采样违反攻击全保留/样本总数守恒约束")
    plan["attack_retention_verified"] = attacks_retained
    plan["sampling_counts_verified"] = True
    plan["sampling_seed"] = int(cfg.SEED)
    plan["sampling_method"] = "source_stratified_random_normal_only"
    logger.info("[前置采样] Normal %s -> %s，Attack %s -> %s（全部保留），Normal:Attack目标=%.1f:1",
                f"{plan['normal_before']:,}", f"{plan['normal_after']:,}",
                f"{plan['attack_before']:,}", f"{plan['attack_after']:,}", plan["normal_to_attack_target"])
    logger.info("[前置来源平衡] %s -> %s；来源目标达标=%s",
                plan["source_counts_before"], plan["source_counts_after"], plan["source_goal_met"])
    if plan["limitations"]:
        logger.warning("[前置采样限制] %s；保留真实样本，不删除攻击或伪造来源。可行2017占比范围=%s",
                       "; ".join(plan["limitation_messages"]), plan["source_2017_feasible_share_range"])
    try:
        audit_dir = os.path.join(cfg.OUTPUT_DIR, "balance_debug_logs")
        os.makedirs(audit_dir, exist_ok=True)
        with open(os.path.join(audit_dir, "pre_ae_sampling_plan.json"), "w", encoding="utf-8") as handle:
            json.dump(plan, handle, ensure_ascii=False, indent=2)
        metadata = pd.DataFrame({"dataset_source": sources, "Label": labels.to_numpy()})
        before = metadata.value_counts(sort=False).rename("Before")
        after = metadata.take(kept_positions).value_counts(sort=False).rename("After")
        audit = pd.concat([before, after], axis=1).fillna(0).astype(np.int64)
        audit["Removed"] = audit["Before"] - audit["After"]
        audit.reset_index().to_csv(os.path.join(audit_dir, "pre_ae_source_label_counts.csv"),
                                   index=False, encoding="utf-8-sig")
    except Exception as exc:
        logger.warning("保存前置采样审计失败: %s", exc)
    return result


def save_latest_outputs(cfg, df_train, df_val, df_test, df_balanced):
    os.makedirs(cfg.MAIN_OUTPUT_DIR,exist_ok=True)
    temporary=str(cfg.LATEST_SUMMARY)+".tmp"
    with open(temporary,"w",encoding="utf-8") as handle:
        handle.write(f"latest_run={cfg.OUTPUT_DIR}\ndata_files_saved_only_in_timestamp_dir=true\n")
        for name,path,frame in [("train",cfg.OUTPUT_TRAIN,df_train),("val",cfg.OUTPUT_VAL,df_val),
                                ("test",cfg.OUTPUT_TEST,df_test),("balanced",cfg.OUTPUT_BALANCED,df_balanced)]:
            handle.write(f"{name}_csv={path}\n{name}_rows={len(frame)}\n")
        handle.write(f"balanced_is_training_alias=true\nlabel_vocabulary_size={len(cfg.STANDARD_LABELS)}\n")
        handle.write(f"forecast_rows_per_window={cfg.FORECAST_ROWS_PER_WINDOW}\nflow_order_only=true\n")
    os.replace(temporary,cfg.LATEST_SUMMARY)


# -------------------------- 2. 核心工具函数 --------------------------
def get_memory_usage() -> float:
    """获取当前进程的内存占用（单位：GB）"""
    process = psutil.Process()
    mem_usage = process.memory_info().rss / (1024 ** 3)
    return round(mem_usage, 2)


def get_memory_percent() -> float:
    """获取当前运行环境的真实内存使用率。

    容器/cgroup v2环境优先读取 memory.current / memory.max，避免
    psutil.virtual_memory()错误使用宿主机总内存，导致120GiB容器的保护阈值失效。
    返回百分比(0~100)。
    """
    try:
        current_path = "/sys/fs/cgroup/memory.current"
        max_path = "/sys/fs/cgroup/memory.max"
        if os.path.exists(current_path) and os.path.exists(max_path):
            with open(current_path, "r", encoding="utf-8") as f:
                current = int(f.read().strip())
            with open(max_path, "r", encoding="utf-8") as f:
                limit_text = f.read().strip()
            if limit_text and limit_text != "max":
                limit = int(limit_text)
                if limit > 0:
                    return float(current / limit * 100.0)
    except Exception as exc:
        logger.debug(f"读取cgroup内存使用率失败，回退psutil: {exc}")
    return float(psutil.virtual_memory().percent)


def log_memory_usage(stage: str):
    """记录当前内存占用"""
    mem_usage = get_memory_usage()
    mem_percent = get_memory_percent()
    logger.info(f"【内存监控】{stage} - 进程内存：{mem_usage} GB | 系统内存使用率：{mem_percent:.1f}%")


def release_memory():
    """强制释放内存"""
    import gc
    gc.collect()
    logger.info("【内存管理】已强制执行垃圾回收，释放内存")


def remove_duplicate_header_rows(df: pd.DataFrame, dataset_name: str) -> pd.DataFrame:
    """删除意外出现的特征值行（重复的表头行）
    
    只删除真正的表头行复制：当最后一列的值与第一行完全相同，且是表头关键词
    """
    logger.info("检查并删除重复的表头行...")
    
    original_rows = len(df)
    column_names = df.columns.tolist()
    
    print(f"[进度] 开始检查表头行，数据量：{len(df)}...")
    
    # 检查是否为空
    if len(df) == 0:
        return df
    
    # 只检查最后一列
    if len(column_names) > 0:
        last_col = column_names[-1]
        first_row_value = str(df.iloc[0][last_col]).strip()
        first_row_lower = first_row_value.lower()
        
        # 检查第一行值是否是表头关键词（精确匹配）
        header_keywords = ["label", "lable"]
        is_header_value = first_row_lower in header_keywords
        
        if is_header_value:
            print(f"[进度] 第一行最后一列值 '{first_row_value}' 是表头关键词，检查重复行...")
            
            # 删除与第一行完全相同的行
            header_rows = []
            for idx in range(1, min(1000, len(df))):
                try:
                    current_value = str(df.iloc[idx][last_col]).strip()
                    if current_value == first_row_value:
                        header_rows.append(idx)
                except:
                    pass
            
            if header_rows:
                logger.warning(f"发现 {len(header_rows)} 行最后一列与第一行相同")
                print(f"[进度] 发现 {len(header_rows)} 行疑似重复表头行...")
                df = df.drop(index=header_rows)
                df.reset_index(drop=True, inplace=True)
        else:
            print(f"[进度] 第一行最后一列值 '{first_row_value}' 是正常数据，保留")
    
    removed_rows = original_rows - len(df)
    if removed_rows > 0:
        logger.info(f"删除了 {removed_rows} 行重复表头行")
        print(f"[进度] 删除了 {removed_rows} 行重复表头行")
    else:
        logger.info("未发现重复表头行")
        print(f"[进度] 未发现重复表头行")
    
    return df


def force_ensure_label_column(df: pd.DataFrame, dataset_name: str) -> pd.DataFrame:
    """快速确保Label列存在，并将所有标签严格映射到配置中的标准类别标准标签。

    关键修复：先从 Label / Label.1 / Class / Attack 等候选列中选择“可映射配置中的标准类别最多、攻击样本最多”的标签列，
    再删除重复列。避免真实攻击标签落在 pandas 自动重命名列（如 Label.1）中时被提前删除，导致
    Web_SQL_Injection 这类小类从官方数量被误削减到很低。
    """
    df = df.copy()

    label_col = _choose_best_label_column(df, dataset_name)
    if label_col != "Label":
        if "Label" in df.columns:
            # 保留原Label列为审计辅助，避免与新Label冲突；后续重复/无关列会被清理。
            backup_name = "Label_original_before_selection"
            suffix = 1
            while backup_name in df.columns:
                suffix += 1
                backup_name = f"Label_original_before_selection_{suffix}"
            df.rename(columns={"Label": backup_name}, inplace=True)
        df.rename(columns={label_col: "Label"}, inplace=True)

    # 删除未被选中的其它标签候选列，避免旧 Label / Label.1 作为普通特征混入后续训练。
    extra_label_cols = [c for c in _label_candidate_columns(df) if c != "Label"]
    if extra_label_cols:
        df = df.drop(columns=extra_label_cols, errors="ignore")
        print(f"  删除未选中的标签候选列: {extra_label_cols}")

    initial_len = len(df)
    label_clean = clean_label_series_fast(df["Label"])
    label_lower = label_clean.str.lower().str.strip()

    # 只做便宜的向量化表头/空标签过滤；其他未知标签交给enforce_standard_labels统一审计删除。
    column_names = set(str(col).strip().lower() for col in df.columns)
    invalid_terms = {"", "nan", "none", "null", "label", "class", "type", "category", "attack", "attack_type"}
    invalid_terms |= {c for c in column_names if c not in {"label", "class"}}
    valid_mask = ~label_lower.isin(invalid_terms)

    n_invalid_header = int((~valid_mask).sum())
    if n_invalid_header > 0:
        print(f"  快速过滤 {n_invalid_header} 个无效标签/重复表头行（原{initial_len}行 -> {initial_len - n_invalid_header}行）")
        df = df.loc[valid_mask.values].copy()
        label_clean = label_clean.loc[valid_mask.values].reset_index(drop=True)
        df.reset_index(drop=True, inplace=True)

    df["Label"] = map_traffic_label_series(label_clean, source=dataset_name).values
    df = enforce_standard_labels(df, stage_name=f"{dataset_name}_force_ensure_label_column", label_col="Label", copy=False)
    export_label_count_audit(df, f"read_{dataset_name}_after_label_enforce", label_col="Label")

    print(f"  标签强约束后唯一值({df['Label'].nunique()}类): {sorted(df['Label'].unique().tolist())}")
    # 特别打印两个容易混淆/丢失的小类，方便立刻定位是否在读取阶段就异常。
    counts = df["Label"].value_counts()
    if counts.get("Web_SQL_Injection", 0) or counts.get("Heartbleed", 0):
        print(f"  小类检查: Web_SQL_Injection={int(counts.get('Web_SQL_Injection', 0))}, Heartbleed={int(counts.get('Heartbleed', 0))}")
    return df

def optimize_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """优化数据类型，减少内存占用"""
    # 调用者拥有读取工作表；不复制整表。数值越界时保留原64位类型。
    
    # 优化数值型列
    for col in df.select_dtypes(include=[np.float64]).columns:
        finite = df[col].replace([np.inf, -np.inf], np.nan)
        if finite.abs().max() <= np.finfo(np.float32).max:
            df[col] = df[col].astype(np.float32)
    
    for col in df.select_dtypes(include=[np.int64]).columns:
        min_val = df[col].min()
        max_val = df[col].max()
        if min_val >= 0:
            if max_val <= 255:
                df[col] = df[col].astype(np.uint8)
            elif max_val <= 65535:
                df[col] = df[col].astype(np.uint16)
            elif max_val <= 4294967295:
                df[col] = df[col].astype(np.uint32)
        else:
            if min_val >= -128 and max_val <= 127:
                df[col] = df[col].astype(np.int8)
            elif min_val >= -32768 and max_val <= 32767:
                df[col] = df[col].astype(np.int16)
            elif min_val >= np.iinfo(np.int32).min and max_val <= np.iinfo(np.int32).max:
                df[col] = df[col].astype(np.int32)
    
    # 读取阶段速度优化：object列全量nunique很慢，默认不在读取阶段做category探测。
    # 后续特征处理会显式完成数值转换/离散编码。
    if bool(getattr(config, "OPTIMIZE_OBJECT_CATEGORY_ON_READ", False)):
        sample_n = min(len(df), 100000)
        for col in df.select_dtypes(include=[object]).columns:
            if col == "Label":
                continue
            try:
                sample = df[col].sample(n=sample_n, random_state=getattr(config, "SEED", 42)) if len(df) > sample_n else df[col]
                if sample.nunique(dropna=False) / max(len(sample), 1) < 0.5:
                    df[col] = df[col].astype("category")
            except Exception:
                pass
    
    return df


def clean_malformed_rows(df: pd.DataFrame) -> pd.DataFrame:
    """清理异常格式的数据行
    
    处理以下异常情况：
    1. 数值列中包含多个点号连在一起的字符串（如 "0.00.00.00"）
    2. 数值列中包含过长的字符串
    3. 非数值的字符串尝试转换为数值失败的情况
    """
    if len(df) == 0:
        return df

    if getattr(get_active_config(), "SKIP_MALFORMED_ROW_SCAN", True):
        print("  [快速模式] 跳过逐行异常格式扫描；后续pd.to_numeric(errors='coerce')会自动处理异常数值")
        return df
    
    original_rows = len(df)
    numeric_cols = []
    
    for col in df.columns:
        if col in ["Label", "dataset_source"]:
            continue
        if df[col].dtype == 'object':
            numeric_cols.append(col)
    
    if not numeric_cols:
        return df
    
    def is_malformed_value(val):
        if pd.isna(val):
            return False
        val_str = str(val).strip()
        if '.' not in val_str and not val_str.replace('-', '').replace('+', '').isdigit():
            return False
        dot_count = val_str.count('.')
        if dot_count > 2:
            return True
        if dot_count >= 1:
            parts = val_str.split('.')
            for part in parts:
                if part and not part.isdigit():
                    if part not in ['e', 'E', 'nan', 'inf', '-inf']:
                        return True
        if len(val_str) > 50:
            return True
        return False
    
    malformed_mask = pd.Series([False] * len(df), index=df.index)
    
    for col in numeric_cols:
        try:
            col_is_malformed = df[col].apply(is_malformed_value)
            malformed_mask = malformed_mask | col_is_malformed
        except Exception:
            pass
    
    n_malformed = malformed_mask.sum()
    
    if n_malformed > 0:
        print(f"  [进度] 检测到 {n_malformed} 个异常格式行，将尝试修复...")
        for col in numeric_cols:
            try:
                df.loc[malformed_mask, col] = np.nan
            except Exception:
                pass
    
    cleaned_rows = len(df)
    removed_rows = original_rows - cleaned_rows
    
    if n_malformed > 0:
        print(f"  [完成] 清理完成，修复行数: {n_malformed}")
    
    return df


def normalize_feature_name(col_name: str) -> str:
    """兼容旧接口，内部调用 normalize_col"""
    return normalize_col(col_name)


def drop_duplicate_columns_fast(df: pd.DataFrame, stage_name: str = "") -> pd.DataFrame:
    """快速删除重复列名：没有重复列时不做整表切片，避免百万/千万行数据被无意义复制。"""
    if df is None or len(df.columns) == 0:
        return df
    if not df.columns.has_duplicates:
        if stage_name:
            print(f"  {stage_name}无重复列，跳过整表切片")
        return df
    orig_cols = len(df.columns)
    keep_mask = ~df.columns.duplicated()
    df = df.iloc[:, np.flatnonzero(keep_mask)].copy()
    removed = orig_cols - len(df.columns)
    print(f"  {stage_name}删除重复列: {removed} 列 ({orig_cols} -> {len(df.columns)})")
    return df


def _evenly_spaced_series_sample(series: pd.Series, sample_size: int) -> pd.Series:
    """从整列均匀抽取少量位置用于dtype判断，不执行整列dropna/astype。

    与head相比，均匀抽样还能覆盖多源融合后前半段全NaN、后半段才有值的C类字段。
    """
    n = len(series)
    if n == 0:
        return series.iloc[0:0]
    sample_size = max(1, min(int(sample_size), n))
    if n <= sample_size:
        return series
    idx = np.linspace(0, n - 1, num=sample_size, dtype=np.int64)
    return series.iloc[idx]


def convert_object_columns_to_numeric_fast(df: pd.DataFrame,
                                           exclude_cols: Optional[List[str]] = None,
                                           sample_size: Optional[int] = None) -> pd.DataFrame:
    """快速识别数值型object/category列；千万级大表默认延迟全列转换。

    性能优化说明：
    1. 只从每列均匀抽取少量样本做pd.to_numeric判断，不再扫描整列。
    2. 当数据量超过PRE_MAPPING_FULL_NUMERIC_CONVERSION_MAX_ROWS时，只记录候选数值列，
       不在映射阶段执行昂贵的整列pd.to_numeric。
    3. 真正需要数值的连续特征会在无泄漏Train/Val/Test切分后由
       apply_log_transform_train_fitted按需转换；离散特征则在训练集拟合编码器。
    4. 不改变标签、字段映射、Train/Val/Test切分、AE→IF→ASMOTE流程。
    """
    if df is None or len(df) == 0:
        return df
    cfg = get_active_config()
    if getattr(cfg,"DEFER_NUMERIC_TO_TRAIN",True):
        print("  [数值转换] 延迟到Train特征判定后的显式处理；映射不学习全量分布")
        return df
    if exclude_cols is None:
        exclude_cols = ["Label", "dataset_source"]
    if sample_size is None:
        sample_size = int(getattr(cfg, "FAST_NUMERIC_CONVERSION_SAMPLE", 10000))

    object_cols = [
        c for c in df.columns
        if c not in exclude_cols and (df[c].dtype == "object" or str(df[c].dtype) == "category")
    ]
    if not object_cols:
        print("  [快速数值转换] 无object/category候选列，无需处理")
        return df

    numeric_candidates = []
    text_candidates = []
    for i, col in enumerate(object_cols, 1):
        try:
            sample = _evenly_spaced_series_sample(df[col], sample_size).dropna()
            if sample.empty:
                text_candidates.append(col)
                continue
            # 直接对小样本to_numeric，不再先astype(str)，进一步减少临时对象。
            numeric_sample = pd.to_numeric(sample, errors="coerce")
            numeric_ratio = float(numeric_sample.notna().mean()) if len(numeric_sample) else 0.0
            if numeric_ratio >= 0.90:
                numeric_candidates.append(col)
            else:
                text_candidates.append(col)
        except Exception as exc:
            text_candidates.append(col)
            logger.warning(f"快速数值类型识别跳过列 {col}: {exc}")

    max_rows = int(getattr(cfg, "PRE_MAPPING_FULL_NUMERIC_CONVERSION_MAX_ROWS", 2_000_000))
    defer_large = bool(getattr(cfg, "DEFER_LARGE_OBJECT_NUMERIC_CONVERSION", True))
    if defer_large and len(df) > max_rows:
        print(
            f"  [快速数值转换] 大表延迟转换：rows={len(df):,} > {max_rows:,}; "
            f"识别数值候选 {len(numeric_candidates)} 列、文本/离散 {len(text_candidates)} 列。"
        )
        print("  [快速数值转换] 当前阶段不再扫描整列；数值转换延迟到Train/Val/Test切分后按需执行")
        if numeric_candidates:
            preview = ", ".join(map(str, numeric_candidates[:8]))
            suffix = " ..." if len(numeric_candidates) > 8 else ""
            print(f"    数值候选示例: {preview}{suffix}")
        return df

    converted = 0
    for i, col in enumerate(numeric_candidates, 1):
        try:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            converted += 1
            if i == 1 or i == len(numeric_candidates) or i % 10 == 0:
                print(f"  [快速数值转换] {i}/{len(numeric_candidates)} 列完成")
        except Exception as exc:
            logger.warning(f"快速数值转换跳过列 {col}: {exc}")
    print(f"  [快速数值转换] 转换 {converted} 列，文本/离散 {len(text_candidates)} 列")
    return df

def normalize_dataframe_columns(df, print_stats=False):
    """统一名称并安全合并重复表头列，冲突必须显式报错，不能盲删第一列之外的列。"""
    if df is None or not len(df):
        return df
    originals = set(map(str,df.columns))
    groups = {}
    for i,col in enumerate(df.columns):
        raw = str(col)
        base = re.sub(r"\.\d+$","",raw)
        if base in originals:
            raw = base
        # 关联旁表的UID和其他溯源字段保持原名，禁止被改成flow uid后当作特征删除。
        name = raw if raw in ({"Label", "dataset_source"} | PROVENANCE_COLUMNS) else normalize_feature_name(raw)
        groups.setdefault(name,[]).append(i)
    result = {}
    for name,positions in groups.items():
        value = df.iloc[:,positions[0]]
        for i in positions[1:]:
            other = df.iloc[:,i]
            overlap = value.notna() & other.notna()
            left = pd.to_numeric(value[overlap],errors="coerce")
            right = pd.to_numeric(other[overlap],errors="coerce")
            numeric = left.notna() & right.notna()
            equal = value[overlap].astype(str).eq(other[overlap].astype(str)).to_numpy(copy=True)
            if numeric.any():
                equal[numeric.to_numpy()] = np.isclose(left[numeric],right[numeric],rtol=1e-6,atol=1e-8)
            if not equal.all():
                raise ValueError(f"重复表头{name!r}存在冲突，禁止静默删列")
            value = value.fillna(other)
        result[name] = value
    if print_stats:
        print(f"  列名归一：{len(df.columns)} → {len(result)}；重复列经一致性检查后合并")
    return pd.DataFrame(result,index=df.index,copy=False)


# ========================== 3.5 四类特征映射系统 ==========================
@dataclass
class FeatureMappingResult:
    """特征映射结果数据类"""
    category_a: Dict[str, Tuple[str, str, str]]
    category_b: Dict[str, Tuple[str, str, str, str]]
    category_c_2017: List[str]
    category_c_2018: List[str]
    category_d: List[str]
    feature_mapping_confirmed: Dict[str, str]
    feature_mapping_candidate: Dict[str, Tuple[str, str]]
    feature_mapping_removed: List[str]


def classify_feature_mapping(col_2017, col_2018, df_2017_sample=None,
                            df_2018_sample=None, mapping_dict=None):
    """固定字段定义决定同义关系，不用全量样本均值推断单位。"""
    mapping_dict = mapping_dict or get_active_config().FEATURE_MAPPING
    left = canonical_feature_name(col_2017)
    right = canonical_feature_name(col_2018)
    if left != right:
        return "B", "字段定义不匹配，保留独立字段，等待人工核实", left
    explicit = normalize_col(col_2017) in mapping_dict and normalize_col(col_2018) in mapping_dict
    return ("A" if explicit else "B",
            "固定同义映射；单位沿用已声明字段定义" if explicit else "同名未知字段；单位未核实",
            left)


def run_four_category_mapping(df_2017, df_2018, output_dir):
    """A固定同义；B同名待核实；C单源字段；D标识/时间字段。"""
    cfg = get_active_config()
    groups = {}
    removed = set()
    confirmed = {}
    rows = []
    for source, frame in (("2017", df_2017), ("2018", df_2018)):
        by_canonical = {}
        for col in frame.columns:
            if col in {"Label", "dataset_source"}:
                continue
            if is_excluded_feature(col):
                removed.add(col)
                rows.append({"source": source, "raw": col, "canonical": "", "category": "D"})
                continue
            canon = canonical_feature_name(col)
            by_canonical.setdefault(canon, []).append(col)
            confirmed[col] = canon
            rows.append({"source": source, "raw": col, "canonical": canon,
                         "category": "pending"})
        groups[source] = by_canonical
    common = set(groups["2017"]) & set(groups["2018"])
    a, b, candidates = {}, {}, {}
    for canon in sorted(common):
        left, right = groups["2017"][canon][0], groups["2018"][canon][0]
        kind, reason, _ = classify_feature_mapping(left, right, mapping_dict=cfg.FEATURE_MAPPING)
        if kind == "A":
            a[canon] = (canon, reason, "both")
        else:
            b[canon] = (canon, reason, "canonical_name_only", left)
            candidates[left] = (right, reason)
    c17 = sorted(set(groups["2017"]) - common)
    c18 = sorted(set(groups["2018"]) - common)
    for row in rows:
        if row["category"] != "D":
            row["category"] = "A" if row["canonical"] in a else "B" if row["canonical"] in b else "C"
    os.makedirs(output_dir, exist_ok=True)
    atomic_csv(pd.DataFrame(rows), os.path.join(output_dir, "feature_match_debug.csv"))
    atomic_csv(pd.DataFrame([{"original": k, "canonical": v} for k, v in sorted(confirmed.items())]),
               os.path.join(output_dir, "feature_mapping_confirmed.csv"))
    atomic_csv(pd.DataFrame([{"original": k, "candidate": v[0], "reason": v[1]} for k, v in candidates.items()],
                            columns=["original", "candidate", "reason"]),
               os.path.join(output_dir, "feature_mapping_candidate.csv"))
    atomic_csv(pd.DataFrame({"feature_name": sorted(removed)}),
               os.path.join(output_dir, "feature_mapping_removed.csv"))
    write_json(os.path.join(output_dir, "mapping_policy.json"),
                   {"fit_free": True, "unit_policy": "declared_schema_not_distribution_mean",
                    "unit_multipliers": cfg.FEATURE_UNIT_MULTIPLIERS})
    return FeatureMappingResult(a, b, c17, c18, sorted(removed), confirmed, candidates, sorted(removed))


def merge_datasets_with_four_categories(df_2017, df_2018, mapping_result):
    """先标准化/合并同义列并排除标识字段，再纵向融合；不删除流量行。"""
    left = canonicalize_feature_frame(df_2017, "2017")
    right = canonicalize_feature_frame(df_2018, "2018")
    features = sorted((set(left.columns) | set(right.columns)) - {"Label", "dataset_source"})
    columns = ["dataset_source"] + features + ["Label"]
    memory_budget_guard(int(left.memory_usage(deep=True).sum()+right.memory_usage(deep=True).sum()),"两来源融合")
    result = pd.concat([left.reindex(columns=columns), right.reindex(columns=columns)],
                       ignore_index=True, copy=False)
    assert_feature_schema(result)
    return result


# -------------------------- 3. 数据读取函数 --------------------------
@timer_decorator
def read_multi_csv(folder_path: str, dataset_name: str) -> pd.DataFrame:
    """批量读取文件夹中的所有CSV文件（120G内存服务器全量读取优化版）。

    优化点：
    1. 取消按文件大小随机skiprows采样，所有CSV默认全量读取。
    2. 不再为了统计总行数提前扫一遍文件，减少重复I/O。
    3. 每个文件读完后做必要清洗，最后一次性pd.concat(copy=False)，避免循环concat的O(n^2)拷贝。
    4. 内存超过MAX_MEMORY_USAGE只报警和释放缓存，不再自动把数据砍半。
    """
    logger.info(f"开始读取{dataset_name}数据集：{folder_path}")
    print(f"开始读取{dataset_name}数据集：{folder_path}")

    csv_files = sorted([f for f in os.listdir(folder_path) if f.lower().endswith('.csv')])
    if not csv_files:
        raise ValueError(f"文件夹中未找到CSV文件：{folder_path}")

    print(f"找到 {len(csv_files)} 个CSV文件；服务器模式：全量读取，不随机采样")
    df_list = []
    total_rows = 0
    if not hasattr(config,"_input_manifest"):
        config._input_manifest = []
    encodings = getattr(config, 'READ_ENCODINGS', ['utf-8', 'latin1', 'gbk', 'utf-16'])


    def load_and_prepare_csv(file_name: str) -> Optional[pd.DataFrame]:
        file_path = os.path.join(folder_path, file_name)
        file_size = os.path.getsize(file_path) / (1024 * 1024)
        logger.info(f"读取文件：{file_name}，大小：{file_size:.2f} MB，全量读取")
        print(f"读取文件：{file_name}，大小：{file_size:.2f} MB，全量读取")

        df = None
        last_error = None
        for encoding in encodings:
            try:
                df = pd.read_csv(
                    file_path,
                    low_memory=False,
                    encoding=encoding,
                    memory_map=getattr(config, "READ_MEMORY_MAP", True),
                    on_bad_lines='error'
                )
                logger.info(f"使用编码 {encoding} 成功全量读取文件：{file_name}，行数={len(df):,}")
                print(f"使用编码 {encoding} 成功全量读取文件，行数={len(df):,}")
                break
            except Exception as e:
                last_error = e
                logger.warning(f"使用编码 {encoding} 读取文件失败：{str(e)}")
                continue

        if df is None:
            if getattr(config, "STRICT_INPUT_FILES", True):
                raise RuntimeError(f"CSV读取失败，禁止以不完整数据继续运行: {file_path}; {last_error}")
            logger.error(f"无法读取文件 {file_name}，所有编码都失败，最后错误：{last_error}")
            print(f"无法读取文件 {file_name}，所有编码都失败，跳过")
            return None

        # 先解析标签列，再删除 pandas 自动生成的 .1/.2 重复列。
        # 如果先删 Label.1，可能会误删真正包含攻击标签的重复标签列，导致 Web_SQL_Injection 等小类少算。
        df = force_ensure_label_column(df, f"{dataset_name}_{file_name}")

        dup_cols = [c for c in df.columns if re.search(r'\.\d+$', str(c)) and c != "Label"]
        if dup_cols:
            df = normalize_dataframe_columns(df)
            print(f"  pandas重复列已进行冲突检查并合并: {dup_cols}")

        df = remove_duplicate_header_rows(df, dataset_name)
        if getattr(config, "EXPORT_FUSED_FORECAST", False):
            df = forecast_attach_row_identity(df, dataset_name, file_name, config)
        df = normalize_dataframe_columns(df)
        forecast_require_row_identity(df, config, f"读取标准化后:{dataset_name}/{file_name}")
        df = optimize_dtypes(df)
        identity = os.stat(file_path)
        config._input_manifest.append({"source":str(dataset_name),"file":os.path.abspath(file_path),
                "encoding":encoding,"cleaned_rows":len(df),"bytes":identity.st_size,"mtime_ns":identity.st_mtime_ns})
        return df

    # 只保留少量在途文件，既能并行解析和清洗，也避免一次提交全部大文件。
    for file_name,df in iter_ordered_csv_frames(csv_files,folder_path,load_and_prepare_csv,config):
        if df is None:continue
        df_list.append(df);total_rows+=len(df)
        logger.info("成功读取：%s，累计行数：%s",file_name,f"{total_rows:,}")
        if get_memory_percent()/100.0>config.MAX_MEMORY_USAGE:release_memory()
        if get_memory_percent()/100.0>config.CRITICAL_MEMORY_USAGE:
            raise MemoryError("读取达到容器内存强保护阈值，未自动减少数据")

    if not df_list:
        raise ValueError(f"没有成功读取到{dataset_name}的任何有效数据")

    logger.info(f"开始一次性合并{dataset_name}数据集，共 {len(df_list)} 个文件，总行数约 {total_rows:,}")
    print(f"开始一次性合并{dataset_name}数据集，共 {len(df_list)} 个文件，总行数约 {total_rows:,}")

    for i, df in enumerate(df_list):
        df.columns = df.columns.str.strip().str.replace(r"\s+", " ", regex=True)
        df_list[i] = df

    memory_budget_guard(sum(int(f.memory_usage(deep=True).sum()) for f in df_list),"文件集合合并")
    df_combined = pd.concat(df_list, axis=0, ignore_index=True, copy=False, sort=False)
    df_list.clear()
    del df
    df_combined = optimize_dtypes(df_combined)
    log_memory_usage(f"{dataset_name}一次性concat后")

    logger.info(f"{dataset_name}数据集读取完成，总数据量：{len(df_combined):,}")
    print(f"{dataset_name}数据集读取完成，总数据量：{len(df_combined):,}")

    mem_percent = get_memory_percent() / 100.0
    if mem_percent > config.MAX_MEMORY_USAGE:
        logger.warning(
            f"合并后内存使用率较高：{mem_percent:.1%}。已保留全量数据，不再执行原代码的50%二次采样。"
        )
        release_memory()

    return df_combined


def export_mapping_summary_to_csv(mapping_result: FeatureMappingResult, 
                                   output_dir: str,
                                   cols_2017_original: int,
                                   cols_2018_original: int):
    """导出四类映射结果的汇总日志到CSV文件"""
    
    summary_data = {
        "统计项目": [
            "2017数据集原始字段数",
            "2018数据集原始字段数",
            "原始字段总数",
            "A类(严格同义/已对齐)字段数",
            "B类(弱同义/候选)字段数",
            "C类(2017独有)字段数",
            "C类(2018独有)字段数",
            "D类(明确异义/已删除)字段数",
            "最终融合后总字段数"
        ],
        "数量": [
            cols_2017_original,
            cols_2018_original,
            cols_2017_original + cols_2018_original,
            len(mapping_result.category_a),
            len(mapping_result.category_b),
            len(mapping_result.category_c_2017),
            len(mapping_result.category_c_2018),
            len(mapping_result.category_d),
            len(mapping_result.category_a) + len(mapping_result.category_b) + 
            len(mapping_result.category_c_2017) + len(mapping_result.category_c_2018)
        ]
    }
    
    os.makedirs(output_dir, exist_ok=True)
    
    summary_df = pd.DataFrame(summary_data)
    summary_path = os.path.join(output_dir, "mapping_summary.csv")
    summary_df.to_csv(summary_path, index=False, encoding='utf-8-sig')
    
    print("\n" + "=" * 60)
    print("【四类映射统计日志】")
    print("=" * 60)
    for _, row in summary_df.iterrows():
        print(f"  {row['统计项目']}: {row['数量']}")
    print("=" * 60)
    print(f"  汇总文件已保存: {summary_path}")
    
    return summary_df


# -------------------------- 4. 数据预处理函数 --------------------------

@timer_decorator
def classify_features(df, schema_only=False):
    """已知字段按定义分类；未知字段只允许在早期Train上做抽样判断。"""
    cfg = get_active_config()
    continuous, discrete, ports, audit = [], [], [], []
    known = set(cfg.FEATURE_MAPPING.values())
    overrides = getattr(cfg, "FEATURE_TYPE_OVERRIDES", {})
    for col in df.columns:
        if col in {"Label", "dataset_source"} or col in PROVENANCE_COLUMNS:
            continue
        if is_excluded_feature(col):
            raise ValueError(f"禁止将标识/时间字段用于分类: {col}")
        kind = overrides.get(col)
        reason = "manual_override"
        if kind is None:
            if col in {"src_port", "dst_port"}:
                kind, reason = "port", "fixed_port_schema"
            elif col == "protocol" or "flag" in col:
                kind, reason = "discrete", "fixed_category_schema"
            elif col in known:
                kind, reason = "continuous", "fixed_numeric_schema"
            elif schema_only:
                kind, reason = "continuous", "pending_train_inference"
            else:
                sample = _evenly_spaced_series_sample(df[col], cfg.FEATURE_CLASSIFY_SAMPLE_SIZE).dropna()
                numeric = pd.to_numeric(sample, errors="coerce")
                valid_ratio = float(numeric.notna().mean()) if len(sample) else 0.
                kind = "continuous" if valid_ratio >= .9 and numeric.nunique() > 10 else "discrete"
                reason = "train_sample_only"
        if kind not in {"continuous", "discrete", "port"}:
            raise ValueError(f"未知特征类型配置 {col}: {kind}")
        {"continuous": continuous, "discrete": discrete, "port": ports}[kind].append(col)
        audit.append({"feature": col, "type": kind, "reason": reason})
    if not schema_only:
        cfg._continuous_features, cfg._discrete_features, cfg._port_features = continuous, discrete, ports
        classification_state()["feature_types"] = {r["feature"]: r["type"] for r in audit}
        atomic_csv(pd.DataFrame(audit), os.path.join(cfg.OUTPUT_DIR, "feature_type_audit", "feature_types_train_only.csv"))
    return continuous, discrete, ports


def assert_train_class_coverage(source_df: pd.DataFrame,
                                        train_df: pd.DataFrame,
                                        stage_name: str = "early_split",
                                        label_col: str = "Label") -> None:
    """强制保证训练集覆盖全部配置中的标准类别标准标签。

    规则：
    1. 融合后的源数据必须先包含配置中的标准类别；若源数据本身缺类，直接报错，禁止伪造/复制样本补类。
    2. Train必须包含配置中的标准类别；若切分后缺类，直接报错。
    3. Val/Test不做类别补齐要求，保持真实自然分布。
    """
    cfg = get_active_config()
    required_base = set(getattr(cfg, "REQUIRED_BASE_LABELS", cfg.STANDARD_LABELS))

    source_present = set(map_traffic_label_series(source_df[label_col]).dropna().astype(str)) if label_col in source_df.columns else set()
    source_missing = sorted(required_base - source_present)
    required = required_base | source_present
    if source_missing:
        raise ValueError(
            f"[{stage_name}] 融合后的原始数据缺少以下标准类别: {source_missing}。"
            "无法在不复制/伪造样本的前提下保证Train 配置中的标准类别全覆盖，请先检查原始数据读取与标签映射。"
        )

    train_present = set(map_traffic_label_series(train_df[label_col]).dropna().astype(str)) if label_col in train_df.columns else set()
    train_missing = sorted(required - train_present)
    if train_missing:
        raise ValueError(
            f"[{stage_name}] Train切分后缺少以下标准类别: {train_missing}。"
            "训练集必须覆盖全部基础类别及原始数据已出现的新增类别，已停止执行。"
        )


def _export_split_label_audit(source_df: pd.DataFrame,
                              train_df: pd.DataFrame,
                              val_df: pd.DataFrame,
                              test_df: pd.DataFrame,
                              stage_name: str = "early_split",
                              label_col: str = "Label") -> None:
    """导出Train/Val/Test各类别真实样本数量；仅审计，不做任何补样。"""
    try:
        cfg = get_active_config()
        source_counts = map_traffic_label_series(source_df[label_col]).value_counts()
        train_counts = map_traffic_label_series(train_df[label_col]).value_counts()
        val_counts = map_traffic_label_series(val_df[label_col]).value_counts() if len(val_df) else pd.Series(dtype=int)
        test_counts = map_traffic_label_series(test_df[label_col]).value_counts() if len(test_df) else pd.Series(dtype=int)
        rows = []
        for label in cfg.STANDARD_LABELS:
            total = int(source_counts.get(label, 0))
            tr = int(train_counts.get(label, 0))
            va = int(val_counts.get(label, 0))
            te = int(test_counts.get(label, 0))
            rows.append({
                "Label": label,
                "Original_Count": total,
                "Train_Count": tr,
                "Val_Count": va,
                "Test_Count": te,
                "Train_Ratio": tr / max(total, 1),
                "Val_Ratio": va / max(total, 1),
                "Test_Ratio": te / max(total, 1),
                "Train_Covered": tr > 0,
            })
        audit_dir = os.path.join(cfg.OUTPUT_DIR, "split_audit")
        os.makedirs(audit_dir, exist_ok=True)
        safe_stage = re.sub(r"[^A-Za-z0-9_\-]+", "_", str(stage_name))[:120]
        pd.DataFrame(rows).to_csv(
            os.path.join(audit_dir, f"split_label_counts_{safe_stage}.csv"),
            index=False, encoding="utf-8-sig"
        )
    except Exception as exc:
        logger.warning(f"[{stage_name}] 数据切分类别审计导出失败: {exc}")


def split_train_val_test_no_leakage(df, train_size=.8, val_size=.1,
                                    test_size=.1, random_state=42):
    """保留真实重复行，按内容指纹分组；8:1:1为目标，禁止同组跨集合。"""
    cfg = get_active_config()
    if getattr(cfg, "FORECAST_UNIFIED_SPLIT", False):
        return forecast_shared_split(df, train_size, val_size, test_size, cfg)
    if min(train_size,val_size,test_size) <= 0 or abs(train_size+val_size+test_size-1)>1e-6:
        raise ValueError("三个切分比例必须为正数且总和为1")
    valid = enforce_standard_labels(df, stage_name="before_early_split", copy=False)
    assert_train_class_coverage(valid,valid,"source_coverage")
    labels = valid.Label.astype(str).to_numpy()
    fingerprints = classification_row_fingerprints(valid)
    unique, first, inverse, weights = np.unique(fingerprints, return_index=True, return_inverse=True, return_counts=True)
    rng = np.random.default_rng(random_state)
    assignments = np.zeros(len(valid),dtype=np.int8)
    if cfg.DUPLICATE_SPLIT_MODE == "group":
        groups_split = np.zeros(len(unique),dtype=np.int8)
        group_labels = labels[first]
        for label in cfg.STANDARD_LABELS:
            groups = rng.permutation(np.flatnonzero(group_labels == label))
            if not len(groups):
                continue
            if len(groups) <= 2:
                groups_split[groups] = 0
                continue
            cumulative = np.cumsum(weights[groups])
            total = int(cumulative[-1])
            n_train = int(np.argmin(np.abs(cumulative-train_size*total)))+1
            n_train = min(max(n_train,1),len(groups)-2)
            n_val_end = int(np.argmin(np.abs(cumulative-(train_size+val_size)*total)))+1
            n_val_end = min(max(n_val_end,n_train+1),len(groups)-1)
            groups_split[groups[n_train:n_val_end]] = 1
            groups_split[groups[n_val_end:]] = 2
        assignments = groups_split[inverse]
        # 同特征却不同标签的组仍保持整体；确保每个已出现类别都有真实Train。
        for label in cfg.STANDARD_LABELS:
            pos = np.flatnonzero(labels == label)
            if len(pos) and not np.any(assignments[pos] == 0):
                groups = np.unique(inverse[pos])
                chosen = groups[np.argmin(weights[groups])]
                groups_split[chosen] = 0
        assignments = groups_split[inverse]
    elif cfg.DUPLICATE_SPLIT_MODE == "audit":
        for label in cfg.STANDARD_LABELS:
            idx = rng.permutation(np.flatnonzero(labels == label))
            if len(idx) <= 2:
                continue
            raw = np.array([train_size,val_size,test_size])*len(idx)
            alloc = np.floor(raw).astype(int)
            for j in np.argsort(-(raw-alloc),kind="stable")[:len(idx)-int(alloc.sum())]:
                alloc[j] += 1
            assignments[idx[alloc[0]:alloc[0]+alloc[1]]] = 1
            assignments[idx[alloc[0]+alloc[1]:]] = 2
    else:
        raise ValueError("DUPLICATE_SPLIT_MODE仅支持group或audit")
    indices = [rng.permutation(np.flatnonzero(assignments == j)) for j in range(3)]
    if any(len(x)==0 for x in indices):
        raise ValueError("重复分组后某分区为空；不能拆开重复组凑比例。请检查数据或由用户选择audit模式")
    membership = np.bincount(np.concatenate(indices),minlength=len(valid))
    if np.any(membership != 1):
        raise RuntimeError("分组切分未满足行号守恒")
    hashes = [np.unique(fingerprints[x]) for x in indices]
    overlaps = {name:int(len(np.intersect1d(hashes[a],hashes[b],assume_unique=True)))
                for name,a,b in [("train_val",0,1),("train_test",0,2),("val_test",1,2)]}
    if cfg.DUPLICATE_SPLIT_MODE == "group" and any(overlaps.values()):
        raise RuntimeError("重复流指纹跨分区，已停止")
    # 同一特征指纹对应多个标签时只审计、不删除、不改标签、不拆组。
    mixed = pd.DataFrame({"fingerprint":fingerprints,"Label":labels}).groupby("fingerprint").Label.nunique()
    report = {"policy":cfg.DUPLICATE_SPLIT_MODE,"original_rows":len(valid),"groups":len(unique),
              "repeated_rows":int(len(valid)-len(unique)),"mixed_label_groups":int((mixed>1).sum()),
              "cross_split_fingerprint_overlap":overlaps,"rows":[len(x) for x in indices],
              "actual_ratios":[len(x)/len(valid) for x in indices],
              "hash_policy":"64-bit conservative grouping; collision merges groups rather than leaking them"}
    manifest=sorted(getattr(cfg,"_input_manifest",[]),key=lambda r:(r["source"],r["file"]))
    if manifest and sum(r["cleaned_rows"] for r in manifest)==len(valid):
        captures=[]
        start=0
        for record in manifest:
            end=start+record["cleaned_rows"]
            counts=np.bincount(assignments[start:end],minlength=3).tolist()
            captures.append({"source":record["source"],"file":record["file"],"train_val_test":counts})
            start=end
        report["capture_file_audit"]=captures
        report["capture_files_crossing_splits"]=sum(sum(n>0 for n in r["train_val_test"])>1 for r in captures)
    report["claim_scope"]="content duplicate separation only; not capture/day/session holdout"
    write_json(os.path.join(cfg.OUTPUT_DIR,"split_audit","duplicate_group_audit.json"),report)
    classification_state()["split_policy"] = report
    result = tuple(valid.iloc[x].reset_index(drop=True) for x in indices)
    assert_train_class_coverage(valid,result[0],"after_group_split")
    _export_split_label_audit(valid,*result,stage_name="early_group_split")
    return result


def apply_log_transform_train_fitted(train_df, val_df, test_df, continuous_features):
    """填补完成后沿用训练集偏移log1p；记录测试值超出训练定义域的截断。"""
    shifts, clipped = {}, {}
    for col in continuous_features:
        if col not in train_df:
            continue
        minimum = float(train_df[col].min())
        shift = -minimum + 1. if minimum < 0 else 0.
        shifts[col] = shift
        for split, frame in (("train", train_df), ("val", val_df), ("test", test_df)):
            values = pd.to_numeric(frame[col], errors="coerce").to_numpy(dtype=np.float64, copy=True)
            if not np.isfinite(values).all():
                raise ValueError(f"{split}/{col}尚有缺失/无穷值；必须先执行训练集拟合填补")
            values += shift
            count = int((values < 0).sum())
            if count:
                clipped[f"{split}/{col}"] = count
            np.maximum(values, 0, out=values)
            np.log1p(values, out=values)
            frame[col] = values.astype(np.float32)
    classification_state().update(log_shifts=shifts, log_domain_clipping=clipped)
    if clipped:
        logger.warning("log1p定义域截断已审计: %s", clipped)
    return train_df, val_df, test_df, shifts

def fill_missing_train_fitted(train_df, val_df, test_df):
    """先保留缺失语义，再使用Train的中位数/众数填补，不提前填零。"""
    cfg = get_active_config()
    numeric_cols = set(getattr(cfg, "_continuous_features", [])) | set(getattr(cfg, "_port_features", []))
    numeric_cols |= set(train_df.select_dtypes(include=[np.number]).columns)
    port_cols = set(getattr(cfg, "_port_features", []))
    numeric_cols -= {"Label", "dataset_source"} | PROVENANCE_COLUMNS | port_cols | set(getattr(cfg,"_discrete_features",[]))
    numeric_fill, categorical_fill = {}, {}
    for col in sorted(numeric_cols):
        if col not in train_df:
            continue
        for frame in (train_df, val_df, test_df):
            if col in frame:
                frame[col] = pd.to_numeric(frame[col], errors="coerce").replace([np.inf, -np.inf], np.nan)
        median = train_df[col].median()
        numeric_fill[col] = float(median) if pd.notna(median) else 0.
    for col in train_df.columns:
        if col in numeric_cols or col in port_cols or col in {"Label", "dataset_source"} or col in PROVENANCE_COLUMNS:
            continue
        for frame in (train_df,val_df,test_df):
            if col=="protocol" or "flag" in col:
                value = pd.to_numeric(frame[col],errors="coerce").replace([np.inf,-np.inf],np.nan)
                invalid = (value<0) | (value%1 != 0)
                if col=="protocol":
                    invalid |= value>255
                frame[col] = value.mask(invalid,np.nan)
            elif pd.api.types.is_numeric_dtype(frame[col]):
                frame[col] = frame[col].replace([np.inf,-np.inf],np.nan)
        mode = train_df[col].dropna().mode()
        categorical_fill[col] = mode.iloc[0] if len(mode) else "Unknown"
    for frame in (train_df, val_df, test_df):
        for col, value in {**numeric_fill, **categorical_fill}.items():
            if col in frame:
                frame[col] = frame[col].fillna(value)
    classification_state().update(numeric_fill=numeric_fill, categorical_fill=categorical_fill)
    return train_df, val_df, test_df

def robust_scale_train_fitted(train_df: pd.DataFrame,
                              val_df: pd.DataFrame,
                              test_df: pd.DataFrame,
                              continuous_features: List[str]):
    """RobustScaler只fit训练集，val/test只transform（服务器低峰值内存版）。

    优化原则：
    1. 不再复制Train/Val/Test三张完整DataFrame；直接更新当前工作表。
    2. 若训练集超过ROBUST_SCALER_FIT_MAX_SAMPLES，只从训练集抽样拟合中位数/IQR；
       不使用Val/Test，因此不会引入数据泄露。
    3. 使用float32矩阵降低峰值内存。
    4. Train/Val/Test按块transform，避免一次性创建超大二维临时数组。
    """
    cfg = get_active_config()
    features = [c for c in continuous_features if c in train_df.columns]
    if not features:
        return train_df, val_df, test_df, RobustScaler()

    print(
        f"  [RobustScaler] train={len(train_df):,}, val={len(val_df):,}, "
        f"test={len(test_df):,}, features={len(features)}"
    )

    # 3.3/3.4已经完成主要数值化和缺失处理；这里只对异常dtype做兜底，避免重复全列to_numeric。
    for frame_name, frame in (("train", train_df), ("val", val_df), ("test", test_df)):
        non_numeric = [c for c in features if c in frame.columns and not pd.api.types.is_numeric_dtype(frame[c])]
        if non_numeric:
            print(f"  [RobustScaler] {frame_name}仍有{len(non_numeric)}个非数值连续特征，执行兜底转换")
            for col in non_numeric:
                frame[col] = pd.to_numeric(frame[col], errors='coerce')

    fit_max = int(getattr(cfg, "ROBUST_SCALER_FIT_MAX_SAMPLES", 2_000_000))
    fit_n = min(len(train_df), max(fit_max, 1))
    if len(train_df) > fit_n:
        rng = np.random.default_rng(int(getattr(cfg, "SEED", 42)))
        fit_idx = np.sort(rng.choice(len(train_df), size=fit_n, replace=False))
        fit_block = train_df.iloc[fit_idx][features]
        print(f"  [RobustScaler] 训练集过大，仅用训练集子样本fit: {fit_n:,}/{len(train_df):,}")
    else:
        fit_block = train_df[features]
        print(f"  [RobustScaler] 使用完整训练集fit: {fit_n:,}")

    X_fit = fit_block.to_numpy(dtype=np.float32, copy=True)
    del fit_block
    np.nan_to_num(X_fit, copy=False, nan=0.0, posinf=0.0, neginf=0.0)

    # log1p之后理论上不会出现>1e10的常规值；保留训练集来源的极端值截断兜底。
    max_abs = np.max(np.abs(X_fit), axis=0) if X_fit.size else np.zeros(len(features), dtype=np.float32)
    extreme_mask = (~np.isfinite(max_abs)) | (max_abs > 1e10)
    clip_low = np.full(len(features), -np.inf, dtype=np.float32)
    clip_high = np.full(len(features), np.inf, dtype=np.float32)
    if extreme_mask.any():
        extreme_idx = np.flatnonzero(extreme_mask)
        q = np.percentile(X_fit[:, extreme_idx], [1, 99], axis=0)
        clip_low[extreme_idx] = q[0].astype(np.float32, copy=False)
        clip_high[extreme_idx] = q[1].astype(np.float32, copy=False)
        X_fit[:, extreme_idx] = np.clip(
            X_fit[:, extreme_idx], clip_low[extreme_idx], clip_high[extreme_idx]
        )
        print(f"  [RobustScaler] 对{len(extreme_idx)}个极端特征按训练集1%-99%分位截断")

    scaler = RobustScaler(copy=False)
    print("  [RobustScaler] 开始fit训练集统计量...")
    scaler.fit(X_fit)
    classification_state()["robust_clip"] = {
        features[i]: [float(clip_low[i]), float(clip_high[i])] for i in np.flatnonzero(extreme_mask)}
    del X_fit
    release_memory()
    print("  [RobustScaler] fit完成，开始分块transform...")

    chunk_size = int(getattr(cfg, "ROBUST_SCALER_TRANSFORM_CHUNK_SIZE", 250_000))
    chunk_size = max(chunk_size, 10_000)
    feature_positions_cache = {}

    def _transform_frame_in_chunks(frame: pd.DataFrame, frame_name: str) -> None:
        if frame is None or len(frame) == 0:
            return
        positions = feature_positions_cache.get(tuple(frame.columns))
        if positions is None:
            positions = [frame.columns.get_loc(c) for c in features]
            feature_positions_cache[tuple(frame.columns)] = positions
        n = len(frame)
        total_chunks = int(np.ceil(n / chunk_size))
        for chunk_no, start_row in enumerate(range(0, n, chunk_size), 1):
            end_row = min(start_row + chunk_size, n)
            X_chunk = frame.iloc[start_row:end_row, positions].to_numpy(dtype=np.float32, copy=True)
            np.nan_to_num(X_chunk, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
            if extreme_mask.any():
                extreme_idx = np.flatnonzero(extreme_mask)
                X_chunk[:, extreme_idx] = np.clip(
                    X_chunk[:, extreme_idx], clip_low[extreme_idx], clip_high[extreme_idx]
                )
            X_chunk = scaler.transform(X_chunk).astype(np.float32, copy=False)
            frame.iloc[start_row:end_row, positions] = X_chunk
            if chunk_no == 1 or chunk_no == total_chunks or chunk_no % 5 == 0:
                print(
                    f"    [{frame_name}] chunk {chunk_no}/{total_chunks} "
                    f"({end_row:,}/{n:,})"
                )
            del X_chunk

    _transform_frame_in_chunks(train_df, "train")
    _transform_frame_in_chunks(val_df, "val")
    _transform_frame_in_chunks(test_df, "test")
    release_memory()
    print("  [RobustScaler] Train/Val/Test分块变换完成")
    return train_df, val_df, test_df, scaler

def encode_discrete_train_fitted(train_df, val_df, test_df, discrete_features):
    """只在Train建立编码表；数值类别以统一字符串键消除80/80.0差异。"""
    mappings = {}
    for col in discrete_features:
        values = category_keys(train_df[col])
        mapping = {v: i for i, v in enumerate(sorted(values.unique().tolist()))}
        mappings[col] = mapping
        for frame in (train_df, val_df, test_df):
            frame[col] = category_keys(frame[col]).map(mapping).fillna(-1).astype(np.int32)
    classification_state()["discrete_mappings"] = mappings
    return train_df, val_df, test_df, mappings


def port_features_train_fitted(train_df, val_df, test_df, port_features):
    """统一合法整数端口；缺失/越界为-1而不是虚构服务；频率只由Train拟合。"""
    frequency = {}
    invalid = {}
    for col in port_features:
        for split, frame in (("train", train_df), ("val", val_df), ("test", test_df)):
            value = pd.to_numeric(frame[col], errors="coerce").astype(np.float64)
            bad = value.isna() | ~np.isfinite(value) | (value < 0) | (value > 65535) | (value % 1 != 0)
            invalid[f"{split}/{col}"] = int(bad.sum())
            frame[col] = value.mask(bad, -1).astype(np.int32)
        frequency[col] = {str(int(k)): float(v) for k, v in train_df[col].value_counts(normalize=True).items()}
    pair_frequency = {}
    if "src_port" in port_features and "dst_port" in port_features:
        pairs = train_df.src_port.astype(str) + "_" + train_df.dst_port.astype(str)
        pair_frequency = pairs.value_counts(normalize=True).to_dict()
    classification_state().update(port_frequency=frequency, port_pair_frequency=pair_frequency,
                                  invalid_ports=invalid, port_features=list(port_features))
    for frame in (train_df, val_df, test_df):
        add_port_derived_features(frame, frequency, pair_frequency)
    return train_df, val_df, test_df


def encode_scaled_autoencoder_batches(values, model: nn.Module, batch_size: int = 4096) -> np.ndarray:
    """已缩放数据分批编码；复用z做解码，峰值显存只与batch_size有关。"""
    if batch_size < 1:
        raise ValueError("AE推理batch_size必须为正数")
    latent_dim = int(model.encoder[-1].out_features)
    output = np.empty((len(values), latent_dim + 1), dtype=np.float32)
    device = next(model.parameters()).device
    model.eval()
    with torch.no_grad():
        for start in range(0, len(values), batch_size):
            part = values[start:start + batch_size]
            if isinstance(part, np.ndarray):
                part = torch.from_numpy(np.ascontiguousarray(part, dtype=np.float32))
            batch = part.to(device)
            z = model.encode(batch)
            reconstructed = model.decoder(z)
            error = torch.mean((batch - reconstructed) ** 2, dim=1, keepdim=True)
            output[start:start + len(batch)] = torch.cat([z, error], dim=1).cpu().numpy()
    return output


def transform_with_fitted_autoencoder(X: np.ndarray, model: nn.Module, scaler: StandardScaler,
                                      batch_size: int = 4096) -> np.ndarray:
    """使用训练集拟合的AE和AE内部StandardScaler变换val/test。"""
    if X is None or len(X) == 0:
        latent_dim = int(model.encoder[-1].out_features)
        return np.empty((0, latent_dim + 1), dtype=np.float32)
    if batch_size < 1:
        raise ValueError("AE推理batch_size必须为正数")
    latent_dim = int(model.encoder[-1].out_features)
    output = np.empty((len(X), latent_dim + 1), dtype=np.float32)
    # 清洗和scaler.transform也按块进行，不再为val/test各复制完整大矩阵。
    for start in range(0, len(X), batch_size):
        clean = np.nan_to_num(X[start:start + batch_size].astype(np.float32, copy=False),
                              nan=0, posinf=0, neginf=0)
        scaled = scaler.transform(clean).astype(np.float32, copy=False)
        output[start:start + len(clean)] = encode_scaled_autoencoder_batches(scaled, model, batch_size)
    return output


def _make_ae_batch_loader(tensor: torch.Tensor, batch_size: int, shuffle: bool = False,
                          generator: Optional[torch.Generator] = None,
                          pin_memory: bool = False, row_indices=None):
    """整批索引CPU张量，避免逐样本__getitem__和重复自重构目标的collate。

    沿用DataLoader原有的RandomSampler/SequentialSampler和BatchSampler，
    保留随机种子消耗、每轮打乱顺序及最后一个不足批次。这里有意把批索引
    传给sampler而不是batch_sampler，并用batch_size=None关闭逐样本自动拼接；
    DataLoader每次直接执行tensor[一批索引]，返回一个二维输入张量。
    不把全量输入常驻显存，CUDA时仍只对当前批次pin_memory/非阻塞传输。
    """
    if batch_size < 1:
        raise ValueError("AE训练batch_size必须为正数")
    if row_indices is not None:
        tensor = IndexedBatchTensor(tensor,row_indices)
    sampler = RandomSampler(tensor, generator=generator) if shuffle else SequentialSampler(tensor)
    batches = BatchSampler(sampler, batch_size=batch_size, drop_last=False)
    return DataLoader(tensor, batch_size=None, sampler=batches,
                      generator=generator, pin_memory=pin_memory)


class AutoEncoder(nn.Module):
    """增强版全连接AutoEncoder。

    与原始 input→64→32→latent 结构相比，这里根据输入维度动态扩展隐藏层容量，
    并默认使用LayerNorm，避免小批次或最后一个batch过小时BatchNorm不稳定。
    """
    def __init__(self, input_dim: int, latent_dim: int = 16, dropout: float = 0.05, use_layer_norm: bool = True):
        super(AutoEncoder, self).__init__()

        hidden1 = int(min(512, max(64, input_dim * 4)))
        hidden2 = int(min(256, max(32, input_dim * 2)))
        hidden3 = int(min(128, max(latent_dim * 2, input_dim)))

        def block(in_dim: int, out_dim: int):
            layers = [nn.Linear(in_dim, out_dim), nn.GELU()]
            if use_layer_norm:
                layers.append(nn.LayerNorm(out_dim))
            else:
                layers.append(nn.BatchNorm1d(out_dim))
            if dropout and dropout > 0:
                layers.append(nn.Dropout(dropout))
            return layers

        self.encoder = nn.Sequential(
            *block(input_dim, hidden1),
            *block(hidden1, hidden2),
            *block(hidden2, hidden3),
            nn.Linear(hidden3, latent_dim)
        )

        self.decoder = nn.Sequential(
            *block(latent_dim, hidden3),
            *block(hidden3, hidden2),
            *block(hidden2, hidden1),
            nn.Linear(hidden1, input_dim)
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

    def encode(self, x):
        """仅返回编码后的特征"""
        return self.encoder(x)

@timer_decorator
def apply_autoencoder(X: np.ndarray, latent_dim: int = 16, epochs: int = 100,
                     batch_size: int = 512, learning_rate: Optional[float] = None,
                     labels: Optional[np.ndarray] = None, normal_oversample_ratio: float = 1.0,
                     return_artifacts: bool = False):
    """应用AutoEncoder进行特征编码 + 重构误差（服务器增强版）。

    核心改动：
    1. epochs作为训练上限，默认由Config.AE_EPOCHS控制，建议80~120。
    2. 新增验证集、Early Stopping、ReduceLROnPlateau和best model回滚。
    3. 导出训练曲线CSV/PNG/PDF/SVG，便于判断loss是否仍在下降。
    4. 使用更大容量的AutoEncoder结构，降低欠拟合导致的loss平台期。
    """
    cfg = get_active_config()
    if learning_rate is None:
        learning_rate = float(getattr(cfg, "AE_LEARNING_RATE", 0.001))

    patience = int(getattr(cfg, "AE_EARLY_STOPPING_PATIENCE", 12))
    min_delta = float(getattr(cfg, "AE_MIN_DELTA", 1e-4))
    val_split = float(getattr(cfg, "AE_VAL_SPLIT", 0.1))
    weight_decay = float(getattr(cfg, "AE_WEIGHT_DECAY", 1e-5))
    grad_clip = float(getattr(cfg, "AE_GRAD_CLIP_NORM", 5.0))
    dropout = float(getattr(cfg, "AE_DROPOUT", 0.05))
    use_layer_norm = bool(getattr(cfg, "AE_USE_LAYER_NORM", True))

    logger.info(
        f"开始AutoEncoder特征编码，latent_dim={latent_dim}, epochs上限={epochs}, "
        f"batch_size={batch_size}, lr={learning_rate}, val_split={val_split}, patience={patience}"
    )
    print(f"[AutoEncoder] 开始特征编码（增强训练版）")
    print(f"  输入形状: {X.shape}")
    print(f"  潜在维度: {latent_dim}")
    print(f"  训练轮数上限: {epochs}（Early Stopping可能提前停止）")
    print(f"  批次大小: {batch_size}")
    print(f"  学习率: {learning_rate}")

    n_samples, n_features = X.shape
    used_indices = np.arange(n_samples)

    if labels is not None:
        print(f"  检测到标签，开始Normal样本过采样...")
        unique_labels, counts = np.unique(labels, return_counts=True)
        label_counts = dict(zip(unique_labels, counts))
        print(f"    原始标签分布: {label_counts}")

        normal_mask = (labels == 'Normal') | (labels == 'BENIGN') | (labels == 'Benign')
        attack_mask = ~normal_mask
        n_normal = int(normal_mask.sum())
        n_attack = int(attack_mask.sum())
        print(f"    Normal样本数: {n_normal}")
        print(f"    Attack样本数: {n_attack}")

        if n_normal > n_attack and normal_oversample_ratio > 0 and n_attack > 0:
            rng = np.random.default_rng(getattr(cfg, "SEED", 42))
            target_normal = min(int(n_attack * normal_oversample_ratio), n_normal)
            if target_normal > n_attack:
                oversample_factor = target_normal / n_attack
                attack_indices = np.where(attack_mask)[0]
                attack_samples_needed = int(oversample_factor * n_attack) - n_attack
                if attack_samples_needed > 0:
                    additional_indices = rng.choice(attack_indices, attack_samples_needed, replace=True)
                    all_indices = np.concatenate([np.where(normal_mask)[0], np.where(attack_mask)[0], additional_indices])
                else:
                    all_indices = np.concatenate([np.where(normal_mask)[0], np.where(attack_mask)[0]])
                print(f"    Attack样本过采样: {n_attack} -> {len(np.where(all_indices < n_samples)[0])}")
            else:
                normal_indices = np.where(normal_mask)[0]
                sampled_normal_indices = rng.choice(normal_indices, target_normal, replace=False)
                all_indices = np.concatenate([sampled_normal_indices, np.where(attack_mask)[0]])

            X = X[all_indices]
            n_samples = len(X)
            used_indices = all_indices
            print(f"    过采样后数据量: {n_samples}")
            print(f"    过采样后标签分布: {dict(zip(*np.unique(labels[all_indices], return_counts=True)))}")

    # 数据标准化。这里的MSE约等于标准化空间下的平均重构误差，0.05通常不算高。
    scaler = StandardScaler() if getattr(cfg, "AE_INPUT_SCALE_MODE", "robust") == "standard" else FunctionTransformer(validate=False)
    raw_input = X.astype(np.float32,copy=False)
    clean_input = np.nan_to_num(raw_input,copy=not raw_input.flags.writeable,nan=0,posinf=0,neginf=0)
    X_scaled = scaler.fit_transform(clean_input)
    X_tensor = torch.FloatTensor(X_scaled)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"  使用设备: {device}")

    # 训练/验证划分：只用于选择best AE，不影响最终编码样本数量。
    rng = np.random.default_rng(getattr(cfg, "SEED", 42))
    all_idx = np.arange(n_samples)
    if n_samples >= 100 and 0 < val_split < 0.4:
        val_size = max(1, int(n_samples * val_split))
        val_idx = rng.choice(all_idx, size=val_size, replace=False)
        train_mask = np.ones(n_samples, dtype=bool)
        train_mask[val_idx] = False
        train_idx = all_idx[train_mask]
    else:
        train_idx = all_idx
        val_idx = np.array([], dtype=int)

    train_tensor = X_tensor  # 只保存行号视图，不再复制90%的完整特征矩阵。
    train_loader = _make_ae_batch_loader(train_tensor, batch_size, shuffle=True,
                                         pin_memory=device.type == "cuda",row_indices=train_idx)
    val_loader = None
    if len(val_idx) > 0:
        val_tensor = X_tensor
        val_loader = _make_ae_batch_loader(val_tensor, batch_size, shuffle=False,
                                           pin_memory=device.type == "cuda",row_indices=val_idx)
        print(f"  AE验证集: {len(val_idx):,} 条；训练集: {len(train_idx):,} 条")
    else:
        print("  数据量较小或val_split关闭，AE仅使用训练loss监控")

    model = AutoEncoder(
        input_dim=n_features,
        latent_dim=latent_dim,
        dropout=dropout,
        use_layer_norm=use_layer_norm
    ).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=float(getattr(cfg, "AE_LR_SCHEDULER_FACTOR", 0.5)),
        patience=int(getattr(cfg, "AE_LR_SCHEDULER_PATIENCE", 5))
    )

    def _eval_loss(loader):
        if loader is None:
            return None
        model.eval()
        # 原实现用Python float累计；设备端float64保留相同的样本加权口径。
        total = torch.zeros((), dtype=torch.float64, device=device)
        n_obs = 0
        with torch.no_grad():
            for batch_x in loader:
                batch_x = batch_x.to(device, non_blocking=device.type == "cuda")
                output = model(batch_x)
                loss = criterion(output, batch_x)
                total.add_(loss.detach().to(dtype=torch.float64), alpha=len(batch_x))
                n_obs += len(batch_x)
        # 每轮验证只读一次标量，不再逐批触发CUDA -> CPU同步。
        return total.item() / max(n_obs, 1)

    print("  开始训练AE...")
    best_metric = float('inf')
    best_state = None
    best_epoch = 0
    no_improve = 0
    history = []

    for epoch in range(epochs):
        model.train()
        total_loss = torch.zeros((), dtype=torch.float64, device=device)
        n_seen = 0
        for batch_x in train_loader:
            batch_x = batch_x.to(device, non_blocking=device.type == "cuda")
            output = model(batch_x)
            loss = criterion(output, batch_x)

            optimizer.zero_grad()
            loss.backward()
            if grad_clip and grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

            # detach避免保留计算图；累计器常量大小，不随样本量增长。
            total_loss.add_(loss.detach().to(dtype=torch.float64), alpha=len(batch_x))
            n_seen += len(batch_x)

        train_loss = total_loss.item() / max(n_seen, 1)
        val_loss = _eval_loss(val_loader)
        monitor_loss = val_loss if val_loss is not None else train_loss
        if not np.isfinite(train_loss) or not np.isfinite(monitor_loss):
            raise FloatingPointError("分类AE监控损失不是有限数值")
        scheduler.step(monitor_loss)
        current_lr = optimizer.param_groups[0]['lr']

        history.append({
            "epoch": epoch + 1,
            "train_loss": float(train_loss),
            "val_loss": float(val_loss) if val_loss is not None else np.nan,
            "monitor_loss": float(monitor_loss),
            "lr": float(current_lr),
        })

        improved = monitor_loss < (best_metric - min_delta)
        if improved:
            best_metric = monitor_loss
            best_epoch = epoch + 1
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1

        if (epoch + 1) % 5 == 0 or epoch == 0 or improved:
            if val_loss is not None:
                print(f"    Epoch [{epoch+1}/{epochs}], train={train_loss:.6f}, val={val_loss:.6f}, best={best_metric:.6f}@{best_epoch}, lr={current_lr:.2e}")
            else:
                print(f"    Epoch [{epoch+1}/{epochs}], train={train_loss:.6f}, best={best_metric:.6f}@{best_epoch}, lr={current_lr:.2e}")

        if patience > 0 and no_improve >= patience:
            print(f"  Early Stopping触发：连续 {patience} 轮无显著改善，停止于Epoch {epoch+1}，最佳Epoch {best_epoch}")
            break

    if best_state is None:
        raise RuntimeError("分类AE没有有效最优权重")
    if best_state is not None:
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
        print(f"  已回滚到最佳AE权重：Epoch {best_epoch}, best_monitor_loss={best_metric:.6f}")

    # 导出训练曲线，便于判断扩大epoch是否真的有效。
    if bool(getattr(cfg, "AE_EXPORT_TRAINING_CURVE", True)):
        try:
            ae_dir = os.path.join(cfg.OUTPUT_VISUALIZATION, "ae_training")
            os.makedirs(ae_dir, exist_ok=True)
            hist_df = pd.DataFrame(history)
            hist_df.to_csv(os.path.join(ae_dir, "ae_training_history.csv"), index=False, encoding="utf-8-sig")
            setup_thesis_plot_style()
            fig, ax = plt.subplots(figsize=(8.4, 4.7))
            ax.plot(hist_df["epoch"], hist_df["train_loss"], label="Training loss",
                    color=EI_COLORS["blue"], linewidth=1.8)
            if hist_df["val_loss"].notna().any():
                ax.plot(hist_df["epoch"], hist_df["val_loss"], label="Validation loss",
                        color=EI_COLORS["orange"], linewidth=1.8)
            ax.axvline(best_epoch, color=EI_COLORS["gray"], linestyle="--", linewidth=1.0,
                       label=f"Best epoch = {best_epoch}")
            best_row = hist_df.loc[hist_df["epoch"] == best_epoch].iloc[0]
            best_y = best_row["val_loss"] if np.isfinite(best_row["val_loss"]) else best_row["train_loss"]
            ax.scatter([best_epoch], [best_y], s=34, color=EI_COLORS["red"], edgecolor="white", zorder=5)
            ax.annotate(f"Best loss = {best_y:.5f}", (best_epoch, best_y), xytext=(7, 10),
                        textcoords="offset points", fontsize=7.5,
                        bbox={"boxstyle": "round,pad=0.22", "fc": "white", "ec": "#BBBBBB", "alpha": 0.95})
            ax.set_xlabel("Epoch")
            ax.set_ylabel("MSE reconstruction loss")
            ax.set_title("AutoEncoder training convergence")
            style_axis(ax, "y")
            ax.legend(frameon=False)
            save_thesis_figure(fig, os.path.join(ae_dir, "ae_training_curve.png"))
        except Exception as exc:
            logger.warning(f"AE训练曲线保存失败: {exc}")

    print("  提取编码特征和重构误差...")
    model.eval()
    Z_with_error = encode_scaled_autoencoder_batches(X_tensor, model, batch_size)
    reconstruction_error = Z_with_error[:, -1:]
    print(f"  AE特征形状: {(len(Z_with_error), Z_with_error.shape[1] - 1)}")
    print(f"  重构误差形状: {reconstruction_error.shape}")
    print(f"  最终输出形状: {Z_with_error.shape}")
    print(f"  重构误差均值: {float(reconstruction_error.mean()):.6f}, 中位数: {float(np.median(reconstruction_error)):.6f}")

    logger.info(f"AutoEncoder特征编码完成，输出形状: {Z_with_error.shape}, best_epoch={best_epoch}, best_loss={best_metric:.6f}")
    if return_artifacts:
        # 论文/服务器测试版：仅训练集拟合AE和其内部StandardScaler，
        # 验证集/测试集只复用该模型和scaler做transform，避免数据泄漏。
        return Z_with_error, used_indices, model, scaler
    return Z_with_error, used_indices

def _mean_rbf_kernel_blockwise(X, Y, gamma, block_size):
    """精确累计全部样本对（包括对角线），不采用近似核或额外抽样。"""
    total = 0.0
    for left in range(0, len(X), block_size):
        for right in range(0, len(Y), block_size):
            distances = pairwise_distances(X[left:left + block_size], Y[right:right + block_size],
                                           metric="sqeuclidean")
            np.multiply(distances, -gamma, out=distances)
            np.exp(distances, out=distances)
            total += float(distances.sum(dtype=np.float64))
    return total / (len(X) * len(Y))


@timer_decorator
def calculate_mmd(X: np.ndarray, Y: np.ndarray, gamma: Optional[float] = None, sample_size: Optional[int] = None) -> float:
    """计算最大均值差异（MMD）：保留原抽样和带对角线估计，核矩阵分块累计。"""
    logger.info("开始计算MMD")
    cfg = get_active_config()
    if sample_size is None:
        sample_size = int(getattr(cfg, "MMD_SAMPLE_SIZE", 15000))

    rng = np.random.default_rng(cfg.SEED)
    n_X, n_Y = len(X), len(Y)

    if n_X > sample_size:
        X_sample = X[rng.choice(n_X, sample_size, replace=False)]
        logger.info(f"对X数据进行采样，从 {n_X:,} 减少到 {sample_size:,}")
    else:
        X_sample = X

    if n_Y > sample_size:
        Y_sample = Y[rng.choice(n_Y, sample_size, replace=False)]
        logger.info(f"对Y数据进行采样，从 {n_Y:,} 减少到 {sample_size:,}")
    else:
        Y_sample = Y

    X_sample = np.nan_to_num(X_sample.astype(np.float32, copy=False), nan=0, posinf=0, neginf=0)
    Y_sample = np.nan_to_num(Y_sample.astype(np.float32, copy=False), nan=0, posinf=0, neginf=0)

    if gamma is None:
        gamma = 1.0 / max(1, X_sample.shape[1])

    block_size = max(1, int(getattr(cfg, "MMD_BLOCK_SIZE", 1024)))
    mmd = (_mean_rbf_kernel_blockwise(X_sample, X_sample, gamma, block_size)
           + _mean_rbf_kernel_blockwise(Y_sample, Y_sample, gamma, block_size)
           - 2 * _mean_rbf_kernel_blockwise(X_sample, Y_sample, gamma, block_size))
    logger.info(f"MMD计算完成：{mmd:.6f}")
    return float(mmd)


def calculate_wasserstein_distance(X: np.ndarray, Y: np.ndarray, sample_size: Optional[int] = None) -> float:
    """计算一维Wasserstein距离（Earth Mover's Distance）
    
    对每列特征分别计算Wasserstein距离，再取均值作为总体距离度量
    """
    from scipy.stats import wasserstein_distance
    
    cfg = get_active_config()
    if sample_size is None:
        sample_size = int(getattr(cfg, "WASSERSTEIN_SAMPLE_SIZE", 50000))
    rng = np.random.default_rng(cfg.SEED)
    n_X = len(X)
    n_Y = len(Y)

    if n_X > sample_size:
        X = X[rng.choice(n_X, sample_size, replace=False)]
    if n_Y > sample_size:
        Y = Y[rng.choice(n_Y, sample_size, replace=False)]
    X = np.nan_to_num(X.astype(np.float32, copy=False), nan=0, posinf=0, neginf=0)
    Y = np.nan_to_num(Y.astype(np.float32, copy=False), nan=0, posinf=0, neginf=0)
    
    distances = []
    for i in range(X.shape[1]):
        dist = wasserstein_distance(X[:, i], Y[:, i])
        distances.append(dist)
    
    return np.mean(distances)


@timer_decorator
def balance_data(df, feature_cols):
    """沿用自适应目标与KMeansSMOTE→SMOTE回退；仅连续AE插值，类别/端口继承。"""
    cfg = get_active_config()
    frame = enforce_standard_labels(df, "before_mixed_asmote", copy=False)
    feature_cols = [c for c in feature_cols if c in frame and c not in PROVENANCE_COLUMNS
                    and pd.api.types.is_numeric_dtype(frame[c])]
    continuous = [c for c in feature_cols if c.startswith("AE_")]
    if not continuous:
        raise ValueError("缺少连续AE空间，不能在原始端口/类别整数上执行SMOTE")
    labels = frame.Label.to_numpy()
    original_counts = frame.Label.value_counts()
    targets = build_adaptive_sampling_targets(original_counts, include_normal=False)
    strategy = {k:v for k,v in targets.items() if original_counts[k]>=cfg.SMOTE_MIN_REQUIRED}
    copies = {k:v for k,v in targets.items() if original_counts[k]<cfg.SMOTE_MIN_REQUIRED}
    provenance = [pd.DataFrame({"is_synthetic":False,"synthetic_method":"original",
                    "donor_row":np.arange(len(frame),dtype=np.int64),
                    "donor_source":frame.dataset_source.astype(str).to_numpy() if "dataset_source" in frame else "unknown",
                    "origin_note":"real_training_row_after_IF"},index=frame.index)]
    parts = [frame]
    generated_counts = {}
    method = "none"
    if strategy:
        if cfg.MIXED_RESAMPLING_MODE != "inherit":
            raise ValueError("本版按用户选择使用连续插值＋离散继承，不接受未验证的模式")
        from sklearn.cluster import MiniBatchKMeans
        metric_scaler = StandardScaler()
        matrix = metric_scaler.fit_transform(frame[continuous].to_numpy(dtype=np.float32))
        # 仅为近邻/聚类距离统一尺度，不改变AE训练输入或原样本输出。
        sampler = KMeansSMOTE(sampling_strategy=strategy,k_neighbors=1,cluster_balance_threshold=0.,
                    kmeans_estimator=MiniBatchKMeans(n_clusters=min(cfg.KMEANS_SMOTE_CLUSTERS,len(frame)),
                        random_state=cfg.SEED,batch_size=1024,n_init=3),
                    random_state=cfg.SEED)
        try:
            resampled,y_resampled = sampler.fit_resample(matrix,labels)
            method = "KMeansSMOTE_continuous_with_discrete_donor"
        except Exception as exc:
            logger.warning("KMeansSMOTE失败，回退SMOTE: %s",exc)
            try:
                resampled,y_resampled = SMOTE(sampling_strategy=strategy,k_neighbors=1,
                                        random_state=cfg.SEED).fit_resample(matrix,labels)
                method = "SMOTE_continuous_with_discrete_donor"
            except Exception as fallback:
                raise RuntimeError("两种过采样均失败；禁止将未增强数据标为成功") from fallback
        synthetic_matrix = resampled[len(frame):]
        synthetic_labels = y_resampled[len(frame):]
        new_values = metric_scaler.inverse_transform(synthetic_matrix).astype(np.float32)
        for label in np.unique(synthetic_labels):
            positions = np.flatnonzero(synthetic_labels == label)
            needed = int(strategy[label]-original_counts[label])
            if len(positions)>needed:
                # 按簇向上取整可能多生成：只丢弃超额合成行，不删除真实攻击。
                positions = np.sort(np.random.default_rng(cfg.SEED).choice(positions,needed,replace=False))
            candidates = np.flatnonzero(labels == label)
            nearest = NearestNeighbors(n_neighbors=1,n_jobs=-1).fit(matrix[candidates])
            # 小类优先，候选仅同类真实样本；不把合成结果作为后续类别捐赠者。
            donor = candidates[nearest.kneighbors(synthetic_matrix[positions],return_distance=False)[:,0]]
            part = frame.iloc[donor].copy().reset_index(drop=True)
            part.loc[:,continuous] = new_values[positions]
            part["dataset_source"] = "synthetic"
            add_port_derived_features(part,classification_state()["port_frequency"],
                                      classification_state()["port_pair_frequency"])
            parts.append(part)
            provenance.append(pd.DataFrame({"is_synthetic":True,"synthetic_method":method,
                "donor_row":donor,"donor_source":frame.dataset_source.iloc[donor].astype(str).to_numpy(),
                "origin_note":"donor identifies discrete/port inheritance; interpolation parents not tracked"}))
            generated_counts[str(label)] = len(part)
    rng = np.random.default_rng(cfg.SEED)
    for label,target in copies.items():
        candidates = np.flatnonzero(labels == label)
        donor = rng.choice(candidates,int(target-original_counts[label]),replace=True)
        part = frame.iloc[donor].copy().reset_index(drop=True)
        part["dataset_source"] = "synthetic"
        parts.append(part)
        provenance.append(pd.DataFrame({"is_synthetic":True,"synthetic_method":"random_copy_tiny",
            "donor_row":donor,"donor_source":frame.dataset_source.iloc[donor].astype(str).to_numpy(),
            "origin_note":"exact_copy_of_real_training_donor"}))
        generated_counts[str(label)] = len(part)
    result = pd.concat(parts,ignore_index=True,copy=False)
    details = pd.concat(provenance,ignore_index=True)
    if len(details) != len(result) or len(result)<len(frame):
        raise RuntimeError("过采样溯源/行数守恒检查失败")
    final_counts = result.Label.value_counts()
    shortfalls = {k:{"target":int(v),"actual":int(final_counts.get(k,0))}
                  for k,v in targets.items() if final_counts.get(k,0)<v}
    if shortfalls:
        raise RuntimeError(f"过采样目标未达成: {shortfalls}")
    assert_no_class_disappears(frame,result,"after_mixed_asmote")
    directory = os.path.join(cfg.OUTPUT_DIR,"balance_debug_logs")
    atomic_csv(details,os.path.join(directory,"train_sample_provenance.csv"))
    cfg._resampling_provenance_path = os.path.join(directory,"train_sample_provenance.csv")
    write_json(os.path.join(directory,"smote_status.json"),
                   {"status":"completed","method":method,"generated":generated_counts,
                    "original_rows":len(frame),"final_rows":len(result),"categorical_values_inherited":True,
                    "synthetic_source":"synthetic","normal_not_oversampled":True,
                    "fully_equal_class_counts":bool(final_counts.nunique()==1)})
    atomic_csv(pd.DataFrame({"Original_Count":original_counts,"Final_Count":final_counts}).fillna(0).reset_index(),
               os.path.join(directory,"enhancement_stats.csv"))
    if cfg.EXPORT_SAMPLING_AUDIT_PLOTS:
        plot_sampling_audit(original_counts,final_counts,final_counts,targets,directory)
    return result
    
# 此分支在原始CSV读取时旁路采集，在原分类流程结束后独立拟合和导出。
# 不改分类train/val/test、Normal下采样、AE→IF→ASMOTE及原有可视化。
# 不把SMOTE合成行放回序列，IF只提供异常分数。
# 时间戳不进入AE/分类特征。原分类AE不可复用，因为其训练集是随机划分的。
# 参考数据说明：https://www.unb.ca/cic/datasets/ids-2017.html
#                 https://www.unb.ca/cic/datasets/ids-2018.html


def write_json(path, data):
    def cast(x):
        if isinstance(x, dict):
            return {str(k): cast(v) for k, v in x.items()}
        if isinstance(x, (list, tuple)):
            return [cast(v) for v in x]
        if isinstance(x, np.ndarray):
            return cast(x.tolist())
        if isinstance(x, np.generic):
            return cast(x.item())
        if isinstance(x, float) and not np.isfinite(x):
            return None
        return x
    os.makedirs(os.path.dirname(path), exist_ok=True)
    temporary = path + ".tmp"
    with open(temporary, "w", encoding="utf-8") as handle:
        json.dump(cast(data), handle, ensure_ascii=False, indent=2, allow_nan=False)
    os.replace(temporary, path)


def _sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def save_performance_report(cfg):
    try:
        write_json(os.path.join(cfg.OUTPUT_DIR, "performance_report.json"), {
            "revision": cfg.OPTIMIZATION_REVISION,
            "measurement_note": "RSS peak sampled every 0.25s, may miss shorter spikes; CUDA peak=cumulative",
            "memory_peak":getattr(cfg,"_memory_peak",{}),
            "memory_budget_observations":getattr(cfg,"_memory_observations",[]),
            "stages": getattr(cfg, "_performance_records", [])})
    except Exception as exc:
        logger.warning("性能报告保存失败: %s", exc)


# -------------------------- 5. 可视化函数 --------------------------
def plot_label_distribution_comparison(df_before: pd.DataFrame, df_after: pd.DataFrame,
                                       title: str, save_path: str):
    """论文版：绘制处理前后标签分布对比图。"""
    try:
        setup_thesis_plot_style()
        # 此图只使用Label；不要为类别计数复制完整特征表。
        counts_before = df_before if isinstance(df_before,pd.Series) else map_traffic_label_series(df_before["Label"]).value_counts()
        counts_after = df_after if isinstance(df_after,pd.Series) else map_traffic_label_series(df_after["Label"]).value_counts()
        all_labels = ordered_standard_labels(list(set(counts_before.index) | set(counts_after.index)))
        if not all_labels:
            return

        plot_df = pd.DataFrame({
            "Label": all_labels,
            "Before processing": [int(counts_before.get(l, 0)) for l in all_labels],
            "After processing": [int(counts_after.get(l, 0)) for l in all_labels],
        }).melt(id_vars="Label", var_name="Stage", value_name="Count")

        fig, ax = plt.subplots(figsize=(10.8, max(5.3, 0.38 * len(all_labels))))
        sns.barplot(data=plot_df, y="Label", x="Count", hue="Stage", orient="h", ax=ax,
                    palette={"Before processing": EI_COLORS["blue"], "After processing": EI_COLORS["orange"]},
                    edgecolor="white", linewidth=0.35)
        ax.set_xscale("log")
        ax.set_xlabel("Number of samples (log scale)")
        ax.set_ylabel("")
        ax.set_title(title, pad=10)
        annotate_horizontal_bars(ax, fontsize=6.3, kind="count")
        ax.margins(x=0.20)
        style_axis(ax, "x")
        ax.legend(title="Stage", frameon=False, loc="lower right")
        save_thesis_figure(fig, save_path)
        logger.info(f"论文版标签分布对比图已保存: {save_path}")
    except Exception as e:
        logger.warning(f"绘制论文版标签分布对比图失败: {e}")

def plot_ae_pca_tsne(df_original: pd.DataFrame, ae_features: np.ndarray, 
                    labels: np.ndarray, save_path: str):
    """论文版AutoEncoder降维前后的PCA/t-SNE可视化。"""
    if not get_active_config().EXPORT_HEAVY_VISUALS:
        return
    try:

        count = min(len(df_original),len(ae_features),len(labels))
        idx = stratified_plot_indices(np.asarray(labels)[:count],get_active_config().PLOT_SAMPLE_SIZE,get_active_config().SEED)
        df_original = df_original.iloc[idx].reset_index(drop=True)
        ae_features, labels = np.asarray(ae_features)[idx], np.asarray(labels)[idx]
        from sklearn.decomposition import PCA
        from sklearn.manifold import TSNE
        
        labels_standardized = np.asarray([str(canonicalize_traffic_label(l) or config.LABEL_MAPPING.get(str(l), str(l))) for l in labels], dtype=object)
        
        setup_thesis_plot_style()
        fig = plt.figure(figsize=(14.6, 9.4))
        
        plt.subplot(2, 2, 1)
        
        numeric_cols = df_original.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 50:
            numeric_cols = numeric_cols[:50]
        
        X_original = df_original[numeric_cols].values
        X_original = np.nan_to_num(X_original, nan=0)
        
        min_len = min(len(X_original), len(labels_standardized))
        X_original = X_original[:min_len]
        labels_plot = labels_standardized[:min_len]
        
        pca_original = PCA(n_components=2, random_state=42)
        X_pca_original = pca_original.fit_transform(X_original)
        
        unique_labels = list(set(labels_plot))
        unique_labels = sorted(unique_labels)
        colors = EI_SEQUENCE[:len(unique_labels)]
        color_map = {label: colors[i] for i, label in enumerate(unique_labels)}
        
        for label in unique_labels:
            mask = np.array(labels_plot) == label
            if mask.sum() > 0:
                plt.scatter(X_pca_original[mask, 0], X_pca_original[mask, 1],
                           c=[color_map[label]], label=f"{label} (n={int(mask.sum()):,})",
                           alpha=0.56, s=10, linewidths=0, rasterized=True)
        
        plt.title('AE input continuous features (PCA)', fontsize=12, fontweight='bold')
        plt.xlabel(f'PC1 ({pca_original.explained_variance_ratio_[0]*100:.1f}%)')
        plt.ylabel(f'PC2 ({pca_original.explained_variance_ratio_[1]*100:.1f}%)')
        plt.legend(loc='best', fontsize=5.8, ncol=2, frameon=False, markerscale=1.5)
        
        plt.subplot(2, 2, 2)
        
        sample_size = min(config.PLOT_SAMPLE_SIZE, len(X_original))
        if len(X_original) > sample_size:
            indices = np.random.choice(len(X_original), sample_size, replace=False)
            X_sample = X_original[indices]
            labels_sample = labels_plot[indices]
        else:
            X_sample = X_original
            labels_sample = labels_plot
        
        tsne_perplexity = max(2, min(30, (len(X_sample) - 1) // 3))
        tsne = TSNE(n_components=2, random_state=42, perplexity=tsne_perplexity, max_iter=1000)
        X_tsne = tsne.fit_transform(X_sample)
        
        for label in unique_labels:
            mask = np.array(labels_sample) == label
            if mask.sum() > 0:
                plt.scatter(X_tsne[mask, 0], X_tsne[mask, 1],
                           c=[color_map.get(label, 'gray')], label=f"{label} (n={int(mask.sum()):,})",
                           alpha=0.56, s=10, linewidths=0, rasterized=True)
        
        plt.title('AE input continuous features (t-SNE)', fontsize=12, fontweight='bold')
        plt.xlabel('t-SNE 1')
        plt.ylabel('t-SNE 2')
        plt.legend(loc='best', fontsize=5.8, ncol=2, frameon=False, markerscale=1.5)
        
        plt.subplot(2, 2, 3)
        
        min_len_ae = min(len(ae_features), len(labels_standardized))
        ae_features_plot = ae_features[:min_len_ae]
        labels_plot_ae = labels_standardized[:min_len_ae]
        
        pca_ae = PCA(n_components=2, random_state=42)
        X_pca_ae = pca_ae.fit_transform(ae_features_plot)
        
        for label in unique_labels:
            mask = np.array(labels_plot_ae) == label
            if mask.sum() > 0:
                plt.scatter(X_pca_ae[mask, 0], X_pca_ae[mask, 1],
                           c=[color_map.get(label, 'gray')], label=f"{label} (n={int(mask.sum()):,})",
                           alpha=0.56, s=10, linewidths=0, rasterized=True)
        
        plt.title('AE Features (PCA)', fontsize=12, fontweight='bold')
        plt.xlabel(f'PC1 ({pca_ae.explained_variance_ratio_[0]*100:.1f}%)')
        plt.ylabel(f'PC2 ({pca_ae.explained_variance_ratio_[1]*100:.1f}%)')
        plt.legend(loc='best', fontsize=5.8, ncol=2, frameon=False, markerscale=1.5)
        
        plt.subplot(2, 2, 4)
        
        if len(ae_features_plot) > sample_size:
            indices_ae = np.random.choice(len(ae_features_plot), sample_size, replace=False)
            ae_sample = ae_features_plot[indices_ae]
            labels_ae_sample = labels_plot_ae[indices_ae]
        else:
            ae_sample = ae_features_plot
            labels_ae_sample = labels_plot_ae
        
        tsne_ae_perplexity = max(2, min(30, (len(ae_sample) - 1) // 3))
        tsne_ae = TSNE(n_components=2, random_state=42, perplexity=tsne_ae_perplexity, max_iter=1000)
        X_tsne_ae = tsne_ae.fit_transform(ae_sample)
        
        for label in unique_labels:
            mask = np.array(labels_ae_sample) == label
            if mask.sum() > 0:
                plt.scatter(X_tsne_ae[mask, 0], X_tsne_ae[mask, 1],
                           c=[color_map.get(label, 'gray')], label=f"{label} (n={int(mask.sum()):,})",
                           alpha=0.56, s=10, linewidths=0, rasterized=True)
        
        plt.title('AE Features (t-SNE)', fontsize=12, fontweight='bold')
        plt.xlabel('t-SNE 1')
        plt.ylabel('t-SNE 2')
        plt.legend(loc='best', fontsize=5.8, ncol=2, frameon=False, markerscale=1.5)
        
        for idx, ax in enumerate(fig.axes):
            if idx < 4:
                style_axis(ax, "both")
                add_panel_label(ax, f"({chr(97 + idx)})")
        plt.suptitle('Feature-space structure before and after AutoEncoder encoding',
                     fontsize=12.5, fontweight='semibold', y=1.01)
        save_thesis_figure(fig, save_path)
        
        logger.info(f"AutoEncoder降维可视化已保存: {save_path}")
        print(f"  [可视化] AutoEncoder降维可视化已保存")
        
    except Exception as e:
        logger.warning(f"绘制AutoEncoder降维可视化失败: {e}")

def plot_ae_latent_tsne(ae_features, labels, save_path):
    """纯latent特征；按所有类别合计的总预算抽样，不将误差列混入。"""
    if not get_active_config().EXPORT_HEAVY_VISUALS:
        return
    from sklearn.manifold import TSNE
    cfg = get_active_config()
    idx = stratified_plot_indices(np.asarray(labels),cfg.PLOT_SAMPLE_SIZE,cfg.SEED)
    if len(idx)<4:
        logger.warning("t-SNE样本过少，跳过")
        return
    values, sampled = np.asarray(ae_features)[idx], np.asarray(labels)[idx]
    projected = TSNE(n_components=2,perplexity=min(30,max(2,(len(idx)-1)//3)),
                     random_state=cfg.SEED).fit_transform(values)
    fig,ax = plt.subplots(figsize=(10.4,8.2))
    for number,label in enumerate(sorted(np.unique(sampled))):
        mask = sampled == label
        ax.scatter(projected[mask,0],projected[mask,1],s=9,alpha=.58,
                   color=EI_SEQUENCE[number % len(EI_SEQUENCE)],label=f"{label} (sample n={int(mask.sum())})",
                   rasterized=True)
    ax.set_title(f"AE latent t-SNE (total sampled n={len(idx):,})")
    ax.legend(fontsize=6,ncol=2,frameon=False,bbox_to_anchor=(1.01,1))
    save_thesis_figure(fig,save_path)

def plot_if_anomaly_distribution(ae_features: np.ndarray, outlier_labels: np.ndarray,
                                labels: np.ndarray, save_path: str):
    """绘制Isolation Forest异常点分布图"""
    if not get_active_config().EXPORT_HEAVY_VISUALS:
        return
    try:

        count = min(len(ae_features),len(outlier_labels),len(labels))
        idx = stratified_plot_indices(np.asarray(labels)[:count],get_active_config().PLOT_SAMPLE_SIZE,get_active_config().SEED)
        ae_features, outlier_labels, labels = np.asarray(ae_features)[idx], np.asarray(outlier_labels)[idx], np.asarray(labels)[idx]
        from sklearn.decomposition import PCA
        
        min_len = min(len(ae_features), len(outlier_labels), len(labels))
        ae_features = ae_features[:min_len]
        outlier_labels = outlier_labels[:min_len]
        labels = labels[:min_len]
        
        setup_thesis_plot_style()
        fig = plt.figure(figsize=(14.4, 6.8))
        
        pca = PCA(n_components=2, random_state=42)
        X_pca = pca.fit_transform(ae_features)
        
        plt.subplot(1, 2, 1)
        
        colors_if = [EI_COLORS["red"] if x == -1 else EI_COLORS["blue"] for x in outlier_labels]
        
        plt.scatter(X_pca[:, 0], X_pca[:, 1], c=colors_if, alpha=0.52, s=10,
                    linewidths=0, rasterized=True)
        
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor=EI_COLORS["blue"], label=f'Inlier (n={sum(outlier_labels == 1):,})'),
            Patch(facecolor=EI_COLORS["red"], label=f'Outlier (n={sum(outlier_labels == -1):,})')
        ]
        plt.legend(handles=legend_elements, loc='upper right', fontsize=10)
        
        plt.title('Isolation Forest Anomaly Detection\n(AE Features)', 
                 fontsize=12, fontweight='bold')
        plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)')
        plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)')
        
        plt.subplot(1, 2, 2)
        
        unique_labels = np.unique(labels)
        colors = EI_SEQUENCE[:len(unique_labels)]
        
        for idx, label in enumerate(unique_labels):
            mask = labels == label
            plt.scatter(X_pca[mask, 0], X_pca[mask, 1],
                      c=[colors[idx]], label=f'{label} (n={sum(mask)})', 
                      alpha=0.52, s=10, linewidths=0, rasterized=True)
        
        plt.title('True Labels Distribution\n(AE Features)', 
                 fontsize=12, fontweight='bold')
        plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)')
        plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)')
        plt.legend(loc='best', fontsize=6.0, ncol=2, frameon=False, markerscale=1.8)
        
        for idx, ax in enumerate(fig.axes[:2]):
            style_axis(ax, "both")
            add_panel_label(ax, f"({chr(97 + idx)})")
        plt.suptitle('Isolation Forest decisions in the AutoEncoder feature space',
                     fontsize=12.5, fontweight='semibold', y=1.01)
        save_thesis_figure(fig, save_path)
        
        logger.info(f"Isolation Forest异常分布图已保存: {save_path}")
        print(f"  [可视化] Isolation Forest异常分布图已保存")
        
    except Exception as e:
        logger.warning(f"绘制Isolation Forest异常分布图失败: {e}")

def plot_dataset_alignment_kde(df_2017, df_2018, feature_cols, save_path):
    """同一诊断坐标系下的来源分布；不把重新拟合PCA称作数据对齐。"""
    cfg = get_active_config()
    if not cfg.EXPORT_HEAVY_VISUALS:
        return
    cols = [c for c in feature_cols if c in df_2017 and c in df_2018 and not is_excluded_feature(c)]
    if len(cols)<2:
        return
    n = min(cfg.DISTANCE_SAMPLE_SIZE,len(df_2017),len(df_2018))
    samples = [frame.sample(n=n,random_state=cfg.SEED)[cols].apply(pd.to_numeric,errors="coerce")
               for frame in (df_2017,df_2018)]
    arrays = [np.nan_to_num(frame.to_numpy(dtype=np.float64),nan=0,posinf=0,neginf=0) for frame in samples]
    from sklearn.decomposition import PCA
    joined = np.vstack(arrays)
    coordinates = PCA(n_components=2,random_state=cfg.SEED).fit_transform(RobustScaler().fit_transform(joined))
    left,right = coordinates[:n],coordinates[n:]
    mmd = calculate_mmd(left,right,sample_size=n)
    fig,axes = plt.subplots(1,2,figsize=(12.6,5.4))
    for values,label,color in [(left,"2017",EI_COLORS["blue"]),(right,"2018",EI_COLORS["orange"])]:
        sns.kdeplot(x=values[:,0],label=label,color=color,ax=axes[0],warn_singular=False)
    axes[0].set_title(f"Shared diagnostic projection (MMD={mmd:.4g})")
    axes[0].legend()
    for values,label,color in [(arrays[0],"2017",EI_COLORS["blue"]),(arrays[1],"2018",EI_COLORS["orange"])]:
        sns.kdeplot(x=values[:,0],label=label,color=color,ax=axes[1],warn_singular=False)
    axes[1].set_title(f"Same raw feature: {cols[0]}")
    axes[1].legend()
    fig.suptitle("Source distribution diagnostics — no distribution-changing alignment applied")
    save_thesis_figure(fig,save_path)

def plot_distribution_distance_curve(stages_data, save_path):
    """仅展示已计算的诊断值；不同坐标空间不作为对齐收益曲线。"""
    if not stages_data:
        return
    names = list(stages_data)
    fig,axes = plt.subplots(1,2,figsize=(12,4.8))
    for ax,key in zip(axes,["mmd","wasserstein"]):
        ax.bar(np.arange(len(names)),[stages_data[s][key] for s in names],color=EI_COLORS["blue"])
        ax.set_xticks(np.arange(len(names)),names,rotation=15,ha="right")
        ax.set_title(key.upper()+" diagnostic")
    fig.suptitle("Measured distribution diagnostics — not evidence of alignment improvement")
    save_thesis_figure(fig,save_path)
    write_json(os.path.join(os.path.dirname(save_path),"distribution_distance_scope.json"),
                   {"stages":stages_data,"claim":"diagnostics_only_no_alignment_improvement_claim",
                    "sample_limit":get_active_config().DISTANCE_SAMPLE_SIZE})


def plot_class_distribution_comparison(raw_counts, smote_counts, train_counts, test_counts,
                                      save_dir: str, logger: logging.Logger):
    """绘制类别分布对比图：原始/最终平衡后/train/test 四个阶段。"""
    try:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        os.makedirs(save_dir, exist_ok=True)

        all_labels = sorted(
            set(raw_counts.index) | set(smote_counts.index) |
            set(train_counts.index) | set(test_counts.index),
            key=lambda x: str(x)
        )

        df_plot = pd.DataFrame({
            'Label': all_labels,
            'Raw Data': [raw_counts.get(l, 0) for l in all_labels],
            'Final Balanced': [smote_counts.get(l, 0) for l in all_labels],
            'Train Set': [train_counts.get(l, 0) for l in all_labels],
            'Test Set': [test_counts.get(l, 0) for l in all_labels],
        })

        setup_thesis_plot_style()
        fig, axes = plt.subplots(2, 2, figsize=(14.2, 10.4))
        titles = ['Raw Data Distribution', 'Final Balanced Distribution', 'Train Set', 'Test Set']
        count_lists = [raw_counts, smote_counts, train_counts, test_counts]
        colors = [EI_COLORS["blue"], EI_COLORS["green"], EI_COLORS["orange"], EI_COLORS["purple"]]

        for ax, title, counts, color in zip(axes.flat, titles, count_lists, colors):
            plot_df = pd.DataFrame({'Label': all_labels, 'Count': [counts.get(l, 0) for l in all_labels]})
            sns.barplot(data=plot_df, y='Label', x='Count', ax=ax, color=color, alpha=0.94,
                        edgecolor="white", linewidth=0.35)
            ax.set_xscale('log')
            ax.set_title(title, pad=8)
            ax.set_ylabel('')
            ax.set_xlabel('Number of samples (log scale)')
            annotate_horizontal_bars(ax, fontsize=5.8, kind="count")
            ax.margins(x=0.18)
            style_axis(ax, 'x')

        for idx, ax in enumerate(axes.flat):
            add_panel_label(ax, f"({chr(97 + idx)})")
        fig.suptitle('Class distributions at different stages', fontsize=12.5, fontweight='semibold', y=1.01)
        path = os.path.join(save_dir, f"class_distribution_comparison_{timestamp}.png")
        save_thesis_figure(fig, path)
        logger.info(f"类别分布对比图已保存: {path}")

        hist_path = os.path.join(save_dir, f"class_distribution_history_{timestamp}.csv")
        df_plot.to_csv(hist_path, index=False, encoding='utf-8-sig')
        logger.info(f"类别分布历史数据已保存: {hist_path}")

        print(f"\n【类别分布统计】")
        header = f"{'类别':<20} {'原始':>8} {'最终平衡后':>12} {'训练集':>8} {'测试集':>8}"
        print(header)
        print("-" * 60)
        for lbl in all_labels:
            r = raw_counts.get(lbl, 0)
            s = smote_counts.get(lbl, 0)
            tr = train_counts.get(lbl, 0)
            te = test_counts.get(lbl, 0)
            print(f"{str(lbl):<20} {r:>8} {s:>12} {tr:>8} {te:>8}")

    except Exception as e:
        logger.warning(f"绘制类别分布对比图失败: {e}")

def standardize_labels_for_plot(labels: list) -> list:
    """将标签标准化为配置中的标准类别标准标签；无法映射的标签返回NaN并在绘图统计中自动丢弃。"""
    standardized = []
    for label in labels:
        mapped = canonicalize_traffic_label(label)
        standardized.append(mapped if mapped is not None else np.nan)
    return standardized

@timer_decorator
def plot_label_distribution(df: pd.DataFrame, title: str, save_path: str):
    """论文版单阶段标签分布图：横向条形图 + 对数坐标。"""
    if "Label" not in df.columns or len(df) == 0:
        return
    try:
        setup_thesis_plot_style()
        df_copy = df.copy()
        df_copy["Label"] = standardize_labels_for_plot(df_copy["Label"].tolist())
        label_counts = df_copy["Label"].value_counts()
        labels = ordered_standard_labels(label_counts.index.tolist())
        label_counts = label_counts.reindex(labels).fillna(0).astype(int)
        label_counts = label_counts[label_counts > 0]
        if label_counts.empty:
            return

        fig, ax = plt.subplots(figsize=(9.4, max(5.0, 0.34 * len(label_counts))))
        bars = ax.barh(label_counts.index[::-1], label_counts.values[::-1],
                       color=EI_COLORS["blue"], edgecolor="white", linewidth=0.4, zorder=3)
        ax.set_xscale("log")
        ax.set_xlabel("Number of samples (log scale)")
        ax.set_ylabel("")
        ax.set_title(title, pad=10)
        annotate_horizontal_bars(ax, fontsize=7.2, kind="count")
        ax.margins(x=0.18)
        style_axis(ax, "x")
        save_thesis_figure(fig, save_path)
    except Exception as e:
        logger.warning(f"绘制论文版标签分布图失败: {e}")


@timer_decorator
def plot_feature_distributions_multiple(df: pd.DataFrame, feature_cols: List[str], save_dir: str):
    """绘制多张特征分布图（每张4个特征，共len(feature_cols)//4张）
    
    参数:
        df: 数据框
        feature_cols: 特征列表
        save_dir: 保存目录
    """
    n_features = len(feature_cols)
    n_per_image = 4
    n_images = (n_features + n_per_image - 1) // n_per_image
    
    logger.info(f"绘制{n_images}张特征分布图，每张{n_per_image}个特征...")
    
    for img_idx in range(n_images):
        start_idx = img_idx * n_per_image
        end_idx = min(start_idx + n_per_image, n_features)
        current_features = feature_cols[start_idx:end_idx]
        
        n_cols = 2
        n_rows = 2
        
        setup_thesis_plot_style()
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(10.8, 7.6), squeeze=False)
        for i, col in enumerate(current_features):
            ax = axes.flat[i]
            try:
                values = pd.to_numeric(df[col], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
                sns.histplot(values, kde=True, color=EI_SEQUENCE[i], ax=ax,
                             edgecolor="white", linewidth=0.25)
                ax.set_title(f"{col}  (n = {len(values):,})")
                ax.set_xlabel("Feature value")
                ax.set_ylabel("Frequency")
                style_axis(ax, "y")
            except Exception as e:
                logger.warning(f"绘制特征{col}失败: {e}")
                ax.text(0.5, 0.5, f"Feature {col} unavailable", ha='center', va='center', transform=ax.transAxes)
        for ax in axes.flat[len(current_features):]:
            ax.set_visible(False)
        fig.suptitle(f"AE feature distributions ({img_idx+1}/{n_images})", y=1.01,
                     fontsize=12.5, fontweight="semibold")
        save_path = os.path.join(save_dir, f"特征分布_{img_idx+1}.png")
        save_thesis_figure(fig, save_path)
        
        logger.info(f"第{img_idx+1}张特征分布图已保存: {save_path}")

@timer_decorator
def plot_correlation_heatmap(df: pd.DataFrame, feature_cols: List[str], title: str, save_path: str):
    """论文版相关性热力图：只展示高方差特征，三角矩阵，矢量输出。"""
    if len(feature_cols) < 2:
        logger.warning("特征数量不足，无法绘制相关性热力图")
        return
    try:
        setup_thesis_plot_style()
        candidate_features = [c for c in feature_cols if c in df.columns and pd.api.types.is_numeric_dtype(df[c])]
        if not candidate_features:
            return
        variances = df[candidate_features].var(numeric_only=True).sort_values(ascending=False)
        use_features = [c for c in variances.index[:min(24, len(variances))] if variances[c] > 1e-12]
        if len(use_features) < 2:
            logger.warning("有效特征数量不足，无法绘制相关性热力图")
            return

        corr = df[use_features].corr().replace([np.inf, -np.inf], np.nan).fillna(0)
        mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
        fig, ax = plt.subplots(figsize=(11.2, 9.8))
        sns.heatmap(corr, mask=mask, cmap="vlag", vmin=-1, vmax=1, center=0,
                    square=True, linewidths=0.45, linecolor="white", annot=True, fmt=".2f",
                    annot_kws={"fontsize": 5.9},
                    cbar_kws={"shrink": 0.72, "label": "Pearson correlation coefficient"}, ax=ax)
        ax.set_title(f"{title} (top-{len(use_features)} variance features)", pad=14)
        ax.tick_params(axis="x", rotation=45, labelsize=8)
        ax.tick_params(axis="y", rotation=0, labelsize=8)
        save_thesis_figure(fig, save_path)
        logger.info(f"论文版相关性热力图已保存: {save_path}")
    except Exception as e:
        logger.warning(f"绘制论文版相关性热力图失败: {e}")


@timer_decorator
def plot_dataset_source_distribution(df: pd.DataFrame, save_path: str):
    """论文版数据集来源分布饼图：同时标明精确数量与占比。"""
    if "dataset_source" not in df.columns or len(df) == 0:
        logger.warning("数据集中没有dataset_source列，无法绘制来源分布图")
        return
    try:
        setup_thesis_plot_style()
        source_counts = df["dataset_source"].astype(str).value_counts().sort_index()
        fig, ax = plt.subplots(figsize=(7.2, 5.2))
        values = source_counts.values.astype(int)
        total = int(values.sum())

        def _autopct(pct):
            count = int(round(pct / 100.0 * total))
            return f"{pct:.1f}%\n(n = {count:,})"

        wedges, _, autotexts = ax.pie(
            values,
            labels=None,
            colors=EI_SEQUENCE[:len(source_counts)],
            startangle=90,
            counterclock=False,
            autopct=_autopct,
            pctdistance=0.66,
            wedgeprops={"width": 0.48, "edgecolor": "white", "linewidth": 1.2},
            textprops={"fontsize": 8.2, "color": "#222222"},
        )
        for text in autotexts:
            text.set_fontweight("semibold")
        legend_labels = [f"{source}: {count:,} samples" for source, count in source_counts.items()]
        ax.legend(wedges, legend_labels, title="Dataset source", loc="center left",
                  bbox_to_anchor=(0.98, 0.5), frameon=False)
        ax.text(0, 0, f"Total\n{total:,}", ha="center", va="center", fontsize=9.2,
                fontweight="semibold", color="#333333")
        ax.set_title("Dataset-source composition", pad=10)
        ax.set_aspect("equal")
        save_thesis_figure(fig, save_path)
        logger.info(f"论文版数据集来源分布饼图已保存: {save_path}")
    except Exception as e:
        logger.warning(f"绘制论文版数据集来源分布图失败: {e}")

@timer_decorator
def plot_label_by_source(df: pd.DataFrame, save_path: str):
    """论文版：不同数据源下各类别分布，横向分组条形图。"""
    if "dataset_source" not in df.columns or "Label" not in df.columns or len(df) == 0:
        logger.warning("数据集中缺少必要的列，无法绘制标签来源分布图")
        return
    try:
        setup_thesis_plot_style()
        df_plot = df.copy()
        df_plot["Label"] = standardize_labels_for_plot(df_plot["Label"].tolist())
        df_plot["dataset_source"] = df_plot["dataset_source"].astype(str)
        labels = ordered_standard_labels(df_plot["Label"].unique().tolist())
        cross_tab = pd.crosstab(df_plot["Label"], df_plot["dataset_source"]).reindex(labels).fillna(0).astype(int)
        if cross_tab.empty:
            return
        cross_tab = cross_tab.iloc[::-1]
        fig, ax = plt.subplots(figsize=(11.2, max(5.3, 0.38 * len(cross_tab))))
        colors = EI_SEQUENCE
        cross_tab.plot(kind="barh", stacked=False, ax=ax, color=colors[:len(cross_tab.columns)],
                       edgecolor="white", linewidth=0.5, zorder=3)
        ax.set_xscale("log")
        ax.set_xlabel("Number of samples (log scale)")
        ax.set_ylabel("")
        ax.set_title("Class distribution by dataset source", pad=10)
        annotate_horizontal_bars(ax, fontsize=5.8, kind="count")
        ax.margins(x=0.20)
        style_axis(ax, "x")
        ax.legend(title="Dataset source", frameon=False, loc="lower right")
        save_thesis_figure(fig, save_path)
        logger.info(f"论文版标签来源分布图已保存: {save_path}")
    except Exception as e:
        logger.warning(f"绘制论文版标签来源分布图失败: {e}")


def plot_ae_2d_scatter(df: pd.DataFrame, x_col: str, y_col: str, save_path: str) -> None:
    """论文版AE二维特征空间图；仅对显示数据分层抽样，不改变处理结果。"""
    if any(c not in df.columns for c in [x_col, y_col, "Label"]) or len(df) == 0:
        return
    try:
        setup_thesis_plot_style()
        plot_df = df[[x_col, y_col, "Label"]].copy()
        plot_df["Label"] = standardize_labels_for_plot(plot_df["Label"].tolist())
        plot_df = plot_df.dropna(subset=[x_col, y_col, "Label"])
        full_counts = plot_df["Label"].value_counts()
        labels = ordered_standard_labels(full_counts.index.tolist())
        max_points = int(getattr(config, "PLOT_SAMPLE_SIZE", 50000))
        per_class = max(1, int(np.ceil(max_points / max(len(labels), 1))))
        sampled_parts = []
        for label in labels:
            part = plot_df[plot_df["Label"] == label]
            if len(part) > per_class:
                part = part.sample(n=per_class, random_state=getattr(config, "SEED", 42))
            sampled_parts.append(part)
        sampled_df = pd.concat(sampled_parts, ignore_index=True) if sampled_parts else plot_df.iloc[:0]

        fig, ax = plt.subplots(figsize=(10.2, 7.6))
        for idx, label in enumerate(labels):
            subset = sampled_df[sampled_df["Label"] == label]
            if subset.empty:
                continue
            ax.scatter(subset[x_col], subset[y_col], color=EI_SEQUENCE[idx % len(EI_SEQUENCE)],
                       label=f"{label} (total n={int(full_counts.get(label, 0)):,})",
                       alpha=0.55, s=9, linewidths=0, rasterized=True)
        ax.set_xlabel(x_col)
        ax.set_ylabel(y_col)
        ax.set_title(f"AutoEncoder feature space (displayed n = {len(sampled_df):,})")
        ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1), ncol=1,
                  fontsize=6.5, markerscale=2.0, frameon=False)
        style_axis(ax, "both")
        save_thesis_figure(fig, save_path)
    except Exception as exc:
        logger.warning(f"绘制论文版AE二维特征空间图失败: {exc}")

# -------------------------- 6. 主处理函数 --------------------------

# ==================== 模块化纠错/可复用预处理工具 ====================
PROVENANCE_COLUMNS = {"is_synthetic", "synthetic_method", "donor_row", "donor_source",
                      "origin_note", "row_fingerprint", "flow_uid"}

class IndexedBatchTensor:
    """批量行号视图，保持快速整批取数，避免复制AE内部Train/Holdout矩阵。"""
    def __init__(self,tensor,indices):
        self.tensor = tensor
        self.indices = torch.as_tensor(indices,dtype=torch.long)
    def __len__(self):
        return len(self.indices)
    def __getitem__(self,index):
        return self.tensor[self.indices[index]]

def seed_everything(cfg):
    random.seed(cfg.SEED)
    np.random.seed(cfg.SEED)
    torch.manual_seed(cfg.SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.SEED)
    if getattr(cfg, "DETERMINISTIC_TORCH", False):
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark = False

def stratified_plot_indices(labels, limit, seed):
    """总采样预算，不是每类预算；稀有类别优先保留。"""
    labels = np.asarray(labels)
    limit = min(len(labels),max(1,int(limit)))
    if len(labels)<=limit:
        return np.arange(len(labels))
    rng = np.random.default_rng(seed)
    chosen = []
    classes,counts = np.unique(labels,return_counts=True)
    allocation = np.minimum(counts,max(1,limit//len(classes)))
    while int(allocation.sum())>limit:
        candidates = np.flatnonzero(allocation>0)
        allocation[candidates[-1]] -= 1
    remaining = limit-int(allocation.sum())
    for i in np.argsort(counts):
        take = min(remaining,int(counts[i]-allocation[i]))
        allocation[i] += take
        remaining -= take
    for label,n in zip(classes,allocation):
        if n:
            chosen.append(rng.choice(np.flatnonzero(labels==label),int(n),replace=False))
    return np.concatenate(chosen)

def save_balanced_training_alias(train_path, balanced_path):
    temporary = str(balanced_path)+".tmp"
    try:
        if get_active_config().BALANCED_ALIAS_MODE!="hardlink":
            raise OSError("用户选择独立副本")
        os.link(train_path,temporary)
    except OSError:
        ensure_disk_capacity(os.path.dirname(balanced_path),os.path.getsize(train_path))
        shutil.copyfile(train_path,temporary)
    os.replace(temporary,balanced_path)


def classification_state():
    cfg = get_active_config()
    if not hasattr(cfg, "_classification_state"):
        cfg._classification_state = {}
    return cfg._classification_state

def canonical_feature_name(name):
    key = normalize_col(name)
    return get_active_config().FEATURE_MAPPING.get(key, key.replace(" ", "_"))

def is_excluded_feature(name):
    key = normalize_col(name)
    compact = re.sub(r"[^a-z0-9]", "", key)
    excluded = {normalize_col(x) for x in get_active_config().DROP_FEATURES}
    return (key in excluded or compact in {
        "flowid", "flowuid", "timestamp", "timestart", "starttime", "flowstarttime",
        "date", "time", "datetime", "endtime", "flowendtime", "startdate", "enddate",
        "starttimestamp", "endtimestamp", "eventtime", "capturetime", "flowtimestamp",
        "firstseen", "lastseen", "unixtime", "timestampms", "timestampus", "timestampns",
        "timeend", "endtime", "srcip", "dstip", "sourceip", "destinationip",
        "srcipaddress", "dstipaddress", "sourceipaddress", "destinationipaddress",
        "externalip", "unnamed0"} or "ip" in key.split())

def assert_feature_schema(frame):
    if frame.columns.has_duplicates:
        raise ValueError("canonical映射后仍有重复列")
    bad = [c for c in frame if c not in ({"Label", "dataset_source"} | PROVENANCE_COLUMNS) and is_excluded_feature(c)]
    if bad:
        raise ValueError(f"分类输入含禁止字段: {bad}")

def canonicalize_feature_frame(frame, source):
    """按固定规则归一同义列；冲突不静默保留第一列。"""
    cfg = get_active_config()
    groups, result = {}, {}
    for col in frame.columns:
        if col in {"Label", "dataset_source"} or col in PROVENANCE_COLUMNS:
            result[col] = frame[col]
        elif not is_excluded_feature(col):
            groups.setdefault(canonical_feature_name(col), []).append(col)
    for canonical, columns in groups.items():
        values = frame[columns[0]]
        for alias in columns[1:]:
            other = frame[alias]
            overlap = values.notna() & other.notna()
            a = pd.to_numeric(values[overlap], errors="coerce")
            b = pd.to_numeric(other[overlap], errors="coerce")
            numeric = a.notna() & b.notna()
            equal = values[overlap].astype(str).eq(other[overlap].astype(str)).to_numpy(copy=True)
            if numeric.any():
                equal[numeric.to_numpy()] = np.isclose(a[numeric], b[numeric], rtol=1e-6, atol=1e-8)
            if not equal.all():
                raise ValueError(f"来源{source}同义列{columns}存在{int((~equal).sum())}个冲突；请核实，不能静默丢列")
            values = values.fillna(other)
        multiplier = float(cfg.FEATURE_UNIT_MULTIPLIERS.get(str(source), {}).get(canonical, 1.))
        if multiplier != 1.:
            if not np.isfinite(multiplier) or multiplier <= 0:
                raise ValueError("单位换算倍率必须为有限正数")
            values = pd.to_numeric(values, errors="coerce") * multiplier
        result[canonical] = values
    if str(source) in {"2017","2018"}:
        classification_state().setdefault("source_feature_presence",{})[str(source)] = sorted(groups)
    out = pd.DataFrame(result, index=frame.index,copy=False)
    assert_feature_schema(out)
    return out

def category_keys(series):
    """统一数值型类别在CSV中整数/浮点表示不同的问题，文本类别原样保留。"""
    def key(value):
        try:
            number = float(value)
            if np.isfinite(number):
                return format(number,".17g")
        except (TypeError,ValueError):
            pass
        return str(value)
    mapping = {v:key(v) for v in pd.unique(series.dropna())}
    return series.map(mapping).fillna("Unknown").astype(str)

def add_port_derived_features(frame, frequency, pair_frequency):
    for col, table in frequency.items():
        value = frame[col]
        frame[f"{col}_well_known"] = value.between(0, 1023).astype(np.int8)
        frame[f"{col}_registered"] = value.between(1024, 49151).astype(np.int8)
        frame[f"{col}_dynamic"] = value.between(49152, 65535).astype(np.int8)
        frame[f"{col}_common_services"] = value.isin([20,21,22,23,25,53,80,443,3306,3389,5432]).astype(np.int8)
        frame[f"{col}_freq"] = value.astype(str).map(table).fillna(0.).astype(np.float32)
    if pair_frequency and {"src_port", "dst_port"}.issubset(frame):
        pair = frame.src_port.astype(str) + "_" + frame.dst_port.astype(str)
        frame["port_pair_freq"] = pair.map(pair_frequency).fillna(0.).astype(np.float32)
    return frame

def ensure_disk_capacity(directory, estimated_bytes=0):
    os.makedirs(directory, exist_ok=True)
    cfg = get_active_config()
    free = shutil.disk_usage(directory).free
    reserve = int(float(cfg.DISK_FREE_MIN_GB) * 1024**3)
    required = reserve + int(estimated_bytes * cfg.OUTPUT_SIZE_SAFETY_FACTOR)
    if free < required:
        raise OSError(f"输出空间不足: free={free/1024**3:.2f}GiB，需要约{required/1024**3:.2f}GiB"
                      "（含安全余量）。请更换DATAHANDLE_OUTPUT_ROOT，不会自动删除旧结果。")

def atomic_csv(frame, path, **kwargs):
    """按块写入，逐块检查空间；gzip临时快照使用显式压缩，避免.tmp后缀丢失压缩。"""
    directory=os.path.dirname(os.path.abspath(path));os.makedirs(directory,exist_ok=True)
    temporary=str(path)+".tmp"
    options={"index":False,"encoding":"utf-8-sig"};options.update(kwargs)
    chunk_size=int(options.pop("chunksize",get_active_config().IO_CHUNK_ROWS))
    encoding=options.pop("encoding");options.pop("compression",None)
    opener=gzip.open if str(path).endswith(".gz") else open
    extra={"compresslevel":1} if str(path).endswith(".gz") else {}
    try:
        with opener(temporary,"wt",encoding=encoding,newline="",**extra) as handle:
            for start in range(0,max(1,len(frame)),chunk_size):
                ensure_disk_capacity(directory)
                frame.iloc[start:start+chunk_size].to_csv(handle,header=start==0,**options)
        os.replace(temporary,path)
    except Exception:
        if os.path.isfile(temporary):os.remove(temporary)
        raise

def estimated_csv_bytes(frames, compressed=False):
    total=0
    for frame in frames:
        if frame is None or frame.empty:continue
        sample=frame.iloc[np.linspace(0,len(frame)-1,min(2000,len(frame)),dtype=int)]
        raw=sample.to_csv(index=False).encode("utf-8-sig")
        size=len(gzip.compress(raw,compresslevel=1,mtime=0)) if compressed else len(raw)
        total+=int(size/max(1,len(sample))*len(frame))
    return total

def pre_ae_disk_precheck(train,val,test,continuous,discrete,ports):
    cfg=get_active_config()
    counts=train.Label.value_counts()
    targets=build_adaptive_sampling_targets(counts,include_normal=False)
    generated=sum(max(0,int(n)-int(counts.get(label,0))) for label,n in targets.items())
    latent=min(cfg.AE_LATENT_MAX,max(8,len(continuous)//2))
    features=latent+1+len(discrete)+len(ports)*6+(1 if len(ports)>=2 else 0)
    hardlink=cfg.BALANCED_ALIAS_MODE=="hardlink" and detect_hardlink_support(cfg.OUTPUT_DIR)
    cfg._hardlink_supported=hardlink
    train_bytes=(len(train)+generated)*(features*16+100)
    components={"classification_csv_estimate":train_bytes+(len(val)+len(test))*(features*16+100),
        "balanced_alias_additional":0 if hardlink else train_bytes,
        "augmentation_provenance_estimate":(len(train)+generated)*160,
        "model_and_audit_allowance":64*1024**2,
        "forecast_window_csv_estimate":sum(getattr(cfg,"_forecast_precheck",{}).get("complete_windows",{}).values())*(features*20+1000)}
    estimate=sum(components.values())
    write_json(os.path.join(cfg.OUTPUT_DIR,"disk_precheck.json"),{
        "stage":"before_AE","additional_bytes_by_component":components,"additional_total_estimate":estimate,
        "already_written_raw_spools_bytes":sum(os.path.getsize(p) for p in getattr(cfg,"_forecast_spools",{}).values()),
        "hardlink_supported":hardlink,"safety_factor":cfg.OUTPUT_SIZE_SAFETY_FACTOR,
        "fixed_reserve_GiB":cfg.DISK_FREE_MIN_GB,"estimate_not_exact":True})
    ensure_disk_capacity(cfg.OUTPUT_DIR,estimate)

def record_run_status(status, **extra):
    cfg = get_active_config()
    write_json(os.path.join(cfg.OUTPUT_DIR, "run_status.json"),
                   {"status": status, "revision": cfg.OPTIMIZATION_REVISION, **extra})
    if getattr(cfg, "EXPORT_FUSED_FORECAST", False):
        preparing = status == "running" and extra.get("phase") == "reading"
        failed = status == "failed" and not os.path.isfile(
                    os.path.join(cfg.OUTPUT_DIR,"forecast_fused","temporal_metadata.json"))
        if preparing or failed:
            write_json(os.path.join(cfg.MAIN_OUTPUT_DIR,"latest_fused_forecast_summary.json"),
                {"status":"preparing" if preparing else "failed", "run_dir":cfg.OUTPUT_DIR,
                 "error":extra.get("error", "新数据处理中，请等待本次导出完成")})

def save_classification_artifacts(model, ae_scaler, robust_scaler, ae_features, side_features):
    """保存唯一的主融合处理器，预测窗口只重放这套冻结参数。"""
    import joblib
    cfg = get_active_config()
    folder = os.path.join(cfg.OUTPUT_DIR, "classification_preprocessing")
    os.makedirs(folder, exist_ok=True)
    state = classification_state()
    state.update(schema_version="classification_v1", fit_split="train",
                 ae_input_features=list(ae_features), side_features=list(side_features),
                 output_ae_columns=[f"AE_{i}" for i in range(model.encoder[-1].out_features+1)],
                 label_names=list(cfg.STANDARD_LABELS), feature_mapping=cfg.FEATURE_MAPPING,
                 unit_multipliers=cfg.FEATURE_UNIT_MULTIPLIERS,
                 ae_input_scale_mode=cfg.AE_INPUT_SCALE_MODE,
                 optimization_revision=cfg.OPTIMIZATION_REVISION)
    state["environment"] = environment_versions()
    write_json(os.path.join(folder, "preprocessing.json"), state)
    temporary = os.path.join(folder, "scalers.joblib.tmp")
    joblib.dump({"robust":robust_scaler, "ae":ae_scaler}, temporary)
    os.replace(temporary, os.path.join(folder, "scalers.joblib"))
    temporary = os.path.join(folder, "autoencoder.pt.tmp")
    torch.save({"state_dict":{k:v.detach().cpu() for k,v in model.state_dict().items()},
                "input_dim":len(ae_features), "latent_dim":model.encoder[-1].out_features,
                "dropout":cfg.AE_DROPOUT, "use_layer_norm":cfg.AE_USE_LAYER_NORM}, temporary)
    os.replace(temporary, os.path.join(folder, "autoencoder.pt"))

class ClassificationTransformer:
    """新流量只transform，不fit、不采样；输出与分类CSV一致的模型数值特征。"""
    def __init__(self, artifact_dir, batch_size=4096, device=None):
        import joblib
        with open(os.path.join(artifact_dir,"preprocessing.json"), encoding="utf-8") as handle:
            self.state = json.load(handle)
        if self.state.get("fit_split") != "train":
            raise ValueError("预处理器不是仅在Train拟合的版本")
        self.scalers = joblib.load(os.path.join(artifact_dir,"scalers.joblib"))
        try:
            checkpoint = torch.load(os.path.join(artifact_dir,"autoencoder.pt"), map_location="cpu", weights_only=True)
        except TypeError:
            checkpoint = torch.load(os.path.join(artifact_dir,"autoencoder.pt"), map_location="cpu")
        self.model = AutoEncoder(checkpoint["input_dim"], checkpoint["latent_dim"],
                                 checkpoint["dropout"], checkpoint["use_layer_norm"])
        self.model.load_state_dict(checkpoint["state_dict"])
        self.model.to(torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu")))
        self.model.eval()
        self.batch_size = batch_size
    def transform(self, frame, source=None):
        # 以保存版本的固定schema进行归一，不能依赖后来修改过的Config。
        if not len(frame):
            return pd.DataFrame(columns=self.state["output_ae_columns"]+self.state["side_features"],dtype=np.float32)
        mapping = self.state["feature_mapping"]
        groups = {}
        for col in frame:
            canonical = mapping.get(normalize_col(col), normalize_col(col).replace(" ","_"))
            groups.setdefault(canonical, []).append(col)
        values = {}
        required = self.state["ae_input_features"] + list(self.state["discrete_mappings"]) + self.state["port_features"]
        for col in required:
            aliases = groups.get(col, [])
            if not aliases:
                expected = self.state.get("source_feature_presence",{}).get(str(source))
                if expected is None or col in expected:
                    raise ValueError(f"新流量缺少该来源应有的训练schema特征: {col}，不能重新拟合")
                values[col] = pd.Series(np.nan,index=frame.index)
                continue
            value = frame[aliases[0]].copy()
            for alias in aliases[1:]:
                overlap = value.notna() & frame[alias].notna()
                left = pd.to_numeric(value[overlap],errors="coerce")
                right = pd.to_numeric(frame.loc[overlap,alias],errors="coerce")
                numeric = left.notna() & right.notna()
                equal = value[overlap].astype(str).eq(frame.loc[overlap,alias].astype(str)).to_numpy(copy=True)
                if numeric.any():
                    equal[numeric.to_numpy()] = np.isclose(left[numeric],right[numeric],rtol=1e-6,atol=1e-8)
                if not equal.all():
                    raise ValueError(f"新流量同义列冲突: {col}")
                value = value.fillna(frame[alias])
            factor = self.state["unit_multipliers"].get(str(source), {}).get(col, 1.)
            if factor != 1.:
                value = pd.to_numeric(value, errors="coerce") * factor
            values[col] = value
        out = pd.DataFrame(values, index=frame.index)
        for col,value in self.state["numeric_fill"].items():
            if col in out:
                out[col] = pd.to_numeric(out[col],errors="coerce").replace([np.inf,-np.inf],np.nan).fillna(value)
        for col,value in self.state["categorical_fill"].items():
            if col in out:
                if col=="protocol" or "flag" in col:
                    value_series=pd.to_numeric(out[col],errors="coerce").replace([np.inf,-np.inf],np.nan)
                    invalid=(value_series<0) | (value_series%1 != 0)
                    if col=="protocol":invalid |= value_series>255
                    out[col]=value_series.mask(invalid,np.nan)
                elif pd.api.types.is_numeric_dtype(out[col]):
                    out[col]=out[col].replace([np.inf,-np.inf],np.nan)
                out[col] = out[col].fillna(value)
        continuous = self.state["ae_input_features"]
        for col in continuous:
            a = out[col].to_numpy(dtype=np.float64,copy=True) + self.state["log_shifts"][col]
            out[col] = np.log1p(np.maximum(a,0)).astype(np.float32)
        block = out[continuous].to_numpy(dtype=np.float32,copy=True)
        clipping = self.state.get("robust_clip",{})
        if clipping:
            for col, bounds in clipping.items():
                i = continuous.index(col)
                block[:,i] = np.clip(block[:,i],*bounds)
        out[continuous] = self.scalers["robust"].transform(block).astype(np.float32)
        for col,mapping in self.state["discrete_mappings"].items():
            out[col] = category_keys(out[col]).map(mapping).fillna(-1).astype(np.int32)
        for col in self.state["port_features"]:
            value = pd.to_numeric(out[col],errors="coerce").astype(np.float64)
            invalid = value.isna() | ~np.isfinite(value) | (value<0) | (value>65535) | (value%1 != 0)
            out[col] = value.mask(invalid,-1).astype(np.int32)
        add_port_derived_features(out, self.state["port_frequency"], self.state["port_pair_frequency"])
        encoded = transform_with_fitted_autoencoder(out[continuous].to_numpy(dtype=np.float32),
                      self.model, self.scalers["ae"], self.batch_size)
        result = pd.DataFrame(encoded,columns=self.state["output_ae_columns"],index=out.index)
        for col in self.state["side_features"]:
            result[col] = out[col].to_numpy()
        if not np.isfinite(result.to_numpy(dtype=float)).all():
            raise ValueError("新流量预处理产生NaN/Inf")
        return result

def classification_row_fingerprints(frame):
    """忽略来源/标签，对规范化特征做保守指纹分组；不删除重复真实流。"""
    cols = [c for c in frame if c not in {"Label","dataset_source"} | PROVENANCE_COLUMNS]
    combined = np.zeros(len(frame),dtype=np.uint64)
    known = set(get_active_config().FEATURE_MAPPING.values())
    for i,col in enumerate(sorted(cols)):
        value = frame[col]
        if col in known or pd.api.types.is_numeric_dtype(value):
            value = pd.to_numeric(value,errors="coerce").astype(np.float64)
        hashed = pd.util.hash_pandas_object(value,index=False,categorize=True).to_numpy(dtype=np.uint64)
        combined ^= hashed + np.uint64(0x9e3779b97f4a7c15) + (combined << np.uint64(6)) + (combined >> np.uint64(2))
    return combined

def plot_analysis_sample(frame, n=None):
    cfg = get_active_config()
    limit = int(n or cfg.PLOT_SAMPLE_SIZE)
    return frame.sample(n=min(len(frame),limit),random_state=cfg.SEED)

def environment_versions():
    import importlib.metadata
    versions={"python":sys.version,"platform":platform.platform(),"cuda_available":torch.cuda.is_available(),
              "torch_cuda_version":torch.version.cuda}
    for name in ["numpy","pandas","scikit-learn","imbalanced-learn","torch","joblib"]:
        try:versions[name]=importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:versions[name]="unknown"
    return versions


@timer_decorator
def main():
    """主处理函数 - 新流程"""
    global config
    config = Config()
    validate_config(config)
    seed_everything(config)

    print("=" * 60)
    print("新数据处理流程启动")
    print("=" * 60)
    print(f"输出目录: {config.OUTPUT_DIR}")
    
    try:
        stages_data = {}
        viz_stage_snapshots = {}

        def record_viz_stage(stage_name: str, df: Optional[pd.DataFrame], feature_cols: Optional[List[str]] = None):
            """记录阶段快照，只保存统计摘要和小样本，避免复制完整数据集。"""
            if not getattr(config, "EXPORT_COMPREHENSIVE_STAGE_VISUALS", True):
                return
            try:
                snap = make_stage_snapshot(stage_name, df, feature_cols=feature_cols)
                viz_stage_snapshots[stage_name] = snap
                print(f"[可视化快照] {stage_name}: rows={snap.get('rows', 0):,}, labels={len(snap.get('label_counts', pd.Series(dtype=int)))}")
            except Exception as snap_exc:
                logger.warning(f"记录可视化阶段快照失败({stage_name}): {snap_exc}")

        def compute_distances(X_2017: np.ndarray, X_2018: np.ndarray, stage_name: str, n_sample: Optional[int] = None):
            """计算MMD和Wasserstein距离"""
            if n_sample is None:
                n_sample = int(getattr(config, "MMD_SAMPLE_SIZE", 15000))
            n_17 = min(len(X_2017), n_sample)
            n_18 = min(len(X_2018), n_sample)
            idx_17 = np.random.choice(len(X_2017), n_17, replace=False) if len(X_2017) > n_17 else np.arange(len(X_2017))
            idx_18 = np.random.choice(len(X_2018), n_18, replace=False) if len(X_2018) > n_18 else np.arange(len(X_2018))
            X17 = np.nan_to_num(X_2017[idx_17], nan=0, posinf=0, neginf=0)
            X18 = np.nan_to_num(X_2018[idx_18], nan=0, posinf=0, neginf=0)
            if X17.shape[1] != X18.shape[1]:
                return
            gamma = 1.0 / X17.shape[1]
            mmd_val = calculate_mmd(X17, X18, gamma=gamma, sample_size=n_sample)
            w_dist = calculate_wasserstein_distance(X17, X18, sample_size=n_sample)
            stages_data[stage_name] = {"mmd": mmd_val, "wasserstein": w_dist}
            print(f"  [{stage_name}] MMD={mmd_val:.4f}, Wasserstein={w_dist:.4f}")

        os.makedirs(config.OUTPUT_DIR, exist_ok=True)
        os.makedirs(config.OUTPUT_VISUALIZATION, exist_ok=True)
        ensure_disk_capacity(config.OUTPUT_DIR)
        record_run_status("running", phase="reading")
        write_json(os.path.join(config.OUTPUT_DIR,"environment.json"),environment_versions())
        write_json(os.path.join(config.OUTPUT_DIR, "effective_config.json"), {
            k: getattr(config, k) for k in dir(config)
            if k.isupper() and isinstance(getattr(config,k),(str,int,float,bool,dict,list,tuple,type(None)))})
        start_memory_monitor(config)
        
        # ===== 阶段1：跨数据集对齐 - 四类映射策略 =====
        print("\n" + "=" * 60)
        print("【阶段1：跨数据集对齐 - 四类映射策略】")
        print("=" * 60)
        
        df_2017 = None
        df_2018 = None
        
        if os.path.exists(config.FOLDER_2017):
            try:
                df_2017 = read_multi_csv(config.FOLDER_2017, "2017")
                if df_2017 is not None and len(df_2017) > 0:
                    df_2017["dataset_source"] = "2017"
                    df_2017 = enforce_standard_labels(df_2017, stage_name="01_Raw_2017_after_read", label_col="Label", copy=False)
                    print(f"2017数据集读取完成: {len(df_2017)} 行")
                    record_viz_stage("01_Raw_2017", df_2017)
                else:
                    df_2017 = None
                    print("读取2017失败: 读取结果为空")
            except Exception as e:
                if config.STRICT_INPUT_FILES:
                    raise
                df_2017 = None
                print(f"读取2017失败: {e}")
        else:
            print(f"2017数据集路径不存在，跳过: {config.FOLDER_2017}")
        
        if os.path.exists(config.FOLDER_2018):
            try:
                df_2018 = read_multi_csv(config.FOLDER_2018, "2018")
                if df_2018 is not None and len(df_2018) > 0:
                    df_2018["dataset_source"] = "2018"
                    df_2018 = enforce_standard_labels(df_2018, stage_name="01_Raw_2018_after_read", label_col="Label", copy=False)
                    print(f"2018数据集读取完成: {len(df_2018)} 行")
                    record_viz_stage("01_Raw_2018", df_2018)
                else:
                    df_2018 = None
                    print("读取2018失败: 读取结果为空")
            except Exception as e:
                if config.STRICT_INPUT_FILES:
                    raise
                df_2018 = None
                print(f"读取2018失败: {e}")
        else:
            print(f"2018数据集路径不存在，跳过: {config.FOLDER_2018}")
        
        if df_2017 is None and df_2018 is None:
            raise ValueError("未找到任何数据集：请检查 config.FOLDER_2017 / config.FOLDER_2018 路径及CSV文件是否可读")
        write_json(os.path.join(config.OUTPUT_DIR,"input_manifest.json"),
                       sorted(config._input_manifest,key=lambda x:(x["source"],x["file"])))

        # 原始分布距离仅作诊断，独立的可控小样本，不参与任何拟合或样本选择。
        # 否则跳过该步骤，避免 df_2017 / df_2018 为 None 时访问 .columns 报错。
        if config.DISTANCE_ANALYSIS_ENABLED and df_2017 is not None and df_2018 is not None:
            numeric_cols_raw = [
                c for c in df_2017.columns
                if c not in ["Label", "dataset_source"]
                and c in df_2018.columns
                and pd.api.types.is_numeric_dtype(df_2017[c])
                and pd.api.types.is_numeric_dtype(df_2018[c])
            ]
            if numeric_cols_raw:
                # 与compute_distances原来相同的抽样顺序/数量；先抽行再转数组。
                distance_sample = int(getattr(config, "DISTANCE_SAMPLE_SIZE", 5000))
                idx17 = np.random.choice(len(df_2017), distance_sample, replace=False) if len(df_2017) > distance_sample else np.arange(len(df_2017))
                idx18 = np.random.choice(len(df_2018), distance_sample, replace=False) if len(df_2018) > distance_sample else np.arange(len(df_2018))
                X_17_raw = np.nan_to_num(df_2017.iloc[idx17][numeric_cols_raw].to_numpy(dtype=np.float64), nan=0, posinf=0, neginf=0)
                X_18_raw = np.nan_to_num(df_2018.iloc[idx18][numeric_cols_raw].to_numpy(dtype=np.float64), nan=0, posinf=0, neginf=0)
                compute_distances(X_17_raw, X_18_raw, "Raw shared-name features (diagnostic only)",n_sample=distance_sample)
                del X_17_raw, X_18_raw
            else:
                print("[Raw Data] 跳过分布距离计算：2017/2018没有共同数值特征")
        else:
            missing = "2017" if df_2017 is None else "2018"
            print("[Raw Data] 分布距离诊断已关闭" if not config.DISTANCE_ANALYSIS_ENABLED
                  else f"[Raw Data] 跳过分布距离计算：{missing}数据集未成功读取")

        cols_2017_original = len(df_2017.columns) - 1 if df_2017 is not None else 0
        cols_2018_original = len(df_2018.columns) - 1 if df_2018 is not None else 0
        
        print("\n[列名标准化] 统一字段格式...")
        print("\n[映射前统计]")
        if df_2017 is not None:
            print(f"  2017数据集: {len(df_2017.columns)} 列")
        if df_2018 is not None:
            print(f"  2018数据集: {len(df_2018.columns)} 列")
        
        if df_2017 is not None:
            df_2017 = normalize_dataframe_columns(df_2017, print_stats=True)
        if df_2018 is not None:
            df_2018 = normalize_dataframe_columns(df_2018, print_stats=True)
        
        print("\n[映射后统计]")
        if df_2017 is not None:
            print(f"  2017数据集: {len(df_2017.columns)} 列")
        if df_2018 is not None:
            print(f"  2018数据集: {len(df_2018.columns)} 列")
        
        print("\n[删除重复列]")
        if df_2017 is not None:
            orig_cols = len(df_2017.columns)
            if df_2017.columns.has_duplicates:
                df_2017 = drop_duplicate_columns_fast(df_2017, stage_name="2017")
            else:
                print("  2017无重复列，跳过整表切片")
            dup_removed = orig_cols - len(df_2017.columns)
        if df_2018 is not None:
            orig_cols = len(df_2018.columns)
            if df_2018.columns.has_duplicates:
                df_2018 = drop_duplicate_columns_fast(df_2018, stage_name="2018")
            else:
                print("  2018无重复列，跳过整表切片")
            dup_removed = orig_cols - len(df_2018.columns)
        
        if df_2017 is not None:
            print("  [2017] 清理异常格式与数值类型快速识别（大表延迟全列转换）...")
            df_2017 = clean_malformed_rows(df_2017)
            df_2017 = convert_object_columns_to_numeric_fast(df_2017)
        if df_2018 is not None:
            print("  [2018] 清理异常格式与数值类型快速识别（大表延迟全列转换）...")
            df_2018 = clean_malformed_rows(df_2018)
            df_2018 = convert_object_columns_to_numeric_fast(df_2018)
        
        mapping_output_dir = os.path.join(config.OUTPUT_DIR, "feature_mapping_analysis")
        os.makedirs(mapping_output_dir, exist_ok=True)
        
        mapping_result = None
        if df_2017 is not None and df_2018 is not None:
            mapping_result = run_four_category_mapping(df_2017, df_2018, mapping_output_dir)
            export_mapping_summary_to_csv(mapping_result, mapping_output_dir, 
                                         cols_2017_original, cols_2018_original)
        
        # ===== 阶段2：数据融合 - 四类映射策略 =====
        print("\n" + "=" * 60)
        print("【阶段2：数据融合】")
        print("=" * 60)
        
        if df_2017 is not None and df_2018 is not None and mapping_result is not None:
            df_combined = merge_datasets_with_four_categories(df_2017, df_2018, mapping_result)
        elif df_2017 is not None and df_2018 is not None:
            common_features = list(set(df_2017.columns) & set(df_2018.columns))
            common_features = [f for f in common_features if f not in ["Label", "dataset_source"]]
            print(f"公共特征数: {len(common_features)}")
            
            for col in common_features:
                if col not in df_2017.columns:
                    df_2017[col] = 0
                if col not in df_2018.columns:
                    df_2018[col] = 0
            
            df_combined = pd.concat([df_2017, df_2018], axis=0, ignore_index=True)
            print(f"合并后数据量: {len(df_combined)}")
        else:
            df_combined = df_2017 if df_2017 is not None else df_2018
            df_combined = df_combined.copy()
            print(f"单数据集数据量: {len(df_combined)}")
        
        df_combined = enforce_standard_labels(df_combined, stage_name="02_Merged_AfterMapping", label_col="Label", copy=False)
        df_combined = canonicalize_feature_frame(df_combined, "single_or_merged")
        forecast_require_row_identity(df_combined, config, "主融合规范化后")
        # 全量输入已经融合；后续源表仅用于绘图，保留确定性小样本即可。
        if df_2017 is not None:
            df_2017 = plot_analysis_sample(df_2017)
        if df_2018 is not None:
            df_2018 = plot_analysis_sample(df_2018)
        release_memory()
        assert_standard_labels_only(df_combined, "02_Merged_AfterMapping")
        export_label_count_audit(df_combined, "02_Merged_AfterMapping", label_col="Label")
        record_viz_stage("02_Merged_AfterMapping", df_combined)

        # 删除重复表头行（已在读取数据时处理，这里保留作为保险）
        # df_combined = remove_duplicate_header_rows(df_combined, "combined")
        
        # ===== 阶段3：特征工程 + 无泄漏早期切分 =====
        print("\n" + "=" * 60)
        print("【阶段3：特征工程与无泄漏数据切分】")
        print("=" * 60)

        # 3.1 连续/离散分离
        print("\n[3.1] 连续/离散特征分离...")
        continuous_features, discrete_features, port_features = classify_features(df_combined, schema_only=True)
        raw_label_counts_for_viz = df_combined["Label"].value_counts().copy()
        print(f"连续特征: {len(continuous_features)}, 离散特征: {len(discrete_features)}")

        # 3.2 在任何需要fit的数据处理之前固定划分Train/Val/Test。
        print("\n[3.2] 提前划分 Train/Val/Test（修复数据泄漏）...")
        pre_split_label_counts = df_combined["Label"].value_counts()
        df_train_work, df_val_work, df_test_work = split_train_val_test_no_leakage(
            df_combined,
            train_size=config.TRAIN_TEST_SPLIT,
            val_size=config.VAL_SPLIT,
            test_size=config.TEST_SPLIT,
            random_state=config.SEED,
        )
        print(f"  原始切分: train={len(df_train_work)}, val={len(df_val_work)}, test={len(df_test_work)}")
        assert_train_class_coverage(
            source_df=df_combined,
            train_df=df_train_work,
            stage_name="03_EarlyTrainSplit",
            label_col="Label",
        )
        record_viz_stage("03_EarlyTrainSplit", df_train_work, continuous_features)
        record_viz_stage("04_EarlyValidationSplit", df_val_work, continuous_features)
        record_viz_stage("05_EarlyTestSplit", df_test_work, continuous_features)
        # 原融合特征已交给三个集合；之后只保留19类级别的覆盖检查信息。
        df_combined = pd.DataFrame({"Label":pre_split_label_counts.index.to_numpy()})
        release_memory()

        print(f"\n[3.2.1] 前置Normal下采样与来源约束（Normal:Attack={config.NORMAL_MAX_ATTACK_RATIO:g}:1，攻击全部保留）...")
        df_train_work = sample_classification_train_before_ae(df_train_work, config)
        continuous_features, discrete_features, port_features = classify_features(df_train_work)
        pre_ae_disk_precheck(df_train_work,df_val_work,df_test_work,
                            continuous_features,discrete_features,port_features)
        assert_train_class_coverage(df_combined, df_train_work, "03b_PreAESampling")
        export_label_count_audit(df_train_work, "03b_PreAESampling", label_col="Label")
        record_viz_stage("03b_PreAESampling", df_train_work, continuous_features)

        # 3.3 先按训练中位数/众数填补，再执行log1p，避免缺失被提前填零。
        print("\n[3.3] 缺失值填补 (train fit → val/test transform)...")
        df_train_work, df_val_work, df_test_work = fill_missing_train_fitted(
            df_train_work, df_val_work, df_test_work
        )
        print("\n[3.4] 对数变换 (train fit → val/test transform)...")
        df_train_work, df_val_work, df_test_work, log_shift_state = apply_log_transform_train_fitted(
            df_train_work, df_val_work, df_test_work, continuous_features
        )

        # 3.5 RobustScaler：只fit训练集
        print("\n[3.5] Robust Scaling (仅训练集fit)...")
        df_train_work, df_val_work, df_test_work, scaler = robust_scale_train_fitted(
            df_train_work, df_val_work, df_test_work, continuous_features
        )
        record_viz_stage("06_TrainRobustScaled", df_train_work, continuous_features)

        # 前置采样已完成；后续AE、IF、SMOTE全部只操作分类训练集。
        df_combined = df_train_work

        # 【修改说明】异常检测仍位于AutoEncoder之后；这里只改变数据隔离方式，不改变AE→IF→SMOTE功能顺序。

        # ===== 阶段5：仅冗余流程删除，必要采样已前移至早期训练集 =====
        # 时序预测分支仍由读取阶段旁路采集原始自然顺序数据，不受此处影响。
        # Normal下采样/来源约束已在3.2.1联合执行；不再重复执行，
        # 不恢复全局安全截断、额外类别截断及旧BorderlineSMOTE临时计算。
        print("\n" + "=" * 60)
        print("【阶段5冗余流程已删除；必要采样已在AE前完成】")
        print("=" * 60)
        df_balanced = df_combined
        print(f"前置采样后真实训练样本，共 {len(df_balanced)} 条，进入阶段6/7")
        record_viz_stage("10_BeforeFeatureFusion", df_balanced, continuous_features)

        # ===== 阶段6：降维 =====
        print("\n" + "=" * 60)
        print("【阶段6：特征选择】")
        print("=" * 60)
        
        # 【修改说明】删除KPCA降维，保留原始特征
        print(f"[6.1] 使用原始特征（不进行KPCA降维）...")
        print(f"连续特征数: {len(continuous_features)}")
        
        # ===== 阶段7：特征融合（训练集拟合，验证/测试仅变换） =====
        print("\n" + "=" * 60)
        print("【阶段7：特征融合】")
        print("=" * 60)

        # 7.1 离散特征编码：只用训练集建立编码表
        print("\n[7.1] 离散特征编码 (train fit → val/test transform)...")
        df_balanced, df_val_work, df_test_work, discrete_mapping_state = encode_discrete_train_fitted(
            df_balanced, df_val_work, df_test_work, discrete_features
        )
        record_viz_stage("11_AfterDiscreteEncoding", df_balanced, discrete_features)

        # 7.1.1 端口语义特征：频率统计只来自训练集
        print("\n[7.1.1] 端口语义特征构造 (train fit → val/test transform)...")
        df_balanced, df_val_work, df_test_work = port_features_train_fitted(
            df_balanced, df_val_work, df_test_work, port_features
        )
        record_viz_stage("12_AfterPortFeatureEngineering", df_balanced, port_features)

        # 7.2 拼接得到X_final
        print("\n[7.2] 特征拼接...")
        final_features = [col for col in df_balanced.columns
                          if col not in ["Label", "dataset_source"]
                          and pd.api.types.is_numeric_dtype(df_balanced[col])]
        for frame in (df_val_work, df_test_work):
            for col in final_features:
                if col not in frame.columns:
                    frame[col] = 0.0
        print(f"最终特征数: {len(final_features)}")

        # 7.3 AutoEncoder：仅训练集训练；val/test复用训练好的模型和scaler
        print("\n[7.3] AutoEncoder特征编码（仅训练集训练）...")
        ae_input_features = [
            c for c in continuous_features
            if c in df_balanced.columns and pd.api.types.is_numeric_dtype(df_balanced[c])
        ]
        side_features = [c for c in final_features if c not in ae_input_features]
        if not ae_input_features:
            raise ValueError("没有连续AE特征，请核实字段定义，不能将端口/类别当连续输入")

        for frame in (df_val_work, df_test_work):
            for col in ae_input_features + side_features:
                if col not in frame.columns:
                    frame[col] = 0.0

        X_train_ae = dataframe_float32_matrix(df_balanced,ae_input_features)
        print(f"  AE训练输入形状: {X_train_ae.shape}")

        ae_latent_dim = min(
            int(getattr(config, "AE_LATENT_MAX", 32)),
            max(8, len(ae_input_features) // 2)
        )
        Z_train, used_indices, ae_model, ae_scaler = apply_autoencoder(
            X_train_ae,
            latent_dim=ae_latent_dim,
            epochs=int(getattr(config, "AE_EPOCHS", 20)),
            batch_size=int(getattr(config, "AE_BATCH_SIZE", 4096)),
            labels=None,
            normal_oversample_ratio=0.0,
            return_artifacts=True,
        )
        Z_val = transform_frame_with_fitted_autoencoder(
            df_val_work,ae_input_features,ae_model,ae_scaler,batch_size=int(config.AE_BATCH_SIZE))
        Z_test = transform_frame_with_fitted_autoencoder(
            df_test_work,ae_input_features,ae_model,ae_scaler,batch_size=int(config.AE_BATCH_SIZE))
        # 分类AE推理已结束，后续图表仅使用Z，不再持有其模型显存。
        save_classification_artifacts(ae_model, ae_scaler, scaler, ae_input_features, side_features)
        del ae_model, ae_scaler
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print(f"  AE编码后: train={Z_train.shape}, val={Z_val.shape}, test={Z_test.shape}")
        # 兼容原有可视化变量名，不改变后续绘图功能。
        Z = Z_train

        # 保持原有used_indices同步逻辑（当前labels=None时通常为全量索引）。
        before_ae_sync_attack_labels = get_present_attack_labels(df_balanced, label_col="Label")
        df_balanced = df_balanced.iloc[used_indices].reset_index(drop=True)
        assert_no_class_disappears(None, df_balanced, "13_AfterAutoEncoderSync", label_col="Label", before_attack_labels=before_ae_sync_attack_labels)
        export_label_count_audit(df_balanced, "13_AfterAutoEncoderSync", label_col="Label")

        ae_feature_cols = [f"AE_{i}" for i in range(Z_train.shape[1])]

        def _rebuild_ae_frame(base_df: pd.DataFrame, Z_arr: np.ndarray) -> pd.DataFrame:
            ae_df = pd.DataFrame(Z_arr, columns=ae_feature_cols)
            keep_cols = side_features + ["Label"]
            if "dataset_source" in base_df.columns:
                keep_cols.append("dataset_source")
            keep_cols = [c for c in keep_cols if c in base_df.columns]
            side_df = base_df[keep_cols].reset_index(drop=True).copy()
            return pd.concat([ae_df.reset_index(drop=True), side_df], axis=1)

        df_balanced = _rebuild_ae_frame(df_balanced, Z_train)
        df_val_processed = _rebuild_ae_frame(df_val_work, Z_val)
        df_test_processed = _rebuild_ae_frame(df_test_work, Z_test)
        del X_train_ae, df_val_work, df_test_work, Z_val, Z_test, used_indices

        final_features = [col for col in df_balanced.columns
                          if col not in ["Label", "dataset_source"]
                          and pd.api.types.is_numeric_dtype(df_balanced[col])]
        print(f"  AE特征数: {len(ae_feature_cols)}，总特征数: {len(final_features)}")
        record_viz_stage("13_AfterAutoEncoderFeatureReplacement", df_balanced, final_features)
        # 此后原连续工作表仅用于PCA图；保存同一批行的小样本后释放整张大表。
        ae_viz_indices = stratified_plot_indices(df_balanced.Label.to_numpy(),config.PLOT_SAMPLE_SIZE,config.SEED)
        df_ae_input_for_viz = df_combined.iloc[ae_viz_indices][ae_input_features].copy()
        Z_for_input_viz = Z_train[ae_viz_indices,:-1].copy()
        ae_labels_for_input_viz = df_balanced.Label.iloc[ae_viz_indices].to_numpy(copy=True)
        df_combined = df_combined[["Label"]].copy()
        df_train_work = df_combined  # 后续仅用于训练标签覆盖断言，不再持有原连续特征。

        # 7.4 IsolationForest：仅训练集fit并过滤；val/test保持真实样本
        print("\n[7.4] 在训练集AE编码空间进行IsolationForest异常检测...")
        contamination = float(getattr(config, "IF_CONTAMINATION", 0.01))
        scaler_ae = StandardScaler()
        Z_train_scaled = scaler_ae.fit_transform(Z_train)
        # 兼容原有IF可视化变量名。
        Z_scaled = Z_train_scaled

        label_counts_if = df_balanced["Label"].value_counts()
        protected_labels = {l for l in label_counts_if.index if label_counts_if[l] < 500}
        if protected_labels:
            print(f"  【极小类保护】以下类别(<500样本)跳过IF: {protected_labels}")

        pre_if_labels = set(df_balanced["Label"].unique())
        pre_if_attack_labels = get_present_attack_labels(df_balanced, label_col="Label")
        if_class_backups = df_balanced.drop_duplicates(subset=["Label"]).copy()
        df_balanced_with_ae = df_balanced[["Label"]].copy()

        non_protected_mask = ~df_balanced["Label"].isin(protected_labels) if protected_labels else np.ones(len(df_balanced), dtype=bool)
        outlier_labels = np.ones(len(df_balanced), dtype=int)
        if int(np.sum(non_protected_mask)) >= 50:
            iso_forest = IsolationForest(
                contamination=contamination,
                n_estimators=100,
                max_samples='auto',
                random_state=config.SEED,
                n_jobs=-1,
            )
            sub_outliers = iso_forest.fit_predict(Z_train_scaled[non_protected_mask])
            outlier_labels[np.asarray(non_protected_mask)] = sub_outliers
        else:
            print("  参与IF的训练样本不足50，跳过IF异常过滤")

        n_outliers = int((outlier_labels == -1).sum())
        n_normal = int((outlier_labels == 1).sum())
        print(f"  IsolationForest结果: 正常={n_normal}, 异常={n_outliers}")

        normal_mask = outlier_labels == 1
        df_balanced = df_balanced[normal_mask].reset_index(drop=True)
        # 原df_after_if_for_viz没有消费者，不再复制整张IF后数据表。

        post_if_labels = set(df_balanced["Label"].unique())
        lost_labels = pre_if_labels - post_if_labels
        if lost_labels:
            print(f"  [警告] IF过滤导致类别消失: {lost_labels}，从训练集IF前数据补回1条...")
            for lost_label in lost_labels:
                lost_mask = if_class_backups["Label"] == lost_label
                if lost_mask.any():
                    df_balanced = pd.concat([df_balanced, if_class_backups[lost_mask].head(1)], ignore_index=True)

        assert_no_class_disappears(None, df_balanced, "14_AfterIsolationForest", label_col="Label", before_attack_labels=pre_if_attack_labels)
        # 类别补回完成后，后续AE/IF图只需要IF前标签，而不是完整的特征表。
        del if_class_backups
        export_label_count_audit(df_balanced, "14_AfterIsolationForest", label_col="Label")
        final_features = [col for col in df_balanced.columns
                          if col not in ["Label", "dataset_source"]
                          and pd.api.types.is_numeric_dtype(df_balanced[col])]
        record_viz_stage("14_AfterIsolationForest", df_balanced, final_features)

        # 7.5 ASMOTE：仅训练集执行
        print("\n" + "=" * 60)
        print("【阶段7.5：SMOTE增强（仅训练集）】")
        print("=" * 60)
        if len(df_balanced) > 0 and final_features:
            pre_final_smote_attack_labels = get_present_attack_labels(df_balanced, label_col="Label")
            df_balanced = balance_data(df_balanced, final_features)
            assert_no_class_disappears(None, df_balanced, "15_AfterSMOTEFinalBalanced", label_col="Label", before_attack_labels=pre_final_smote_attack_labels)
            export_label_count_audit(df_balanced, "15_AfterSMOTEFinalBalanced", label_col="Label")
            assert_train_class_coverage(
                source_df=df_train_work,
                train_df=df_balanced,
                stage_name="15_AfterSMOTEFinalBalanced",
                label_col="Label",
            )
            final_features = [col for col in df_balanced.columns
                              if col not in ["Label", "dataset_source"]
                              and pd.api.types.is_numeric_dtype(df_balanced[col])]
            record_viz_stage("15_AfterSMOTEFinalBalanced", df_balanced, final_features)

        # ===== 阶段8：使用早期固定切分，不再二次随机切分 =====
        print("\n" + "=" * 60)
        print("【阶段8：固定Train/Val/Test输出】")
        print("=" * 60)
        df_train = enforce_standard_labels(df_balanced, stage_name="before_save_train_split", label_col="Label", copy=False)
        df_val = enforce_standard_labels(df_val_processed, stage_name="before_save_val_split", label_col="Label", raise_if_empty=False, copy=False)
        df_test = enforce_standard_labels(df_test_processed, stage_name="before_save_test_split", label_col="Label", raise_if_empty=False, copy=False)

        # 对齐列结构，确保RF/XGBoost/Informer读取接口保持不变。
        ordered_cols = final_features + (["dataset_source"] if "dataset_source" in df_train.columns else []) + ["Label"]
        for frame in (df_val, df_test):
            for col in final_features:
                if col not in frame.columns:
                    frame[col] = 0.0
            if "dataset_source" in df_train.columns and "dataset_source" not in frame.columns:
                frame["dataset_source"] = "Unknown"
        df_train = df_train[[c for c in ordered_cols if c in df_train.columns]].reset_index(drop=True)
        df_val = df_val[[c for c in ordered_cols if c in df_val.columns]].reset_index(drop=True)
        df_test = df_test[[c for c in ordered_cols if c in df_test.columns]].reset_index(drop=True)

        assert_standard_labels_only(df_train, "before_save_train_split")
        assert_standard_labels_only(df_val, "before_save_val_split")
        assert_standard_labels_only(df_test, "before_save_test_split")
        print(f"  最终训练集（IF+ASMOTE后）: {len(df_train)}")
        print(f"  验证集（真实样本，仅transform）: {len(df_val)}")
        print(f"  测试集（真实样本，仅transform）: {len(df_test)}")
        record_viz_stage("16_TrainSplit", df_train, final_features)
        record_viz_stage("17_ValidationSplit", df_val, final_features)
        record_viz_stage("18_TestSplit", df_test, final_features)

        # 四阶段类别分布图在CSV保存后的原最终调用处生成一次，避免同路径重复覆盖。

        # ===== 保存结果 =====

        print("\n[INFO] 保存处理后的CSV")

        # 保存CSV格式输出：保存前最后兜底，确保CSV中只有配置中的标准类别标签；每个CSV只写入本次时间戳目录一次
        df_train = enforce_standard_labels(df_train, stage_name="final_backup_train", label_col="Label", raise_if_empty=False, copy=False)
        df_val = enforce_standard_labels(df_val, stage_name="final_backup_val", label_col="Label", raise_if_empty=False, copy=False)
        df_test = enforce_standard_labels(df_test, stage_name="final_backup_test", label_col="Label", raise_if_empty=False, copy=False)
        assert_standard_labels_only(df_train, "final_backup_train")
        assert_standard_labels_only(df_val, "final_backup_val")
        assert_standard_labels_only(df_test, "final_backup_test")
        record_run_status("running",phase="saving_classification")
        additional_frames=[df_train,df_val,df_test]
        if not getattr(config,"_hardlink_supported",False):additional_frames.append(df_train)
        ensure_disk_capacity(config.OUTPUT_DIR,estimated_csv_bytes(additional_frames))
        atomic_csv(df_train,config.OUTPUT_TRAIN,encoding="utf-8")
        atomic_csv(df_val,config.OUTPUT_VAL,encoding="utf-8")
        atomic_csv(df_test,config.OUTPUT_TEST,encoding="utf-8")
        print(f"\nCSV格式保存到:")
        print(f"  训练集: {config.OUTPUT_TRAIN}")
        print(f"  验证集: {config.OUTPUT_VAL}")
        print(f"  测试集: {config.OUTPUT_TEST}")
        
        # balanced_dataset与最终训练集相同；不是训练/验证/测试合并数据。
        # 只保留一个 balanced_dataset.csv，不再额外保存 balanced_dataset_时间戳.csv。
        # 本次运行目录本身已经带时间戳：config.OUTPUT_DIR = run_YYYYMMDD_HHMMSS。
        df_balanced = enforce_standard_labels(df_balanced, stage_name="before_save_balanced_dataset", label_col="Label", copy=False)
        assert_standard_labels_only(df_balanced, "before_save_balanced_dataset")
        df_balanced_save = df_train  # 仅审计/保存别名，不复制完整训练特征矩阵。

        # 相同内容使用硬链接节省磁盘；跨文件系统/不支持硬链接时回退原子复制。
        save_balanced_training_alias(config.OUTPUT_TRAIN,config.OUTPUT_BALANCED)
        print(f"  类别增强训练集副本: {config.OUTPUT_BALANCED}")
        print(f"\n  【重要】类别增强训练集标签分布（不含验证/测试）:")
        print(df_balanced_save["Label"].value_counts())
        observed_labels = sorted(df_balanced_save["Label"].dropna().astype(str).unique().tolist())
        print(f"\n  【标签强约束校验】最终CSV标签数: {len(observed_labels)} / {len(config.STANDARD_LABELS)}")
        print(f"  最终CSV标签集合: {observed_labels}")
        missing_standard = sorted(set(config.STANDARD_LABELS) - set(observed_labels))
        if missing_standard:
            print(f"  未出现的标准标签: {missing_standard}")
        assert_standard_labels_only(df_balanced_save, "final_balanced_csv")
        if "dataset_source" in df_balanced_save.columns:
            print(f"\n  【重要】完整数据集来源分布:")
            final_source_counts = df_balanced_save["dataset_source"].astype(str).value_counts()
            print(final_source_counts)
            print("  来源占比(%):", (100 * final_source_counts / max(len(df_balanced_save), 1)).round(2).to_dict())
            # 只审计最终比例，不在IF/ASMOTE之后再次删Normal或攻击来凑比例。
            final_normal = int(df_balanced_save["Label"].eq("Normal").sum())
            final_attack = len(df_balanced_save) - final_normal
            real_source_counts = df_balanced_save.loc[df_balanced_save.dataset_source.ne("synthetic"),"dataset_source"].value_counts()
            final_2017_share = int(real_source_counts.get("2017", 0)) / max(int(real_source_counts.sum()), 1)
            final_source_goal_met = (
                abs(final_2017_share - config.SOURCE_BALANCE_TARGET) <= config.SOURCE_BALANCE_TOLERANCE + 1e-12
                if set(real_source_counts.index) == {"2017", "2018"} else None
            )
            final_summary = {
                "stage": "final_classification_train_after_IF_ASMOTE",
                "source_counts": final_source_counts.to_dict(),
                "source_2017_share": final_2017_share,
                "source_goal_met": final_source_goal_met,
                "normal": final_normal, "attack": final_attack,
                "normal_to_attack": final_normal / final_attack if final_attack else None,
                "post_asmote_downsampling": False,
                "real_source_counts":df_balanced_save.loc[df_balanced_save.dataset_source.ne("synthetic"),"dataset_source"].value_counts().to_dict(),
                "synthetic_count":int(df_balanced_save.dataset_source.eq("synthetic").sum()),
                "note": f"{config.NORMAL_MAX_ATTACK_RATIO:g}:1及来源配额在AE前执行；IF/ASMOTE可能改变最终比例，末尾只审计不采样",
            }
            try:
                write_json(os.path.join(config.OUTPUT_DIR, "balance_debug_logs", "final_train_source_summary.json"),
                               final_summary)
            except Exception as exc:
                logger.warning("保存最终训练集来源审计失败: %s", exc)
            if final_source_goal_met is False:
                logger.warning("IF/ASMOTE后来源占比发生变化；保留真实统计，不执行事后重复采样。详见final_train_source_summary.json")
        
        save_latest_outputs(config,df_train,df_val,df_test,df_balanced_save)
        print(f"\nLatest运行摘要已写入: {config.LATEST_SUMMARY}")
        print(f"数据文件未复制到主输出目录根层，均只保存在本次运行目录: {config.OUTPUT_DIR}")
        del df_balanced_save

        # 新增：在分类CSV保存后，用同一个冻结的融合处理器重放自然顺序记录。
        if getattr(config, "EXPORT_FUSED_FORECAST", False):
            export_fused_forecast(config)
        release_memory()

        # 【新增】绘制类别分布四阶段对比图（备用路径）
        raw_counts = raw_label_counts_for_viz
        smote_counts = df_balanced["Label"].value_counts()
        train_counts = df_train["Label"].value_counts() if len(df_train) > 0 else pd.Series()
        test_counts = df_test["Label"].value_counts() if len(df_test) > 0 else pd.Series()
        plot_class_distribution_comparison(
            raw_counts, smote_counts, train_counts, test_counts,
            config.OUTPUT_VISUALIZATION, logger
        )
        
        # 可视化
        plot_label_distribution(df_combined, "处理后标签分布", 
                               os.path.join(config.OUTPUT_VISUALIZATION, "1_标签分布.png"))
        plot_label_distribution(df_balanced, "平衡后标签分布", 
                               os.path.join(config.OUTPUT_VISUALIZATION, "1_2_平衡后标签分布.png"))
        
        if "dataset_source" in df_balanced.columns:
            plot_dataset_source_distribution(df_balanced, 
                                            os.path.join(config.OUTPUT_VISUALIZATION, "2_数据源分布.png"))
            plot_label_by_source(df_balanced, 
                                os.path.join(config.OUTPUT_VISUALIZATION, "2_2_攻击类型按来源分布.png"))
        
        # 获取可用特征（【修改说明】使用AE特征进行可视化）
        # 获取AE特征列（AE_0 ~ AE_15）
        ae_feature_cols = [col for col in df_balanced.columns if col.startswith("AE_")]
        # 获取其他数值特征（离散+端口特征）
        other_feature_cols = [col for col in df_balanced.columns 
                             if col not in ["Label", "dataset_source"] 
                             and not col.startswith("AE_")
                             and pd.api.types.is_numeric_dtype(df_balanced[col])]
        
        # AE特征可视化
        vis_features = ae_feature_cols[:16]  # 使用AE特征
        
        logger.info(f"用于可视化的AE特征数量: {len(vis_features)}")
        
        # 绘制AE特征分布图
        if len(vis_features) >= 4:
            plot_feature_distributions_multiple(plot_analysis_sample(df_balanced), vis_features, config.OUTPUT_VISUALIZATION)
            print(f"  AE特征分布图已生成")
        
        # AE特征相关性热力图
        if len(vis_features) >= 4:
            plot_correlation_heatmap(plot_analysis_sample(df_balanced), vis_features, "AE特征相关性热力图（抽样）", 
                                    os.path.join(config.OUTPUT_VISUALIZATION, "AE相关性热力图.png"))
            print(f"  AE特征相关性热力图已生成")
        
        # 【修改说明】使用AE特征绘制散点图（用AE_0和AE_1作为坐标）
        if len(ae_feature_cols) >= 2:
            vis_df = plot_analysis_sample(df_balanced)[[ae_feature_cols[0], ae_feature_cols[1], "Label"]].copy()
            plot_ae_2d_scatter(
                vis_df,
                ae_feature_cols[0],
                ae_feature_cols[1],
                os.path.join(config.OUTPUT_VISUALIZATION, "AE_2D散点图.png")
            )
            print(f"  AE特征2D散点图已生成")
        
        if df_train is not None and len(df_train) > 0:
            plot_label_distribution(df_train, "训练集标签分布", 
                                   os.path.join(config.OUTPUT_VISUALIZATION, "6_训练集标签分布.png"))
        if df_test is not None and len(df_test) > 0:
            plot_label_distribution(df_test, "测试集标签分布", 
                                   os.path.join(config.OUTPUT_VISUALIZATION, "7_测试集标签分布.png"))
        
        # ============ 新增可视化 ============
        print("\n[可视化] 开始生成可视化图表...")
        
        # 确保输出目录存在
        os.makedirs(config.OUTPUT_VISUALIZATION, exist_ok=True)
        print(f"[可视化] 输出目录: {config.OUTPUT_VISUALIZATION}")
        
        # 1. 数据清洗前后标签分布对比图
        try:
            print("[可视化1] 检查数据...")
            if 'df_combined' in dir() and 'df_balanced' in dir():
                print(f"[可视化1] 数据检查: df_combined={len(df_combined)}行, df_balanced={len(df_balanced)}行")
                plot_label_distribution_comparison(
                    raw_label_counts_for_viz,
                    df_balanced["Label"].value_counts(),
                    "融合原始全量与最终类别增强训练集分布（不同样本范围）",
                    os.path.join(config.OUTPUT_VISUALIZATION, "8_原始全量与最终训练集分布对比.png"))
            else:
                print("[可视化1] 跳过: 数据不存在")
        except Exception as e:
            import traceback
            logger.warning(f"数据清洗前后标签分布对比图生成失败: {e}")
            traceback.print_exc()
        
        # 2. AutoEncoder降维前后PCA/t-SNE可视化
        try:
            print("\n[可视化2] 检查AutoEncoder数据...")
            if 'df_combined' in dir() and 'Z' in dir():
                original_features = list(df_ae_input_for_viz.columns)
                print(f"[可视化2] 原始特征数: {len(original_features)}")
                print(f"[可视化2] Z形状: {Z.shape if 'Z' in dir() else 'N/A'}")
                if len(original_features) > 0 and 'Z' in dir():
                    print("[可视化2] 开始PCA/t-SNE降维...")
                    plot_ae_pca_tsne(
                        df_ae_input_for_viz,
                        Z_for_input_viz,
                        ae_labels_for_input_viz,
                        os.path.join(config.OUTPUT_VISUALIZATION, "9_AutoEncoder降维可视化.png"))
            else:
                print("[可视化2] 跳过: df_combined或Z不存在")
        except Exception as e:
            import traceback
            logger.warning(f"AutoEncoder降维可视化生成失败: {e}")
            traceback.print_exc()
        
        # 2.5 AE latent空间t-SNE可视化（仅使用latent特征，不参与训练）
        try:
            print("\n[可视化2.5] 检查AE latent特征数据...")
            if 'Z' in dir() and ('df_balanced_with_ae' in dir() or 'df_balanced' in dir()):
                ae_labels_for_viz = df_balanced_with_ae["Label"].values if 'df_balanced_with_ae' in dir() else df_balanced["Label"].values
                print(f"[可视化2.5] Z形状: {Z.shape}")
                print(f"[可视化2.5] 标签数量: {len(ae_labels_for_viz)}")
                print("[可视化2.5] 开始t-SNE降维...")
                plot_ae_latent_tsne(
                    Z[:, :-1],
                    ae_labels_for_viz,
                    os.path.join(config.OUTPUT_VISUALIZATION, "9b_AE_latent_tSNE.png"))
            else:
                print("[可视化2.5] 跳过: Z或df_balanced不存在")
        except Exception as e:
            import traceback
            logger.warning(f"AE latent t-SNE可视化生成失败: {e}")
            traceback.print_exc()
        
        # 3. Isolation Forest异常点分布图
        try:
            print("\n[可视化3] 检查Isolation Forest数据...")
            if 'Z_scaled' in dir() and 'outlier_labels' in dir() and ('df_balanced_with_ae' in dir() or 'df_balanced' in dir()):
                if_labels_for_viz = df_balanced_with_ae["Label"].values if 'df_balanced_with_ae' in dir() else df_balanced["Label"].values
                print(f"[可视化3] Z_scaled形状: {Z_scaled.shape}")
                print(f"[可视化3] outlier_labels形状: {outlier_labels.shape}")
                normal_count = sum(outlier_labels == 1)
                anomaly_count = sum(outlier_labels == -1)
                print(f"[可视化3] 正常样本: {normal_count}, 异常样本: {anomaly_count}")
                print("[可视化3] 开始绘制异常分布图...")
                plot_if_anomaly_distribution(
                    Z_scaled,
                    outlier_labels,
                    if_labels_for_viz,
                    os.path.join(config.OUTPUT_VISUALIZATION, "10_IF异常分布.png"))
            else:
                print("[可视化3] 跳过: Z_scaled或outlier_labels不存在")
        except Exception as e:
            import traceback
            logger.warning(f"Isolation Forest异常分布图生成失败: {e}")
            traceback.print_exc()
        
        # 4. 2017/2018数据对齐PCA分布图
        try:
            print("\n[可视化4] 检查数据集对齐数据...")
            if 'df_2017' in dir() and 'df_2018' in dir() and df_2017 is not None and df_2018 is not None:
                common_viz_cols = []
                for col in df_2017.columns:
                    if col in ["Label", "dataset_source"]:
                        continue
                    if col not in df_2018.columns:
                        continue
                    try:
                        if df_2017[col].dtype == 'object':
                            s17 = pd.to_numeric(df_2017[col], errors='coerce')
                        else:
                            s17 = df_2017[col]
                        if df_2018[col].dtype == 'object':
                            s18 = pd.to_numeric(df_2018[col], errors='coerce')
                        else:
                            s18 = df_2018[col]
                        if s17.dropna().shape[0] > 0 and s18.dropna().shape[0] > 0:
                            common_viz_cols.append(col)
                    except Exception:
                        pass
                if len(common_viz_cols) >= 2:
                    print(f"[可视化4] 2017数据集: {len(df_2017)}行")
                    print(f"[可视化4] 2018数据集: {len(df_2018)}行")
                    print(f"[可视化4] 使用共同特征数: {min(20, len(common_viz_cols))}")
                    print("[可视化4] 开始KDE对齐可视化...")
                    plot_dataset_alignment_kde(
                        df_2017,
                        df_2018,
                        common_viz_cols[:min(20, len(common_viz_cols))],
                        os.path.join(config.OUTPUT_VISUALIZATION, "11_数据集对齐KDE.png"))
                else:
                    print(f"[可视化4] 跳过: 共同数值特征不足({len(common_viz_cols)}个)，df_2017列={list(df_2017.columns)[:10]}...")
            else:
                print("[可视化4] 跳过: df_2017或df_2018不存在")
        except Exception as e:
            import traceback
            logger.warning(f"数据集对齐PCA图生成失败: {e}")
            traceback.print_exc()
        
        print("\n=== 数据处理完成 ===")
        print(f"可视化图表已保存到: {config.OUTPUT_VISUALIZATION}")

        if stages_data:
            print("\n[可视化] 生成分布距离变化曲线图...")
            plot_distribution_distance_curve(
                stages_data,
                os.path.join(config.OUTPUT_VISUALIZATION, "12_数据分布距离变化曲线图.png"))
            print(f"  各阶段距离汇总: {stages_data}")

        print("\n[综合可视化] 生成完整处理流程前后对比、比例与阶段变化报告...")
        generate_comprehensive_visualization_report(
            viz_stage_snapshots,
            config.OUTPUT_VISUALIZATION,
            df_train=df_train if 'df_train' in dir() else None,
            df_val=df_val if 'df_val' in dir() else None,
            df_test=df_test if 'df_test' in dir() else None,


        )

        record_run_status("completed",phase="all_outputs_saved")

    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"执行失败: {e}")
        logger.error(f"主流程执行失败: {e}", exc_info=True)
        try:
            record_run_status("failed", error_type=type(e).__name__, error=str(e))
        except Exception:
            logger.error("失败状态无法落盘（例如磁盘已满）；保留原始异常")
        cleanup_forecast_spools(config)
        raise
    finally:
        stop_memory_monitor(config)
        save_performance_report(config)


# ================= 新增：主融合特征的顺序、统一划分及预测导出 =================
def forecast_attach_row_identity(frame, source, filename, cfg):
    """只依据原CSV有效行顺序建身份；完全忽略时间戳及其取值。"""
    if "flow_uid" in frame:
        raise ValueError("原始输入包含保留字段flow_uid")
    file_key = hashlib.sha256((str(source)+"/"+str(filename)).encode()).hexdigest()[:20]
    uids = np.asarray([file_key+":"+str(i) for i in range(len(frame))],dtype=object)
    eligible = map_traffic_label_series(frame["Label"]).isin(cfg.STANDARD_LABELS).to_numpy()
    rank = np.full(len(frame),-1,dtype=np.int64)
    rank[eligible] = np.arange(int(eligible.sum()),dtype=np.int64)
    metadata = pd.DataFrame({"flow_uid":uids,"sequence_id":str(source)+"/"+str(filename),
                             "order_index":rank,"eligible":eligible})
    metadata["sequence_id"] = metadata.sequence_id.astype("category")
    cfg._forecast_row_parts.append(metadata)
    out = frame.copy(deep=False)
    out["flow_uid"] = uids
    return out


def forecast_require_row_identity(frame, cfg, stage):
    """在读取/融合边界快速检查UID传递；不从已经打乱的记录重建顺序。"""
    if not (getattr(cfg, "EXPORT_FUSED_FORECAST", False) or getattr(cfg, "FORECAST_UNIFIED_SPLIT", False)):
        return
    if "flow_uid" not in frame:
        renamed = "；发现被改名的flow uid字段" if "flow uid" in frame else ""
        raise ValueError(f"[{stage}] 缺少flow_uid关联标识{renamed}；请检查读取/列名标准化链路，不能按当前行号补造原始顺序")
    if not getattr(cfg, "_forecast_row_parts", None):
        raise ValueError(f"[{stage}] 缺少读取阶段UID/顺序旁表；请保持同一个Config实例并启用EXPORT_FUSED_FORECAST")


def forecast_plan_class_coverage(identity, labels, cfg, train_size, val_size, test_size):
    """对所有文件同时规划整数边界；用约束求解替代不可回退的局部贪心。"""
    from scipy.optimize import milp, Bounds, LinearConstraint
    from scipy.sparse import coo_matrix
    labels = np.asarray(labels)
    if len(labels)!=len(identity): raise ValueError("标签与顺序旁表长度不一致")
    k,need = int(cfg.FORECAST_ROWS_PER_WINDOW),int(cfg.FORECAST_SEQ_LEN+cfg.FORECAST_PRED_LEN)
    required = set(getattr(cfg,"REQUIRED_BASE_LABELS",cfg.STANDARD_LABELS))|set(labels)
    present = set(labels)
    if required-present: raise ValueError(f"有效输入本身缺类：{sorted(required-present)}")
    plans=[]
    for name,ids in identity.groupby("sequence_id",sort=True,observed=True).indices.items():
        ids=np.asarray(ids,dtype=np.int64)
        order=identity.order_index.to_numpy()[ids]
        n=len(ids);complete,tail=divmod(n,k)
        first={}
        for label in np.unique(labels[ids]):
            first[str(label)] = int(order[labels[ids]==label].min())//k
        t=min(complete,max(1,int(complete*train_size))) if complete else 0
        v=min(complete-t,max(0,int(complete*val_size)))
        plans.append({"name":str(name),"ids":ids,"rows":n,"complete":complete,"tail":tail,
                      "first":first,"initial":(t,v,complete-t-v,0)})
    initial_covered={label for p in plans for label,w in p["first"].items() if w<p["initial"][0]}
    nvar=0;lower=[];upper=[];integrality=[];objective=[]
    def variable(lb,ub,integer=True,cost=0.):
        nonlocal nvar
        index=nvar;nvar+=1;lower.append(lb);upper.append(ub);integrality.append(int(integer));objective.append(cost)
        return index
    rows=[];cols=[];values=[];lb=[];ub=[]
    def constraint(terms,lo=-np.inf,hi=np.inf):
        r=len(lb);lb.append(lo);ub.append(hi)
        for col,value in terms.items():
            if value: rows.append(r);cols.append(col);values.append(value)
    cover={label:[] for label in required};indicators={s:[] for s in ("train","val","test")};groups=[]
    global_rows={s:{} for s in indicators}
    fractions=(train_size,val_size,test_size)
    total=len(identity)
    min_groups=next(g for g in range(2,100) if g-int(np.ceil(g/min(cfg.FORECAST_BRIDGE_OOF_FOLDS,g)))>=3)
    for p in plans:
        c,tail=p["complete"],p["tail"]
        t,v,e=[variable(0,c) for _ in range(3)]
        b=variable(0,1 if tail else 0)
        p["variables"]=(t,v,e,b)
        constraint({t:1,v:1,e:1},c,c)
        constraint({t:1,b:-c},0)
        # 非空文件至少保留一个训练完整窗；不足一窗时全部作为分类训练尾部。
        if c: constraint({t:1},1)
        else: constraint({b:1},1,1)
        for split,x in zip(indicators,(t,v,e)):
            z=variable(0,int(c>=need));constraint({x:1,z:-need},0)
            indicators[split].append(z)
            global_rows[split][x]=k
        global_rows["train"][b]=tail;global_rows["test"][b]=-tail
        g=variable(0,int(np.ceil(c/need)))
        constraint({t:1,g:-need},1-need);groups.append(g)
        for label,w in p["first"].items():
            z=variable(0,1);cover[label].append(z)
            if w<c: constraint({t:1,z:-(w+1)},0)
            else: constraint({b:1,z:-1},0)
        # 文件内比例偏差是次要目标；总体比例偏差权重更高。
        for x,f in zip((t,v,e),fractions):
            d=variable(0,np.inf,False,cost=.05/max(1,sum(p2['complete'] for p2 in plans)))
            constraint({x:1,d:-1},hi=f*c);constraint({x:-1,d:-1},hi=-f*c)
    for indices in indicators.values(): constraint({i:1 for i in indices},1)
    constraint({i:1 for i in groups},min_groups)
    for indices in cover.values(): constraint({i:1 for i in indices},1)
    tail_total=sum(p["tail"] for p in plans)
    for split,f in zip(indicators,fractions):
        target=f*total-(tail_total if split=="test" else 0)
        d=variable(0,np.inf,False,cost=1/max(1,total))
        terms=dict(global_rows[split]);terms[d]=-1;constraint(terms,hi=target)
        terms={x:-value for x,value in global_rows[split].items()};terms[d]=-1;constraint(terms,hi=-target)
    matrix=coo_matrix((values,(rows,cols)),shape=(len(lb),nvar)).tocsc()
    result=milp(np.asarray(objective),integrality=np.asarray(integrality),
        bounds=Bounds(lower,upper),constraints=LinearConstraint(matrix,lb,ub),
        options={"time_limit":float(cfg.FORECAST_SPLIT_SOLVER_SECONDS),"mip_rel_gap":.001})
    if result.x is None:
        kind="约束下不存在可行划分" if result.status==2 else "划分搜索尚未找到可行方案，不能判定数据不可行"
        raise ValueError(f"{kind}：训练全类、独立验证/测试连续段至少{need}窗、桥接分组至少{min_groups}组；求解器：{result.message}")
    assignment=np.empty(len(identity),dtype=np.int8);bounds=[];changes=[]
    lengths={s:[] for s in indicators}
    for p in plans:
        t,v,e,b=[int(round(result.x[x])) for x in p["variables"]]
        if min(t,v,e,b)<0 or t+v+e!=p["complete"] or (b and t!=p["complete"]):
            raise RuntimeError("划分求解器返回的整数边界未通过复核")
        ids=p["ids"];position=identity.order_index.to_numpy()[ids]
        codes=np.where(position<t*k,0,np.where(position<(t+v)*k,1,2)).astype(np.int8)
        if b: codes[:]=0
        assignment[ids]=codes
        for s,n in zip(indicators,(t,v,e)): lengths[s].append(n)
        item={"sequence_id":p["name"],"train_complete_windows":t,"val_complete_windows":v,
              "test_complete_windows":e,"training_only":bool(v==0 and e==0 and b==bool(p["tail"])),
              "tail_rows":p["tail"],"tail_split":"train" if b else "test",
              "train_end_order_exclusive":p["rows"] if b else t*k,"test_start_order":(t+v)*k}
        bounds.append(item)
        if (t,v,e,b)!=p["initial"]: changes.append(item)
    identity["split"]=pd.Categorical.from_codes(assignment,["train","val","test"])
    covered=set(labels[assignment==0])
    if required-covered: raise RuntimeError(f"划分后训练集仍缺类：{sorted(required-covered)}")
    readiness=forecast_sequence_precheck(lengths,cfg)
    counts={s:pd.Series(labels[assignment==i]).value_counts().to_dict() for i,s in enumerate(indicators)}
    final_rows={s:int((assignment==i).sum()) for i,s in enumerate(indicators)}
    coverage={"policy":"global_ordered_integer_planning","required_labels":sorted(required),
        "initial_train_missing_labels":sorted(required-initial_covered),"final_train_missing_labels":[],
        "adjustments":changes,"requested_ratios":dict(zip(indicators,fractions)),"final_rows":final_rows,
        "actual_ratios":{s:n/total for s,n in final_rows.items()},"training_only_sequences":[b['sequence_id'] for b in bounds if b['training_only']],
        "label_counts_by_split":counts,"validation_missing_labels":sorted(required-set(counts['val'])),
        "test_missing_labels":sorted(required-set(counts['test'])),"split_selected_using_labels":True,
        "solver_status":int(result.status),"solver_message":str(result.message),
        "readiness":readiness,"selection_unit":"whole 500-flow windows within original CSV row order",
        "evaluation_scope":"within-capture offline later-flow-window evaluation; freeze split before testing"}
    cfg._forecast_precheck=readiness
    return bounds,coverage


def forecast_shared_split(frame, train_size, val_size, test_size, cfg):
    """先规划类别覆盖的顺序划分并审计，再写大规模临时数据；所有fit共用此划分。"""
    if min(train_size,val_size,test_size)<=0 or abs(train_size+val_size+test_size-1)>1e-6:
        raise ValueError("统一划分比例无效")
    forecast_require_row_identity(frame,cfg,"统一划分前")
    valid = enforce_standard_labels(frame,stage_name="before_shared_order_split",copy=False)
    metadata = pd.concat(cfg._forecast_row_parts,ignore_index=True)
    metadata["sequence_id"]=metadata.sequence_id.astype("category")
    cfg._forecast_row_parts.clear()
    metadata=metadata.set_index("flow_uid")
    if metadata.index.has_duplicates or valid.flow_uid.duplicated().any():
        raise ValueError("原始记录UID重复，无法证明划分互斥")
    identity = metadata.reindex(valid.flow_uid.to_numpy()).reset_index()
    del metadata
    identity["sequence_id"]=identity.sequence_id.astype("category")
    if identity.sequence_id.isna().any():
        raise ValueError("存在无法关联原始记录的融合行")
    folder = os.path.join(cfg.OUTPUT_DIR,"forecast_metadata");os.makedirs(folder,exist_ok=True)
    invalid = ~identity.eligible.to_numpy(dtype=bool)
    quarantine = valid.iloc[np.flatnonzero(invalid)] if invalid.any() else None
    if invalid.any():
        valid = valid.iloc[np.flatnonzero(~invalid)].reset_index(drop=True)
        identity = identity.loc[~invalid].reset_index(drop=True)
    else:
        valid.reset_index(drop=True,inplace=True)
    k = int(cfg.FORECAST_ROWS_PER_WINDOW)
    if k<1:
        raise ValueError("FORECAST_ROWS_PER_WINDOW必须为正数")
    identity["window_index"] = identity.order_index.to_numpy(dtype=np.int64)//k
    try:
        assert_train_class_coverage(valid,valid,"ordered_split_eligible_source_coverage")
        bounds,coverage = forecast_plan_class_coverage(identity,valid.Label.to_numpy(),cfg,train_size,val_size,test_size)
        # 仅复制标签列检查类别，不提前复制/写出全量特征CSV。
        label_frames = [valid.loc[identity.split.eq(s).to_numpy(),["Label"]] for s in ("train","val","test")]
        assert_train_class_coverage(valid,label_frames[0],"after_ordered_shared_split")
    except Exception as exc:
        write_json(os.path.join(folder,"class_coverage_failure.json"),
                       {"status":"failed_before_raw_spools","error":str(exc),"eligible_label_counts":
                        {str(k):int(v) for k,v in valid.Label.value_counts().items()},
                        "quarantined_rows":int(invalid.sum())})
        raise
    cfg._forecast_class_coverage = coverage
    write_json(os.path.join(folder,"forecast_sample_precheck.json"),cfg._forecast_precheck)
    write_json(os.path.join(folder,"class_coverage_adjustment.json"),coverage)
    _export_split_label_audit(valid,*label_frames,stage_name="shared_order_split")
    logger.info("[顺序划分类别覆盖] 初始缺类=%s；调整%d个前缀；实际行数=%s；仅训练片段=%s",
                coverage["initial_train_missing_labels"],len(coverage["adjustments"]),coverage["final_rows"],
                coverage["training_only_sequences"])
    if coverage["test_missing_labels"]:
        logger.warning("[评估类别范围] 测试集无样本类别=%s；不能声称这些类别已通过独立测试验证",coverage["test_missing_labels"])
    if quarantine is not None:
        atomic_csv(quarantine,os.path.join(folder,"quarantined_unordered_rows.csv"))
    atomic_csv(identity,os.path.join(folder,"row_split_index.csv"))
    shared_identity = _sha256_file(os.path.join(folder,"row_split_index.csv"))
    cfg._forecast_split_identity = shared_identity
    cfg._forecast_metadata = identity
    cfg._forecast_spools = {}
    result = []
    memory_budget_guard(int(valid.memory_usage(deep=True).sum()),"统一划分集合物化")
    for split in ("train","val","test"):
        ids = np.flatnonzero(identity.split.to_numpy()==split)
        ids = ids[np.lexsort((identity.order_index.to_numpy()[ids],identity.sequence_id.cat.codes.to_numpy()[ids]))]
        part = valid.iloc[ids]
        part.reset_index(drop=True,inplace=True)
        raw_path = os.path.join(folder,"_raw_"+split+".tmp.csv.gz")
        cfg._forecast_spools[split] = raw_path
        try:
            ensure_disk_capacity(folder,estimated_csv_bytes([part],compressed=True))
            atomic_csv(part,raw_path)
        except Exception:
            cleanup_forecast_spools(cfg)
            raise
        result.append(part)
    del part,ids,label_frames
    fingerprints = classification_row_fingerprints(valid)
    sets = [np.unique(fingerprints[identity.split.to_numpy()==split]) for split in ("train","val","test")]
    overlap = {name:int(len(np.intersect1d(sets[a],sets[b]))) for name,a,b in
               [("train_val",0,1),("train_test",0,2),("val_test",1,2)]}
    policy = {"policy":"within_capture_ordered_fixed_flow_windows","shared_split_identity":shared_identity,
              "rows":dict(zip(("train","val","test"),map(len,result))),"bounds":bounds,
              "uid_cross_split_overlap":0,"quarantined_unordered_rows":int(invalid.sum()),
              "content_fingerprint_overlap":overlap,"class_coverage_policy":coverage["policy"],
              "class_coverage_audit":"class_coverage_adjustment.json",
              "split_selected_using_labels":coverage["split_selected_using_labels"],
              "actual_ratios":coverage["actual_ratios"],
              "claim_scope":"within-capture later-window offline evaluation; label-informed split; not global-calendar online deployment"}
    classification_state()["split_policy"] = policy
    classification_state()["forecast_split_identity"] = shared_identity
    write_json(os.path.join(folder,"shared_split_manifest.json"),policy)
    cfg._forecast_row_parts.clear()
    return tuple(result)


def export_fused_forecast(cfg):
    """重放主融合处理器，不重新fit AE，不在真实序列中安排SMOTE流量。"""
    folder = os.path.join(cfg.OUTPUT_DIR,"forecast_fused"); os.makedirs(folder,exist_ok=True)
    pointer = os.path.join(cfg.MAIN_OUTPUT_DIR,"latest_fused_forecast_summary.json")
    try:
        artifacts = os.path.join(cfg.OUTPUT_DIR,"classification_preprocessing")
        transformer = ClassificationTransformer(artifacts,batch_size=cfg.AE_BATCH_SIZE)
        if transformer.state.get("forecast_split_identity") != cfg._forecast_split_identity:
            raise ValueError("主处理器拟合划分与预测划分身份不一致")
        index = cfg._forecast_metadata.set_index("flow_uid")
        del cfg._forecast_metadata
        model_columns = transformer.state["output_ae_columns"]+transformer.state["side_features"]
        categorical = transformer.state["discrete_mappings"]
        ports = transformer.state["port_features"]
        numeric = [c for c in model_columns if c not in categorical and c not in ports]
        categories = {col: sorted(set(mapping.values()))[:32] for col,mapping in categorical.items()}
        features = list(numeric)
        for col, values in categories.items():
            features += [col+"_fraction_"+str(value) for value in values]+[col+"_fraction_other"]
        if set(features)&PROVENANCE_COLUMNS: raise ValueError("元数据进入模型特征")
        rows_per_window = int(cfg.FORECAST_ROWS_PER_WINDOW)
        labels = list(cfg.STANDARD_LABELS); label_map={c:i for i,c in enumerate(labels)}
        files, audits = {}, {}
        for split in ("train","val","test"):
            sums, label_sums, counts = {}, {}, {}
            uid_digest = hashlib.sha256(); processed_rows=0
            for raw in pd.read_csv(cfg._forecast_spools[split], chunksize=cfg.FORECAST_EXPORT_CHUNK_SIZE,
                                   dtype={"flow_uid":"string","Label":"string","dataset_source":"string"}):
                joined = index.reindex(raw.flow_uid.to_numpy())
                if joined.sequence_id.isna().any() or not joined.split.eq(split).all():
                    raise ValueError("自然记录身份与统一划分不匹配")
                # raw已做固定canonical/单位归一；source=None防止同一倍率被再次应用。
                processed = transformer.transform(raw,source=None)
                blocks = [processed[numeric].to_numpy(dtype=np.float32)]
                for col, values in categories.items():
                    blocks += [(processed[col].to_numpy()==value).astype(np.float32)[:,None] for value in values]
                    blocks.append((~processed[col].isin(values)).to_numpy(dtype=np.float32)[:,None])
                transformed = np.column_stack(blocks)
                y = raw.Label.map(label_map)
                if y.isna().any(): raise ValueError("自然流导出出现未注册标签")
                local = pd.DataFrame({"series":joined.sequence_id.to_numpy(),
                            "window":joined.window_index.to_numpy(),"position":np.arange(len(raw))})
                for (series,window), positions in local.groupby(["series","window"],sort=False).indices.items():
                    key=(str(series),int(window))
                    block_sum=transformed[positions].sum(axis=0,dtype=np.float64)
                    block_labels=np.bincount(y.to_numpy(dtype=np.int64)[positions],minlength=len(labels))
                    if key not in counts:
                        sums[key]=block_sum;label_sums[key]=block_labels;counts[key]=len(positions)
                    else:
                        sums[key]+=block_sum;label_sums[key]+=block_labels;counts[key]+=len(positions)
                for uid in raw.flow_uid: uid_digest.update((str(uid)+"\n").encode())
                processed_rows+=len(raw)
            records=[]; dropped=0
            for key in sorted(counts):
                count=counts[key]
                if count != rows_per_window:
                    dropped+=count;continue
                row={"source":"fused","series":key[0],"bucket":key[1]*10**9,"window_index":key[1],"count":count}
                row.update({"label_count_"+c:int(label_sums[key][i]) for i,c in enumerate(labels)})
                row.update({name:float(sums[key][i]/count) for i,name in enumerate(features)})
                records.append(row)
            if not records: raise ValueError(f"{split}没有完整的自然顺序窗口")
            table=pd.DataFrame(records).sort_values(["series","bucket"],kind="stable")
            exported_counts={c:int(table["label_count_"+c].sum()) for c in labels}
            if split == "train":
                required=set(getattr(cfg,"_forecast_class_coverage",{}).get("required_labels",cfg.REQUIRED_BASE_LABELS))
                absent=sorted(c for c in required if exported_counts.get(c,0)==0)
                if absent: logger.warning("训练完整流量窗口缺类=%s；尾部不足一窗的记录保留在分类数据中，不伪造不足500条窗口",absent)
            name=split+"_windows.csv";atomic_csv(table,os.path.join(folder,name));files[split]=name
            cleanup_forecast_spools(cfg,split)
            audits[split]={"processed_natural_rows":processed_rows,"exported_windows":len(table),
                          "complete_windows":len(table),
                          "exported_label_counts":exported_counts,
                          "incomplete_window_rows_not_exported":dropped,"ordered_uid_sha256":uid_digest.hexdigest()}
        basis={str(name):"original_csv_row_order" for name in index.sequence_id.unique()}
        lengths={}
        for split,name in files.items():
            check=pd.read_csv(os.path.join(folder,name),usecols=["series","window_index","count"])
            if not check["count"].eq(rows_per_window).all():raise ValueError("预测窗口必须恰好500条流量")
            lengths[split]=[]
            for _,group in check.groupby("series",sort=True):
                indices=np.sort(group.window_index.to_numpy(dtype=np.int64))
                breaks=np.r_[0,np.flatnonzero(np.diff(indices)!=1)+1,len(indices)]
                lengths[split].extend(np.diff(breaks).tolist())
        readiness=forecast_sequence_precheck(lengths,cfg)
        if readiness["complete_windows"]!=cfg._forecast_precheck["complete_windows"]:
            raise RuntimeError("实际导出窗口数与划分预检不一致")
        if min(readiness["forecast_samples"].values())<100:
            logger.warning("部分集合有效预测样本少于100：%s；能运行不代表评估充分",readiness["forecast_samples"])
        write_json(os.path.join(folder,"forecast_readiness.json"),readiness)
        artifact_hashes={name:_sha256_file(os.path.join(artifacts,name))
                         for name in ("preprocessing.json","scalers.joblib","autoencoder.pt")}
        metadata={"schema_version":"fused_processed_windows_v1","status":"ready",
                  "data_product_type":"processed_fused_natural_windows","chronological_split":True,
                  "contains_synthetic":False,"natural_stream_preserved":True,"preprocessing_fit_split":"train",
                  "shared_split_identity":cfg._forecast_split_identity,"preprocessing_artifact_sha256":artifact_hashes,
                  "feature_columns":features,"label_names":labels,"files":files,
                  "window_seconds":1,"bucket_compatibility_note":"bucket=window_index*1e9 is an opaque ordinal encoding, not seconds or a real timestamp",
                  "rows_per_window":rows_per_window,
                  "seq_len":cfg.FORECAST_SEQ_LEN,"label_len":cfg.FORECAST_LABEL_LEN,"pred_len":cfg.FORECAST_PRED_LEN,
                  "time_axis_mode":"relative_flow_order","time_basis_by_source":{"fused":"row_order_proxy"},
                  "aggregation":"consecutive_flows_per_fixed_count_window",
                  "window_definition":f"{rows_per_window} completed labeled flows in one capture",
                  "ordering_basis_by_series":basis,
                  "has_order_proxy":True,
                  "timestamps_are_real":False,"split_scope":"within_capture_series",
                  "main_pipeline_augmentation":"classification CSV retains IF/ASMOTE; synthetic rows have no invented event time",
                  "preprocessing_space":"same frozen main classification fusion transformer; no independent temporal AE",
                  "aggregation_features":{"numeric_mean":numeric,"categorical_fraction_codes":categories,
                                          "raw_ports":"excluded; retain existing derived service/port indicators"},
                  "output_sha256":{split:_sha256_file(os.path.join(folder,name)) for split,name in files.items()},
                  "class_coverage_policy":getattr(cfg,"FORECAST_TRAIN_COVERAGE_POLICY","extend_prefix"),
                  "class_coverage_summary":getattr(cfg,"_forecast_class_coverage",{}),
                  "training_tail_policy":"all forecast windows complete; remainder rows retained in classification data and counted in audit",
                  "forecast_readiness":readiness,"real_timestamps_used":False,
                  "audit":audits,"optimization_revision":cfg.OPTIMIZATION_REVISION}
        write_json(os.path.join(folder,"temporal_metadata.json"),metadata)
        write_json(pointer,{"status":"ready","run_dir":cfg.OUTPUT_DIR,"forecast_dir":folder})
        logger.info("主融合处理器预测输出完成：%s；原始行顺序代理=%s",folder,metadata["has_order_proxy"])
        del transformer,index
        if torch.cuda.is_available():torch.cuda.empty_cache()
        return folder
    except Exception as exc:
        write_json(os.path.join(folder,"forecast_failure.json"),{"status":"failed","error":str(exc)})
        write_json(pointer,{"status":"failed","run_dir":cfg.OUTPUT_DIR,"error":str(exc)})
        cleanup_forecast_spools(cfg)
        raise RuntimeError("融合预测导出失败，分类CSV已保存；不能回退旧时序支路："+str(exc)) from exc


def memory_budget_guard(estimated_bytes=0, stage=""):
    """在大块分配前检查真实容器预算，不通过删样本缓解内存。"""
    cfg = get_active_config()
    virtual = psutil.virtual_memory()
    limit, used = int(virtual.total), int(virtual.total-virtual.available)
    try:
        with open("/sys/fs/cgroup/memory.max", encoding="utf-8") as handle:
            value = handle.read().strip()
        if value != "max":
            with open("/sys/fs/cgroup/memory.current", encoding="utf-8") as handle:
                container_used = int(handle.read())
            if int(value) < limit:
                limit, used = int(value), container_used
    except (OSError, ValueError):
        pass
    available = max(0, int(limit*cfg.MEMORY_BUDGET_FRACTION)-used)
    rss = psutil.Process().memory_info().rss
    cfg._memory_observations = getattr(cfg, "_memory_observations", [])
    cfg._memory_observations.append({"stage":stage,"rss_bytes":rss,
        "limit_bytes":limit,"used_bytes":used,"additional_budget_bytes":available,
        "requested_bytes":int(estimated_bytes)})
    if estimated_bytes > available:
        raise MemoryError(f"{stage}预计新增{estimated_bytes/1024**3:.2f}GiB，安全预算仅{available/1024**3:.2f}GiB；"
                          "请增加内存或使用更大的运行环境。未自动删除或抽减数据。")
    return available

def start_memory_monitor(cfg):
    stop = threading.Event()
    snapshot = {"sample_interval_seconds":0.25,"sampled_peak_rss_bytes":0,
                "sampled_peak_container_percent":0.0}
    def sample():
        while True:
            snapshot["sampled_peak_rss_bytes"] = max(snapshot["sampled_peak_rss_bytes"],psutil.Process().memory_info().rss)
            snapshot["sampled_peak_container_percent"] = max(snapshot["sampled_peak_container_percent"],get_memory_percent())
            if stop.wait(0.25):
                return
    thread = threading.Thread(target=sample, name="resource-sampler", daemon=True)
    cfg._memory_monitor = (stop,thread,snapshot)
    thread.start()

def stop_memory_monitor(cfg):
    monitor = getattr(cfg,"_memory_monitor",None)
    if monitor:
        monitor[0].set(); monitor[1].join(timeout=2)
        cfg._memory_peak = dict(monitor[2])
        del cfg._memory_monitor

def choose_csv_read_workers(paths, cfg):
    available = memory_budget_guard(stage="CSV读取并发预算")
    # CSV转成DataFrame的膨胀随列类型变化；保守限制在途文件，最终大表另作预算检查。
    largest = max((os.path.getsize(path) for path in paths),default=1)
    workers = min(len(paths),max(1,int(cfg.CSV_READ_WORKERS)),max(1,int(available//max(1,largest*4))))
    logger.info("CSV读取并发=%d；当前新增内存预算=%.2fGiB",workers,available/1024**3)
    return workers

def iter_ordered_csv_frames(csv_files, folder_path, load_frame, cfg):
    """按文件原顺序交付；每次提交新读取任务前重新评估预算。"""
    paths=[os.path.join(folder_path,name) for name in csv_files]
    workers=choose_csv_read_workers(paths,cfg)
    estimates=[max(1,os.path.getsize(path)*4) for path in paths]
    pending={};next_index=0
    with ThreadPoolExecutor(max_workers=workers) as executor:
        for current in range(len(csv_files)):
            available=memory_budget_guard(stage="读取队列补充预算")
            reserved=sum(estimates[i] for i in pending)
            while next_index<len(csv_files) and len(pending)<workers:
                estimate=estimates[next_index]
                if pending and estimate>max(0,available-reserved):break
                if not pending:memory_budget_guard(estimate,"下一个CSV读取")
                pending[next_index]=executor.submit(load_frame,csv_files[next_index])
                reserved+=estimate;next_index+=1
            frame=pending.pop(current).result()
            yield csv_files[current],frame

def forecast_sequence_precheck(lengths, cfg, raise_on_failure=True):
    """与预测器相同的连续段长度和桥接分组规则；不训练任何模型。"""
    from sklearn.model_selection import GroupKFold
    need = int(cfg.FORECAST_SEQ_LEN+cfg.FORECAST_PRED_LEN)
    result = {"seq_len":cfg.FORECAST_SEQ_LEN,"pred_len":cfg.FORECAST_PRED_LEN,
        "forecast_samples":{s:sum(max(0,int(n)-need+1) for n in lengths[s]) for s in ("train","val","test")},
        "complete_windows":{s:sum(map(int,lengths[s])) for s in ("train","val","test")},
        "continuous_lengths":{s:list(map(int,lengths[s])) for s in ("train","val","test")},
        "bridge_oof_folds":int(cfg.FORECAST_BRIDGE_OOF_FOLDS),"errors":[]}
    if not all(result["forecast_samples"].values()):
        result["errors"].append(f"每段至少需要{need}个连续完整窗口；有效样本数={result['forecast_samples']}")
    sizes = []
    for n in lengths["train"]:
        sizes.extend([need]*(int(n)//need))
        if int(n)%need: sizes.append(int(n)%need)
    result["bridge_groups"] = len(sizes)
    folds = min(int(cfg.FORECAST_BRIDGE_OOF_FOLDS),len(sizes))
    if folds < 2 or len(sizes)-int(np.ceil(len(sizes)/max(1,folds))) < 3:
        result["errors"].append("桥接分组不足：外层交叉拟合的每个fit必须至少有训练/早停/校准三个独立组")
    else:
        groups = np.repeat(np.arange(len(sizes)),sizes)
        remaining = [len(np.unique(groups[fit])) for fit,_ in GroupKFold(folds).split(groups,groups,groups)]
        result["bridge_fit_groups_per_fold"] = remaining
        if min(remaining)<3:
            result["errors"].append("桥接组大小不均导致某外层fit不足三个独立组")
    result["ready"] = not result["errors"]
    if raise_on_failure and result["errors"]:
        raise ValueError("；".join(result["errors"]))
    return result

def cleanup_forecast_spools(cfg, split=None):
    """只允许删除当前运行登记的临时单文件；从不扫描或递归删除目录。"""
    expected=Path(cfg.OUTPUT_DIR).resolve()/"forecast_metadata"
    spools=getattr(cfg,"_forecast_spools",{})
    for name,path in list(spools.items()):
        if split is not None and name!=split: continue
        absolute=Path(path).resolve()
        if absolute.parent!=expected or absolute.name!=f"_raw_{name}.tmp.csv.gz":
            raise RuntimeError("临时文件清理路径超出本次运行目录")
        if absolute.is_file(): absolute.unlink()
        spools.pop(name,None)

def detect_hardlink_support(directory):
    """在本次输出目录实测硬链接能力，探测文件始终释放。"""
    os.makedirs(directory,exist_ok=True)
    fd,source=tempfile.mkstemp(prefix=".link_probe_",dir=directory);os.close(fd)
    destination=source+".link"
    try:
        os.link(source,destination)
        return True
    except OSError:
        return False
    finally:
        for path in (source,destination):
            if os.path.isfile(path):os.remove(path)

def dataframe_float32_matrix(frame, columns):
    memory_budget_guard(len(frame)*len(columns)*4, "AE训练矩阵")
    result=np.empty((len(frame),len(columns)),dtype=np.float32)
    chunk=int(get_active_config().IO_CHUNK_ROWS)
    for start in range(0,len(frame),chunk):
        for j,col in enumerate(columns):result[start:start+chunk,j]=frame[col].iloc[start:start+chunk].to_numpy(dtype=np.float32)
    return result

def transform_frame_with_fitted_autoencoder(frame, features, model, scaler, batch_size):
    latent=int(model.encoder[-1].out_features)
    memory_budget_guard(len(frame)*(latent+1)*4,"AE验证/测试编码输出")
    output=np.empty((len(frame),latent+1),dtype=np.float32)
    for start in range(0,len(frame),batch_size):
        chunk=frame.iloc[start:start+batch_size][features].to_numpy(dtype=np.float32)
        output[start:start+len(chunk)]=transform_with_fitted_autoencoder(chunk,model,scaler,batch_size)
    return output

if __name__ == "__main__":
    print("=== 启动数据处理脚本 ===")
    print(f"当前时间: {datetime.now()}")
    print(f"Python版本: {sys.version}")
    
    try:
        # 打印配置信息
        print(f"\n配置信息:")
        print(f"主输出目录: {config.MAIN_OUTPUT_DIR}")
        print(f"本次运行输出目录: {config.OUTPUT_DIR}")
        print(f"2017数据集路径: {config.FOLDER_2017}")
        print(f"2018数据集路径: {config.FOLDER_2018}")
        
        # 检查数据集路径是否存在
        print(f"\n数据集路径检查:")
        print(f"2017数据集路径存在: {os.path.exists(config.FOLDER_2017)}")
        print(f"2018数据集路径存在: {os.path.exists(config.FOLDER_2018)}")
        
        main()
    except Exception as e:
        import traceback
        print(f"\n=== 执行失败 ===")
        traceback.print_exc()
        print(f"错误信息: {str(e)}")
        logger.error(f"执行失败：{str(e)}")
        sys.exit(1)
 
