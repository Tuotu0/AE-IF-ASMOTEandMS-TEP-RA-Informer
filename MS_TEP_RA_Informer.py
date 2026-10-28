#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""MS/TEP/RA-Informer：主融合处理器输出的离线序列预测。

先运行配套数据处理适配版，再使用 --data-dir 指定该run目录或forecast_fused。
只接受主流程同一冻结处理器、同一划分身份导出的融合特征窗口。
不读取旧原始时序旁路，不跨采集片段拼接，不为合成样本编造时间。
默认19类，Unknown权重为人工配置0.85/0.70，高风险阈值0.6。
窗口单位是固定条数的完成流；原CSV无时间戳时，必须显式允许行顺序代理实验。
桥接器按训练内部不相交组训练、早停、校准和交叉拟合，外层验证仅用于预测选模。
所有多尺度输入限定于当前seq_len片段；来源和记录身份仅作审计，不进入模型。
默认输出decoder多步风险；direct保留为另一种预测头对照。
注意力默认官方分母+确定采样，属于明确标注的改编；不是完整官方基线。
--prepare-only只做接口及有效样本检查，不拟合桥接器或预测模型。
--validation-only不输出测试性能；--checkpoint用于相同数据/配置的复现评价。
首次默认重建训练历史；显式--history-run可复用本版本完成的历史估计。
依赖numpy/pandas/scikit-learn/xgboost/matplotlib/torch；不需要本地全量训练。
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import logging
import math
import os
import random
import subprocess
import sys
import threading
import traceback
import time
from collections import defaultdict
from functools import lru_cache
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, f1_score, precision_recall_fscore_support,
                             recall_score)
from sklearn.model_selection import KFold, StratifiedKFold, GroupKFold
from sklearn.preprocessing import StandardScaler

try:
    import torch
    import torch.distributed as dist
    from torch import nn
    from torch.nn import functional as F
    from torch.nn.parallel import DistributedDataParallel as DDP
    from torch.utils.data import DistributedSampler, Sampler, WeightedRandomSampler
except ImportError:
    torch = dist = nn = F = DDP = DistributedSampler = Sampler = WeightedRandomSampler = None


LABELS = [
    "Normal", "FTP_BruteForce", "SSH_BruteForce", "DoS_Hulk", "DoS_GoldenEye",
    "DoS_Slowloris", "DoS_SlowHTTPTest", "DDoS_LOIC_HTTP", "DDoS_LOIC_UDP",
    "DDoS_HOIC", "Web_SQL_Injection", "Web_XSS", "Web_BruteForce",
    "Infiltration", "Botnet", "Heartbleed", "PortScan", "DDoS_Unknown", "DoS_Unknown",
]
# 首版人工设定的严重度（0~1），动态类别顺序与主流程保持一致。
RISK_WEIGHTS = dict(zip(LABELS, [
    0., .50, .50, .75, .65, .60, .60, .85, .85, .95, .90, .60, .65,
    .90, .80, .95, .35, .85, .70,
]))
STATES = ["Normal", "Reconnaissance", "Access", "Foothold", "Impact"]
STATE_IDS = dict(zip(LABELS, [0, 2, 2, 4, 4, 4, 4, 4, 4, 4, 2, 2, 2, 3, 3, 4, 1, 4, 4]))
REVISION = "20261019_fused_interface_and_method_v5"
BRIDGE_REVISION = "train_group_crossfit_calibrated_v2"
ACTIVE_OUTPUT = None
ACTIVE_STAGE = "initializing"
SPLITS = ("train", "val", "test")
FILES = {s: f"{s}_windows.csv" for s in SPLITS}
DEFAULT_ROOT = Path(os.environ.get("DATAHANDLE_OUTPUT_ROOT", "/root/shared-nvme/timeseries_processing_results"))
LOG = logging.getLogger("MS_TEP_RA_Informer_Optimized")
EXCLUDED = {"label", "dataset_source", "timestamp", "sequence_id", "flow_uid",
            "flow_id", "src_ip", "dst_ip", "is_synthetic", "synthetic", "split"}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", type=Path, help="本次run目录或包含run目录的根目录")
    p.add_argument("--history-run", type=Path, help="已完成的历史风险估计run；默认寻找与输入SHA256匹配的缓存")
    p.add_argument("--rebuild-history", action="store_true", help="显式重建上游历史风险；通常无需重复执行")
    p.add_argument("--forecast-method", choices=["direct", "decoder"], default="decoder",
                   help="decoder=完整Informer主模型；direct=上一版实现对照")
    p.add_argument("--loss-mode", choices=["mse", "balanced", "legacy"], default="balanced")
    p.add_argument("--history-aux-weight", type=float, default=.1,
                   help="训练集历史风险辅助监督权重；0关闭；真值不进入前向输入")
    p.add_argument("--no-source-features", action="store_true", default=True)
    p.add_argument("--no-source-balance", action="store_true", default=True)
    p.add_argument("--legacy-input-norm", action="store_true",
                   help="消融：恢复跨特征LayerNorm；默认保留已标准化的绝对幅度")
    p.add_argument("--deep-residual-scale", type=float, default=.2,
                   help="深层多步修正的学习尺度；保持线性输出，不限制可预测风险上限")
    p.add_argument("--mode", choices=["temporal"], default="temporal", help="只接受配套的真实时序数据")
    p.add_argument("--output-dir", type=Path, help="结果父目录，每次另建run子目录")
    p.add_argument("--window-seconds", type=int, default=None, help="默认读取上游时间粒度；不可与上游不同")
    p.add_argument("--seq-len", type=int, default=None)
    p.add_argument("--label-len", type=int, default=None)
    p.add_argument("--pred-len", type=int, default=None)
    p.add_argument("--scales", type=int, nargs="+", default=[3, 12, 48])
    p.add_argument("--max-train-samples", type=int, default=60000,
                   help="只限制完整训练窗口数量；0=全部。不对原始行随机抽样")
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--patience", type=int, default=10)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--d-model", type=int, default=64)
    p.add_argument("--heads", type=int, default=4)
    p.add_argument("--encoder-layers", type=int, default=2)
    p.add_argument("--decoder-layers", type=int, default=1)
    p.add_argument("--factor", type=int, default=5)
    p.add_argument("--dropout", type=float, default=.15)
    p.add_argument("--risk-source", choices=["xgboost", "oracle"], default="xgboost",
                   help="xgboost=默认端到端历史输入；oracle=真实历史标签理想上限")
    p.add_argument("--xgb-sampling", choices=["asmote", "none"], default="none",
                   help="XGBoost训练窗口是否执行ASMOTE；仅作用于时序train")
    p.add_argument("--xgb-oof-folds", type=int, default=3,
                   help="train历史概率的OOF折数；1表示使用训练内概率，仅用于快速联调")
    p.add_argument("--xgb-n-estimators", type=int, default=800)
    p.add_argument("--xgb-learning-rate", type=float, default=.05)
    p.add_argument("--xgb-max-depth", type=int, default=9)
    p.add_argument("--xgb-early-stopping", type=int, default=50)
    p.add_argument("--xgb-device", choices=["auto", "cpu", "cuda"], default="auto")
    p.add_argument("--xgb-attack-threshold", type=float, default=.01,
                   help="窗口攻击总概率达到该值时输出概率最大的攻击类，否则输出Normal")
    p.add_argument("--xgb-risk-bridge", choices=["direct", "class_probability"], default="class_probability",
                   help="direct用训练窗口真实风险拟合独立回归器；class_probability复现旧版类别概率加权风险")
    p.add_argument("--xgb-risk-estimators", type=int, default=350)
    p.add_argument("--xgb-risk-depth", type=int, default=5)
    p.add_argument("--xgb-min-class-target", type=int, default=100,
                   help="ASMOTE期望的最小窗口级类别数，仍受单类扩增倍数限制")
    p.add_argument("--risk-threshold", type=float, default=.6)
    p.add_argument("--risk-alpha", type=float, default=1.)
    p.add_argument("--trend-weight", type=float, default=.05)
    p.add_argument("--trend-epsilon", type=float, default=1e-4)
    p.add_argument("--huber-delta", type=float, default=.08,
                   help="鲁棒Huber损失转折点；降低异常尖峰对总体拟合的破坏")
    p.add_argument("--underprediction-weight", type=float, default=.1,
                   help="真实高风险区间低估惩罚权重")
    p.add_argument("--horizon-weight", type=float, default=.2,
                   help="远期误差附加权重，0表示各步相同")
    p.add_argument("--risk-sampler-alpha", type=float, default=0.,
                   help="默认0避免与高风险损失重复加权；大于0启用含高风险窗口的过采样")
    p.add_argument("--ema-decay", type=float, default=0.,
                   help="指数滑动平均权重；0关闭")
    p.add_argument("--weight-decay", type=float, default=.001)
    p.add_argument("--warmup-epochs", type=int, default=3)
    p.add_argument("--min-lr", type=float, default=1e-6)
    p.add_argument("--transition-smoothing", type=float, default=1.)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--workers", type=int, default=0)
    p.add_argument("--prefetch-factor", type=int, default=2)
    p.add_argument("--threads", type=int, default=8)
    p.add_argument("--device", default="auto", help="auto/cpu/cuda/cuda:0；auto会优先使用CUDA")
    p.add_argument("--gpu-ids", default="0",
                   help="默认单卡0；小数据集避免DataParallel调度开销。多卡填0,1；auto使用全部可见GPU")
    p.add_argument("--no-amp", action="store_true")
    p.add_argument("--amp-dtype", choices=["auto", "bf16", "fp16"], default="auto")
    p.add_argument("--compile", action="store_true", help="PyTorch 2.x编译模型；长训练通常更省时")
    p.add_argument("--deterministic", action="store_true", default=True,
                   help="默认启用确定性计算；同环境同单卡同seed可复现")
    p.add_argument("--fast", dest="deterministic", action="store_false",
                   help="关闭确定性计算以追求速度；结果可能变化")
    p.add_argument("--selection-metric", choices=["macro_skill", "mae", "rmse", "objective"], default="mae",
                   help="默认按来源等权的MAE/RMSE相对Persistence比值选择，越低越好")
    p.add_argument("--no-ms", action="store_true")
    p.add_argument("--no-tep", action="store_true")
    p.add_argument("--no-ra", action="store_true")
    p.add_argument("--no-time-features", action="store_true", default=True,
                   help="默认关闭小时/星期特征，减少混合时间轴和采集日期的伪相关")
    p.add_argument("--time-features", dest="no_time_features", action="store_false",
                   help="显式启用小时/星期特征，供消融实验使用")
    p.add_argument("--no-feature-gate", action="store_true")
    p.add_argument("--no-local-mixer", action="store_true")
    p.add_argument("--no-residual-head", action="store_true")
    p.add_argument("--legacy-probsparse-score", action="store_true",
                   help="仅用于消融：复现原脚本的ProbSparse均值分母")
    p.add_argument("--prepare-only", action="store_true")
    p.add_argument("--validation-only", action="store_true", help="只训练并报告验证集，用于选参数；不评估测试集")
    p.add_argument("--test-role", choices=["development", "final"], default="development",
                   help="留出用途由调用者声明，不自动证明新场景泛化")
    p.add_argument("--checkpoint", type=Path)
    p.add_argument("--allow-row-order-proxy", action="store_true", help="显式允许无时间戳原CSV行顺序代理实验；不恢复真实时间")
    p.add_argument("--risk-config", type=Path, help="可选JSON：risk_weights与state_ids；缺失类别严格报错")
    p.add_argument("--sparse-score", choices=["paper", "sample_mean"], default="paper",
                   help="paper沿用官方分母；sample_mean是需验证的变体")
    p.add_argument("--sparse-sampling", choices=["uniform", "random"], default="uniform",
                   help="uniform为确定采样变体；random为随机键采样")
    p.add_argument("--loss-only-mse", action="store_true", help="只关闭风险损失项，不改网络结构")
    p.add_argument("--transition-mode", choices=["learned", "identity"], default="learned")
    a = p.parse_args()
    for k in ("window_seconds", "seq_len", "label_len", "pred_len", "batch_size", "epochs", "patience", "d_model", "heads",
              "encoder_layers", "decoder_layers", "factor", "threads", "prefetch_factor",
              "xgb_oof_folds", "xgb_n_estimators", "xgb_max_depth", "xgb_early_stopping",
               "xgb_min_class_target", "xgb_risk_estimators", "xgb_risk_depth"):
        if getattr(a, k) is not None and getattr(a, k) <= 0:
            p.error(f"{k} 必须为正整数")
    if a.d_model % a.heads or a.d_model % 2:
        p.error("d-model必须为偶数且可被heads整除")
    if (a.label_len is not None and a.seq_len is not None and a.label_len > a.seq_len) or (a.seq_len is not None and a.seq_len < 4):
        p.error("要求 1 <= label-len <= seq-len，且seq-len >= 4")
    if not a.no_time_features:
        p.error("本版使用固定完成流条数窗口，不使用小时/星期周期特征")
    if a.max_train_samples < 0 or a.workers < 0 or min(a.scales) < 1 or a.xgb_oof_folds < 1:
        p.error("样本限制、workers、scales取值无效")
    if not 0 < a.risk_threshold <= 1 or not 0 <= a.dropout < 1:
        p.error("risk-threshold需在(0,1]，dropout需在[0,1)")
    if not 0 <= a.xgb_attack_threshold <= 1:
        p.error("xgb-attack-threshold需在[0,1]")
    if min(a.lr, a.min_lr, a.transition_smoothing, a.huber_delta,
           a.xgb_learning_rate) <= 0 or min(
            a.risk_alpha, a.trend_weight, a.trend_epsilon, a.underprediction_weight,
            a.horizon_weight, a.risk_sampler_alpha, a.weight_decay, a.ema_decay, a.history_aux_weight) < 0:
        p.error("学习率/平滑系数须大于0；损失权重和趋势阈值不能为负")
    if not 0 <= a.ema_decay < 1 or a.warmup_epochs < 0:
        p.error("ema-decay需在[0,1)，warmup-epochs不能为负")
    if a.xgb_oof_folds < 2:
        p.error("正式历史输入必须交叉拟合，xgb-oof-folds至少为2")
    if a.forecast_method == "direct" and a.no_feature_gate:
        p.error("direct实现没有特征门控；该消融请使用 --forecast-method decoder")
    return a


def dump_json(path, value):
    def clean(x):
        if isinstance(x, dict):
            return {str(k): clean(v) for k, v in x.items()}
        if isinstance(x, (list, tuple)):
            return [clean(v) for v in x]
        if isinstance(x, np.ndarray):
            return clean(x.tolist())
        if isinstance(x, np.generic):
            return clean(x.item())
        if isinstance(x, Path):
            return str(x)
        if isinstance(x, float) and not math.isfinite(x):
            return None
        return x
    Path(path).write_text(json.dumps(clean(value), ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def complete_run(p):
    return p.is_dir() and (p/"temporal_metadata.json").is_file() and all(
        (p/f).is_file() and (p/f).stat().st_size > 0 for f in FILES.values())


def resolve_run(explicit=None):
    """只接受适配版从主融合处理器导出的预测窗口，不回退到旧原始旁路。"""
    roots = [explicit.expanduser().resolve()] if explicit else [DEFAULT_ROOT]
    for root in roots:
        for directory in (root, root/"forecast_fused"):
            if complete_run(directory):
                meta = json.loads((directory/"temporal_metadata.json").read_text(encoding="utf-8-sig"))
                if meta.get("data_product_type") == "processed_fused_natural_windows":
                    return directory
        pointer = root/"latest_fused_forecast_summary.json"
        if pointer.is_file():
            data = json.loads(pointer.read_text(encoding="utf-8-sig"))
            if data.get("status") != "ready":
                raise ValueError("最新融合预测导出未就绪："+data.get("error", "请等待处理完成"))
            path = root/Path(data["run_dir"]).name/"forecast_fused"
            if complete_run(path): return path
            raise FileNotFoundError(f"摘要指向的融合预测文件不完整：{path}")
    raise FileNotFoundError("未找到forecast_fused。先运行本次数据处理适配版；旧temporal_prediction不能代替融合处理输出")


def temporal_contract(run, a):
    meta = json.loads((run/"temporal_metadata.json").read_text(encoding="utf-8-sig"))
    expected = {"status": "ready", "chronological_split": True, "contains_synthetic": False,
                "natural_stream_preserved": True, "preprocessing_fit_split": "train"}
    if any(meta.get(k) != v for k, v in expected.items()):
        raise ValueError("输入不是配套主流程导出的完整时序数据，请重新生成。")
    if meta.get("schema_version") not in {
        "fused_processed_windows_v1"
    }:
        raise ValueError("时序数据版本不兼容，请使用配套数据处理脚本重新生成")
    if meta.get("data_product_type") != "processed_fused_natural_windows":
        raise ValueError("只接受同一主融合处理器导出的自然顺序窗口")
    if not meta.get("shared_split_identity") or not meta.get("preprocessing_artifact_sha256"):
        raise ValueError("缺少统一划分或主流程处理器身份")
    if meta.get("has_order_proxy") and not a.allow_row_order_proxy:
        raise ValueError("部分原始CSV没有可验证时间顺序。请提供含Timestamp数据重新处理；若仅作原CSV行顺序代理实验，显式添加 --allow-row-order-proxy，报告将标明限制")
    configure_label_contract(meta, a)
    time_axis = meta.get("time_axis_mode", "absolute_timestamp")
    if time_axis not in {"absolute_timestamp", "relative_flow_order", "mixed"}:
        raise ValueError("时序元数据中的时间轴模式无效")
    allowed = {"completed_flows_per_fixed_window", "completed_flows_per_fixed_time_window",
               "consecutive_flows_per_fixed_count_window", "mixed_fixed_time_and_flow_count_windows"}
    if meta.get("aggregation") not in allowed:
        raise ValueError("时序元数据中的聚合方式无效")
    if time_axis == "relative_flow_order" and int(meta.get("rows_per_window", 0)) <= 0:
        raise ValueError("相对顺序时序缺少有效的rows_per_window")
    a.time_axis_mode = time_axis
    a.time_basis_by_source = {str(k): str(v) for k, v in meta.get("time_basis_by_source", {}).items()}
    if meta.get("label_names") != LABELS or meta.get("files") != FILES:
        raise ValueError("预测文件名或动态类别映射不一致")
    if a.window_seconds is not None and a.window_seconds != meta["window_seconds"]:
        raise ValueError("--window-seconds与上游不一致；改变时间粒度需要先重新生成数据，不能对已聚合数据再次按流量聚合。")
    a.window_seconds = int(meta["window_seconds"])
    for k in ["seq_len", "label_len", "pred_len"]:
        if getattr(a, k) is None:
            setattr(a, k, int(meta[k]))
    if min(a.seq_len, a.label_len, a.pred_len, a.window_seconds) < 1 or a.seq_len < 4 or a.label_len > a.seq_len:
        raise ValueError("元数据/命令行的预测窗口参数不合法，要求seq-len>=4且label-len<=seq-len")
    features = meta.get("feature_columns", [])
    if not features or len(features) != len(set(features)):
        raise ValueError("时序元数据缺少唯一的预测特征列")
    if any(c.lower() in EXCLUDED or c in {"risk", "Label", "count", "bucket", "timestamp", "source", "series"} or c.startswith("label_count_") for c in features):
        raise ValueError("元数据错误：目标/时间/标签计数不能作为上游AE特征列")
    return meta


def read_temporal_windows(run, split, meta):
    path = run/FILES[split]
    expected = meta.get("output_sha256", {}).get(split)
    if not expected or sha256_file(path) != expected:
        raise ValueError(f"{path.name}校验值与导出契约不一致，请重新完整导出")
    frame = pd.read_csv(path, dtype={"source": "string", "series": "string", "bucket": "int64"})
    features = meta["feature_columns"]
    count_cols = ["label_count_"+c for c in LABELS]
    required = features+count_cols+["source", "series", "bucket", "count"]
    missing = set(required)-set(frame.columns)
    if missing:
        raise ValueError(f"{path.name}缺列：{sorted(missing)}")
    if frame.empty or frame[["source", "series", "bucket"]].isna().any().any():
        raise ValueError(f"{path.name}为空或来源/采集/时间标识缺失")
    if frame.duplicated(["source", "series", "bucket"]).any():
        raise ValueError(f"{path.name}有重复时间窗口")
    if np.any(frame.bucket.to_numpy() % (int(meta["window_seconds"])*10**9)):
        raise ValueError("时序窗口索引未对齐上游固定步长")
    values = frame[features].to_numpy(dtype=np.float64)
    counts = frame[count_cols].to_numpy(dtype=np.float64)
    total = frame["count"].to_numpy(dtype=np.float64)
    if not np.isfinite(values).all() or not np.isfinite(counts).all() or not np.isfinite(total).all():
        raise ValueError("时序输出含NaN/Inf；应由配套上游预处理修复")
    if np.any(counts < 0) or np.any(counts != np.floor(counts)) or np.any(total <= 0) or not np.array_equal(counts.sum(1), total):
        raise ValueError("流量计数不合法或动态类别计数总和不守恒")
    table = frame[["source", "series", "bucket", "count"]].copy()
    for i, column in enumerate(features):
        table[f"flow_mean_{i}"] = values[:, i]
    for column in count_cols:
        table[column] = frame[column].to_numpy(dtype=np.int64)
    severity = np.array([RISK_WEIGHTS[c] for c in LABELS])
    true_fraction = counts/total[:, None]
    table["oracle_risk"] = true_fraction@severity
    # risk始终是未来监督目标；observed_risk才是历史输入。XGBoost桥接后只替换后者。
    table["risk"] = table["oracle_risk"]
    table["observed_risk"] = table["oracle_risk"]
    table["attack_ratio"] = 1-true_fraction[:, 0]
    for i in range(len(STATES)):
        selected = [j for j, c in enumerate(LABELS) if STATE_IDS[c] == i]
        table[f"state_{i}"] = true_fraction[:, selected].sum(1)
    table = table.sort_values(["source", "series", "bucket"], kind="stable").reset_index(drop=True)
    audit = {"real_flow_rows": int(total.sum()), "aggregated_windows": len(table),
             "label_counts": dict(zip(LABELS, counts.sum(0).astype(int).tolist())),
             "oracle_risk_mean": float(table["oracle_risk"].mean()),
             "feature_column_mapping": {f"flow_mean_{i}": c for i, c in enumerate(features)}}
    return table, audit


def _classifier_arrays(table):
    feature_cols = [c for c in table.columns if c.startswith("flow_mean_")]
    x = table[feature_cols].to_numpy(dtype=np.float32)
    counts = table[["label_count_"+c for c in LABELS]].to_numpy(dtype=np.float64)
    fractions = counts/counts.sum(axis=1, keepdims=True)
    dominant = fractions.argmax(axis=1).astype(np.int32)
    if not np.isfinite(x).all() or not np.isfinite(fractions).all():
        raise ValueError("桥接输入包含NaN/Inf")
    return x, fractions, dominant, feature_cols


def _expand_window_distributions(x, fractions, indices=None):
    """将一个混合窗口展开为存在类别的加权训练行，避免只学习多数标签。"""
    if indices is None:
        indices = np.arange(len(x), dtype=np.int64)
    block = fractions[indices]
    window_pos, labels = np.nonzero(block > 0)
    expanded_x = x[indices[window_pos]]
    weights = block[window_pos, labels].astype(np.float32)
    return expanded_x, labels.astype(np.int32), weights


def _adaptive_asmote(x, y, a, seed):
    """KMeansSMOTE优先；极小类先ROS，失败时回退SMOTE/ROS，并严格限制扩增倍数。"""
    before = pd.Series(y).value_counts().sort_index()
    audit = {"requested": a.xgb_sampling, "before": before.to_dict(), "method": "none"}
    if a.xgb_sampling == "none" or len(before) < 2:
        audit["after"] = before.to_dict()
        return x, y, audit
    try:
        from imblearn.over_sampling import KMeansSMOTE, RandomOverSampler, SMOTE
        from sklearn.cluster import MiniBatchKMeans
    except ImportError as exc:
        raise RuntimeError(
            "使用--xgb-sampling asmote需要imbalanced-learn；请在服务器执行 "
            "pip install imbalanced-learn"
        ) from exc

    largest = int(before.max())
    targets = {}
    for cls, count in before.items():
        multiplier = 20 if count < 5 else 4
        target = min(largest, max(int(count), int(a.xgb_min_class_target)), int(count)*multiplier)
        if target > count:
            targets[int(cls)] = int(target)
    if not targets:
        audit["after"] = before.to_dict()
        return x, y, audit

    # SMOTE至少需要两个同类点；极小类先确定性随机复制到2条。
    pre_targets = {int(c): 2 for c, n in before.items() if n < 2 and targets.get(int(c), n) >= 2}
    if pre_targets:
        ros = RandomOverSampler(sampling_strategy=pre_targets, random_state=seed)
        x, y = ros.fit_resample(x, y)
    current = pd.Series(y).value_counts()
    targets = {c: n for c, n in targets.items() if n > int(current.get(c, 0))}
    if not targets:
        audit["method"] = "RandomOverSampler(extreme-only)"
    else:
        try:
            clusters = min(10, len(x))
            sampler = KMeansSMOTE(
                sampling_strategy=targets,
                random_state=seed,
                k_neighbors=1,
                cluster_balance_threshold=0.0,
                kmeans_estimator=MiniBatchKMeans(
                    n_clusters=clusters, batch_size=1024, n_init=3, random_state=seed
                ),
            )
            x, y = sampler.fit_resample(x, y)
            audit["method"] = "KMeansSMOTE"
        except Exception as km_exc:
            LOG.warning("KMeansSMOTE失败，回退SMOTE：%s", km_exc)
            try:
                sampler = SMOTE(sampling_strategy=targets, random_state=seed, k_neighbors=1)
                x, y = sampler.fit_resample(x, y)
                audit["method"] = "SMOTE"
            except Exception as smote_exc:
                LOG.warning("SMOTE失败，回退RandomOverSampler：%s", smote_exc)
                sampler = RandomOverSampler(sampling_strategy=targets, random_state=seed)
                x, y = sampler.fit_resample(x, y)
                audit["method"] = "RandomOverSampler"
    after = pd.Series(y).value_counts().sort_index()
    audit.update({"targets": targets, "after": after.to_dict()})
    return np.asarray(x, dtype=np.float32), np.asarray(y, dtype=np.int32), audit


def _xgb_cuda_requested(a):
    if a.xgb_device == "cpu":
        return False
    available = bool(torch is not None and torch.cuda.is_available())
    if a.xgb_device == "cuda" and not available:
        raise RuntimeError("--xgb-device cuda已指定，但当前环境未检测到CUDA")
    return available


def weighted_asmote(x, y, weights, a, seed):
    """增强时同步插值比例权重，并恢复每类总质量，避免将混合窗口权重重置为1。

    权重仅参与训练增强，不作为分类器输入；合成样本不进入时序窗口。
    此路径只在显式重建历史时生效，本轮固定旧历史的比较不受影响。
    """
    augmented, labels, audit = _adaptive_asmote(np.column_stack([x, weights]), y, a, seed)
    result_weights = np.maximum(augmented[:, -1].astype(np.float64), 1e-12)
    mass = {}
    for label in np.unique(y):
        total = float(weights[y == label].sum())
        mask = labels == label
        if audit["method"] != "none":
            result_weights[mask] *= total / result_weights[mask].sum()
        mass[int(label)] = {"before": total, "after": float(result_weights[mask].sum())}
    audit.update({"weight_policy": "joint_interpolation_then_class_mass_preservation",
                  "class_weight_mass": mass})
    return augmented[:, :-1].astype(np.float32), labels, result_weights.astype(np.float32), audit


def _fit_xgb(x, y_global, weights, x_eval, y_eval_global, a, seed, eval_weights=None):
    try:
        from xgboost import XGBClassifier
    except ImportError as exc:
        raise RuntimeError(
            "缺少xgboost。请在服务器环境执行 pip install xgboost 后重新运行。"
        ) from exc
    classes = np.array(sorted(np.unique(y_global).tolist()), dtype=np.int32)
    if len(classes) < 2:
        raise ValueError("时序train只包含一个可训练攻击类别，无法训练XGBoost多分类器")
    class_to_local = {int(c): i for i, c in enumerate(classes)}
    y = np.array([class_to_local[int(v)] for v in y_global], dtype=np.int32)
    eval_mask = np.isin(y_eval_global, classes) if len(y_eval_global) else np.zeros(0, dtype=bool)
    y_eval = np.array([class_to_local[int(v)] for v in y_eval_global[eval_mask]], dtype=np.int32)
    x_eval_seen = x_eval[eval_mask]

    base = dict(
        objective="multi:softprob", num_class=len(classes),
        n_estimators=a.xgb_n_estimators, learning_rate=a.xgb_learning_rate,
        max_depth=a.xgb_max_depth, subsample=.9, colsample_bytree=.9,
        min_child_weight=1, reg_alpha=.05, reg_lambda=1.,
        tree_method="hist", eval_metric="mlogloss", random_state=seed,
        n_jobs=max(1, a.threads), verbosity=0,
    )
    if len(x_eval_seen) and a.xgb_early_stopping > 0:
        base["early_stopping_rounds"] = a.xgb_early_stopping

    def train(params):
        model = XGBClassifier(**params)
        fit_args = {"sample_weight": weights, "verbose": False}
        if len(x_eval_seen):
            fit_args["eval_set"] = [(x_eval_seen, y_eval)]
            if eval_weights is not None:
                fit_args["sample_weight_eval_set"] = [np.asarray(eval_weights)[eval_mask]]
        model.fit(x, y, **fit_args)
        return model

    use_cuda = _xgb_cuda_requested(a)
    used_device = "cpu"
    if use_cuda:
        try:
            model = train({**base, "device": "cuda"})
            used_device = "cuda"
        except Exception as modern_exc:
            LOG.warning("XGBoost现代CUDA接口失败，尝试gpu_hist：%s", modern_exc)
            legacy = dict(base)
            legacy.pop("early_stopping_rounds", None)
            try:
                model = train({**legacy, "tree_method": "gpu_hist", "predictor": "gpu_predictor"})
                used_device = "cuda(gpu_hist)"
            except Exception as legacy_exc:
                if a.xgb_device == "cuda":
                    raise RuntimeError("XGBoost未能启用CUDA") from legacy_exc
                LOG.warning("XGBoost CUDA不可用，自动回退CPU：%s", legacy_exc)
                model = train(base)
    else:
        model = train(base)
    actual = json.loads(model.get_booster().save_config())["learner"]["generic_param"].get("device", used_device)
    if a.xgb_device == "cuda" and not actual.startswith("cuda"):
        raise RuntimeError(f"XGBoost分类器请求CUDA但实际为{actual}")
    LOG.info("XGBoost分类 seed=%d 样本=%d 实际训练设备=%s", seed, len(y), actual)
    return model, classes, actual


def _predict_global_probabilities(model, classes, x):
    local = np.asarray(model.predict_proba(x), dtype=np.float64)
    if local.ndim == 1:
        local = np.column_stack([1-local, local])
    out = np.zeros((len(x), len(LABELS)), dtype=np.float64)
    out[:, classes] = local
    row_sum = out.sum(axis=1, keepdims=True)
    if np.any(row_sum <= 0) or not np.isfinite(out).all():
        raise RuntimeError("XGBoost产生无效类别概率")
    return out/row_sum


def _temperature_scale(probabilities, temperature):
    powered = np.clip(probabilities, 1e-12, 1.)**(1./float(temperature))
    return powered/powered.sum(axis=1, keepdims=True)


def _fit_temperature(probabilities, true_fraction):
    candidates = np.geomspace(.5, 3., 61)
    losses = []
    for value in candidates:
        calibrated = _temperature_scale(probabilities, value)
        losses.append(float(-(true_fraction*np.log(np.clip(calibrated, 1e-12, 1.))).sum(axis=1).mean()))
    best = int(np.argmin(losses))
    return float(candidates[best]), float(losses[best])


def _predicted_attack_ids(probabilities, threshold):
    attack_mass = 1-probabilities[:, 0]
    predicted = (1+probabilities[:, 1:].argmax(axis=1)).astype(np.int32)
    predicted[attack_mass < threshold] = 0
    return predicted


def _classification_metrics(table, prob, split, threshold):
    _, fractions, dominant, _ = _classifier_arrays(table)
    predicted = prob.argmax(axis=1)
    return {"split": split, "windows": len(table),
            "target_semantics": "window_class_composition",
            "dominant_accuracy": float(accuracy_score(dominant, predicted)),
            "dominant_macro_F1": float(f1_score(dominant, predicted, labels=np.arange(len(LABELS)),
                                                average="macro", zero_division=0)),
            "soft_cross_entropy": float(-(fractions*np.log(np.clip(prob, 1e-12, 1))).sum(1).mean()),
            "class_fraction_MAE": float(np.abs(fractions-prob).mean()),
            "attack_fraction_MAE": float(np.abs(fractions[:, 1:].sum(1)-(1-prob[:, 0])).mean())}


def _risk_regressor_arrays(tables, feature_cols):
    """所有来源共享特征空间；来源信息只用于审计。"""
    names = list(feature_cols)+["log_count"]
    arrays = {split: np.ascontiguousarray(np.column_stack([
        table[feature_cols].to_numpy(dtype=np.float32), np.log1p(table["count"]).to_numpy()
    ]), dtype=np.float32) for split, table in tables.items()}
    if any(not np.isfinite(x).all() for x in arrays.values()):
        raise ValueError("风险回归输入包含NaN/Inf")
    return arrays, names


def _fit_xgb_risk(x, y, a, seed):
    from xgboost import XGBRegressor
    params = dict(
        objective="reg:squarederror", n_estimators=a.xgb_risk_estimators,
        max_depth=a.xgb_risk_depth, learning_rate=a.xgb_learning_rate,
        subsample=.85, colsample_bytree=.9, min_child_weight=20,
        reg_lambda=5., tree_method="hist", n_jobs=max(1, a.threads),
        random_state=seed, verbosity=0,
    )
    use_cuda = _xgb_cuda_requested(a)
    if use_cuda:
        params["device"] = "cuda"
    try:
        model = XGBRegressor(**params)
        model.fit(x, y)
        actual = json.loads(model.get_booster().save_config())["learner"]["generic_param"].get("device", "unknown")
        if use_cuda and not actual.startswith("cuda"):
            raise RuntimeError(f"XGBoost风险回归器请求CUDA但实际为{actual}")
    except Exception:
        if a.xgb_device != "auto" or not use_cuda:
            raise
        LOG.warning("XGBoost风险回归器CUDA失败，回退CPU", exc_info=True)
        params["device"] = "cpu"
        model = XGBRegressor(**params)
        model.fit(x, y)
        actual = "cpu"
    LOG.info("XGBoost风险回归 seed=%d 样本=%d 实际训练设备=%s", seed, len(y), actual)
    return model


def _fit_direct_risk_bridge(tables, feature_cols, folds, use_stratified,
                            dominant_train, a, out):
    """训练集用OOF风险估计，验证/测试只用完整训练集拟合的回归器。"""
    arrays, names = _risk_regressor_arrays(tables, feature_cols)
    y = tables["train"]["oracle_risk"].to_numpy(dtype=np.float32)
    oof = np.empty(len(y), dtype=np.float32)
    if folds > 1:
        splitter = (StratifiedKFold(folds, shuffle=True, random_state=a.seed)
                    if use_stratified else KFold(folds, shuffle=True, random_state=a.seed))
        split_iter = (splitter.split(arrays["train"], dominant_train) if use_stratified
                      else splitter.split(arrays["train"]))
        for fold, (fit_idx, hold_idx) in enumerate(split_iter, 1):
            model = _fit_xgb_risk(arrays["train"][fit_idx], y[fit_idx], a, a.seed+fold)
            oof[hold_idx] = model.predict(arrays["train"][hold_idx])
    model = _fit_xgb_risk(arrays["train"], y, a, a.seed)
    if folds == 1:
        oof = model.predict(arrays["train"])
    model.save_model(str(out/"xgboost_risk_regressor.json"))
    predictions = {"train": np.clip(oof, 0., 1.)}
    for split in ("val", "test"):
        predictions[split] = np.clip(model.predict(arrays[split]), 0., 1.)
    return predictions, names


def fit_xgboost_risk_bridge(tables, out, a):
    """外层分组交叉拟合；早停和温度校准均限于外层fit中的不同组。"""
    x, fractions, dominant, feature_cols = _classifier_arrays(tables["train"])
    groups = bridge_groups(tables["train"], a)
    unique = np.unique(groups)
    folds = min(a.xgb_oof_folds, len(unique))
    if folds < 2:
        raise ValueError("历史交叉拟合至少需要2个独立组")
    if min(len(unique)-int(math.ceil(len(unique)/folds)), len(unique)-1) < 3:
        raise ValueError("分组不足：每个外层fit至少需要训练、早停、校准三个不同组；请增加数据/场景长度")
    features = {split: _classifier_arrays(table)[0] for split, table in tables.items()}
    oof = np.empty((len(x), len(LABELS)), dtype=np.float64)
    oof_risk = np.empty(len(x), dtype=np.float64)
    predicted = {s: [] for s in ("val", "test")}
    direct = {s: [] for s in ("val", "test")}
    risk_arrays, risk_names = _risk_regressor_arrays(tables, feature_cols)
    audits, class_files = [], []
    for fold, (fit_idx, hold_idx) in enumerate(GroupKFold(folds).split(x, dominant, groups), 1):
        core, stop, calibration = bridge_inner_partition(fit_idx, groups, a.seed+fold)
        tx, ty, tw = _expand_window_distributions(x, fractions, core)
        tx, ty, tw, sampling = weighted_asmote(tx, ty, tw, a, a.seed+fold)
        sx, sy, sw = _expand_window_distributions(x, fractions, stop)
        model, classes, device = _fit_xgb(tx, ty, tw, sx, sy, a, a.seed+fold, eval_weights=sw)
        raw = _predict_global_probabilities(model, classes, x[calibration])
        temperature, calibration_loss = _fit_temperature(raw, fractions[calibration])
        oof[hold_idx] = _temperature_scale(_predict_global_probabilities(model, classes, x[hold_idx]), temperature)
        for split in predicted:
            predicted[split].append(_temperature_scale(
                _predict_global_probabilities(model, classes, features[split]), temperature))
        name = f"bridge_fold_{fold:02d}_classifier.json"
        model.save_model(str(out/name)); class_files.append(name)
        risk_file = None
        if a.xgb_risk_bridge == "direct":
            regressor = _fit_xgb_risk(risk_arrays["train"][core],
                tables["train"]["oracle_risk"].to_numpy(dtype=np.float32)[core], a, a.seed+fold)
            oof_risk[hold_idx] = regressor.predict(risk_arrays["train"][hold_idx])
            for split in direct: direct[split].append(regressor.predict(risk_arrays[split]))
            risk_file = f"bridge_fold_{fold:02d}_risk.json"
            regressor.save_model(str(out/risk_file))
        audits.append({"fold": fold, "fit_windows": len(core), "stop_windows": len(stop),
                       "calibration_windows": len(calibration), "holdout_windows": len(hold_idx),
                       "fit_groups": sorted(set(groups[core])), "stop_groups": sorted(set(groups[stop])),
                       "calibration_groups": sorted(set(groups[calibration])),
                       "holdout_groups": sorted(set(groups[hold_idx])),
                       "classes": classes.tolist(), "temperature": temperature,
                       "calibration_soft_cross_entropy": calibration_loss,
                       "classifier_file": name, "risk_file": risk_file,
                       "early_stopping_target": "weighted_class_fraction", "sampling": sampling,
                       "outer_validation_used_for_fit_or_calibration": False, "device": device})
    probabilities = {"train": oof, **{s: np.mean(predicted[s], axis=0) for s in predicted}}
    severity = np.array([RISK_WEIGHTS[c] for c in LABELS], dtype=np.float64)
    class_risks = {s: p@severity for s, p in probabilities.items()}
    risks = class_risks if a.xgb_risk_bridge == "class_probability" else {
        "train": np.clip(oof_risk, 0, 1), **{s: np.clip(np.mean(direct[s], axis=0), 0, 1) for s in direct}}
    metrics, risk_rows, per_class = [], [], []
    for split, table in tables.items():
        prob = probabilities[split]
        _, truth_fraction, true_dominant, _ = _classifier_arrays(table)
        table["observed_risk"] = risks[split]
        table["attack_ratio"] = 1-prob[:, 0]
        for state in range(len(STATES)):
            table[f"state_{state}"] = prob[:, np.array([STATE_IDS[c] for c in LABELS]) == state].sum(1)
        detected = _predicted_attack_ids(prob, a.xgb_attack_threshold)
        table["detected_attack_label"] = [LABELS[i] for i in detected]
        table["selected_class_fraction_estimate"] = prob[np.arange(len(prob)), detected]
        table["detected_attack_confidence"] = table["selected_class_fraction_estimate"]
        table["true_dominant_label"] = [LABELS[i] for i in true_dominant]
        table["true_dominant_fraction"] = truth_fraction[np.arange(len(prob)), true_dominant]
        table["true_selected_class_fraction"] = truth_fraction[np.arange(len(prob)), detected]
        table["class_probability_risk"] = class_risks[split]
        for i, label in enumerate(LABELS): table["xgb_prob_"+label] = prob[:, i]
        if split == "test" and a.validation_only:
            table.to_csv(out/f"xgboost_{split}_windows.csv", index=False, encoding="utf-8-sig")
            continue
        metrics.append(_classification_metrics(table, prob, split, a.xgb_attack_threshold))
        pr, re, f1, support = precision_recall_fscore_support(true_dominant, prob.argmax(1),
                            labels=np.arange(len(LABELS)), zero_division=0)
        per_class.extend({"split": split, "Label": label, "Precision": pr[i], "Recall": re[i],
                          "F1": f1[i], "Window_support": int(support[i])} for i, label in enumerate(LABELS))
        for src, ids in table.groupby("source").indices.items():
            target = table["oracle_risk"].to_numpy()[ids]; observed = np.asarray(risks[split])[ids]
            risk_rows.append({"split": split, "source": str(src), "windows": len(ids),
                              "oracle_mean": float(target.mean()), "observed_mean": float(observed.mean()),
                              "risk_MAE": float(np.abs(observed-target).mean()),
                              "class_probability_risk_MAE": float(np.abs(class_risks[split][ids]-target).mean()),
                              "mean_abs_risk_composition_disagreement": float(np.abs(observed-class_risks[split][ids]).mean()),
                              "high_risk_with_attack_fraction_below_0_1": int(np.sum((observed>=a.risk_threshold)&(table.attack_ratio.to_numpy()[ids]<.1)))})
        table.to_csv(out/f"xgboost_{split}_windows.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(metrics).to_csv(out/"xgboost_classification_metrics.csv", index=False)
    pd.DataFrame(per_class).to_csv(out/"xgboost_classification_per_class.csv", index=False)
    pd.DataFrame(risk_rows).to_csv(out/"xgboost_risk_bridge_metrics.csv", index=False)
    audit = {"bridge_revision": BRIDGE_REVISION, "train_probability_mode": "OOF", "oof_folds": folds,
             "features": feature_cols, "risk_regressor_features": risk_names,
             "label_names": LABELS, "state_ids": STATE_IDS, "risk_weights": RISK_WEIGHTS,
             "risk_bridge": a.xgb_risk_bridge, "risk_metrics": risk_rows, "metrics": metrics,
             "folds": audits, "classifier_files": class_files,
             "outer_validation_used_for_fit_or_calibration": False,
             "grouping_scope": "contiguous nonoverlapping blocks within captures; offline cross-fit, not capture holdout",
             "inference": "mean of independently calibrated fold models", "risk_input_formula":
             "sum(class_fraction_estimate * severity)" if a.xgb_risk_bridge == "class_probability" else "fold risk regressors",
             "confidence_semantics": "estimated selected class fraction, not probability of any attack event"}
    dump_json(out/"xgboost_bridge_audit.json", audit)
    return audit


def bridge_groups(table, a):
    base = table["source"].astype(str)+"/"+table["series"].astype(str)
    position = table.groupby(["source", "series"], sort=False).cumcount().to_numpy()
    size = max(16, a.seq_len+a.pred_len)
    return (base+"/block_"+pd.Series(position//size, index=table.index).astype(str)).to_numpy()


def bridge_inner_partition(fit_idx, groups, seed):
    unique = np.unique(groups[fit_idx])
    if len(unique) < 3:
        raise ValueError("桥接内部划分至少需要3个不相交组")
    order = np.random.default_rng(seed).permutation(unique)
    n = max(1, len(order)//5)
    calibration_groups, stop_groups = order[:n], order[n:2*n]
    calibration = fit_idx[np.isin(groups[fit_idx], calibration_groups)]
    stop = fit_idx[np.isin(groups[fit_idx], stop_groups)]
    core = fit_idx[~np.isin(groups[fit_idx], np.r_[calibration_groups, stop_groups])]
    if not min(len(core), len(stop), len(calibration)):
        raise ValueError("桥接内部训练/早停/校准分区为空")
    return core, stop, calibration


def load_xgboost_augmented_tables(out):
    tables = {}
    for split in SPLITS:
        path = out/f"xgboost_{split}_windows.csv"
        if not path.is_file():
            raise FileNotFoundError(f"DDP主进程未生成XGBoost桥接文件：{path}")
        tables[split] = pd.read_csv(
            path, dtype={"source": "string", "series": "string", "bucket": "int64"}
        )
    return tables


@dataclass
class Segment:
    split: str
    source: str
    series: str
    frame: pd.DataFrame
    x: np.ndarray | None = None

    @property
    def risk(self):
        """真实标签构造的未来监督目标。"""
        return self.frame["risk"].to_numpy(dtype=np.float32)

    @property
    def observed_risk(self):
        """默认来自XGBoost概率；oracle模式下才等于真实标签风险。"""
        return self.frame["observed_risk"].to_numpy(dtype=np.float32)

    @property
    def phases(self):
        return self.frame[[f"state_{i}" for i in range(len(STATES))]].to_numpy(dtype=np.float32)


def make_segments(table, split, a):
    segments = []
    step = a.window_seconds * 10**9 if a.mode == "temporal" else 1
    for (source, series), g in table.groupby(["source", "series"], sort=True):
        g = g.sort_values("bucket", kind="stable")
        breaks = g["bucket"].diff().ne(step).cumsum()
        for part in breaks.unique():
            frame = g.loc[breaks == part].reset_index(drop=True)
            segments.append(Segment(split, str(source), f"{series}:{int(part)}", frame))
    return segments


def check_chronology(tables, meta):
    """检查真实时间或相对流顺序切分均无交叉和倒序。"""
    keys = {split: set(map(tuple, table[["source", "series", "bucket"]].itertuples(index=False, name=None)))
            for split, table in tables.items()}
    if keys["train"] & keys["val"] or keys["train"] & keys["test"] or keys["val"] & keys["test"]:
        raise ValueError("train/val/test含重复时序窗口")
    bases = {str(k): str(v) for k, v in meta.get("time_basis_by_source", {}).items()}
    sources = set().union(*(set(t["source"].astype(str).unique()) for t in tables.values()))
    for src in sources:
        if bases.get(src) == "row_order_proxy":
            series = set().union(*(set(t.loc[t.source.astype(str) == src, "series"].astype(str)) for t in tables.values()))
            for name in series:
                ranges = {}
                for split, table in tables.items():
                    values = table.loc[(table.source.astype(str) == src) & (table.series.astype(str) == name), "bucket"]
                    if len(values):
                        ranges[split] = (int(values.min()), int(values.max()))
                if "train" in ranges and "val" in ranges and ranges["train"][1] >= ranges["val"][0]:
                    raise ValueError(f"{src}/{name}的训练与验证顺序重叠")
                if "val" in ranges and "test" in ranges and ranges["val"][1] >= ranges["test"][0]:
                    raise ValueError(f"{src}/{name}的验证与测试顺序重叠")
                if "train" in ranges and "test" in ranges and ranges["train"][1] >= ranges["test"][0]:
                    raise ValueError(f"{src}/{name}的训练与测试顺序重叠")
        else:
            ranges = {}
            for split, table in tables.items():
                values = table.loc[table.source.astype(str) == src, "bucket"]
                if not len(values):
                    raise ValueError(f"来源{src}在{split}中没有观测")
                ranges[split] = (int(values.min()), int(values.max()))
            if not (ranges["train"][1] < ranges["val"][0]
                    and ranges["val"][1] < ranges["test"][0]):
                raise ValueError(f"来源{src}的train/val/test真实时间重叠或顺序错误")


def fit_transition(segments, smoothing):
    counts = np.full((len(STATES), len(STATES)), smoothing, dtype=np.float64)
    pairs = 0
    for seg in segments:
        phases = seg.phases.astype(np.float64)
        if len(phases) > 1:
            counts += phases[:-1].T @ phases[1:]
            pairs += len(phases) - 1
    if not pairs:
        raise ValueError("训练数据没有连续窗口，无法估计威胁状态转移")
    return (counts / counts.sum(axis=1, keepdims=True)).astype(np.float32), pairs


def causal_features(seg, transition, a):
    """缓存逐窗基础特征；多尺度统计在每个输入片段内部单独计算。"""
    frame = seg.frame
    names = [c for c in frame if c.startswith("flow_mean_")]
    parts = [frame[names].to_numpy(dtype=np.float32),
             frame[["observed_risk", "attack_ratio"]].to_numpy(dtype=np.float32),
             np.log1p(frame["count"].to_numpy(dtype=np.float32))[:, None]]
    names += ["risk", "attack_ratio", "log_count"]
    if not a.no_tep:
        phases = seg.phases
        parts += [phases, phases@transition]
        names += ["state_"+c for c in STATES]+["prior_"+c for c in STATES]
    # 窗口步长为完成流的固定条数，禁止把代理索引伪装为小时/星期。
    return np.column_stack(parts).astype(np.float32), names


def bounded_ms_features(risk, attack, scales):
    """只使用当前片段。短输入批量向量化，长输入用线性滚动算法避免平方内存。"""
    risk, attack = np.asarray(risk, dtype=np.float64), np.asarray(attack, dtype=np.float64)
    length = len(risk)
    scales = tuple(map(int, scales))
    names = [f"{name}_{scale}" for scale in scales for name in
             ("risk_mean", "risk_std", "risk_max", "risk_delta", "attack_mean", "attack_max", "attack_delta", "risk_ewm")]
    if length*length*len(scales) > 262144:
        r, a = pd.Series(risk), pd.Series(attack)
        parts = []
        for scale in scales:
            rr, aa = r.rolling(scale, min_periods=1), a.rolling(scale, min_periods=1)
            parts.extend([rr.mean().to_numpy(), rr.std(ddof=0).fillna(0).to_numpy(), rr.max().to_numpy(),
                (r-r.shift(scale).fillna(r.iloc[0])).to_numpy(), aa.mean().to_numpy(), aa.max().to_numpy(),
                (a-a.shift(scale).fillna(a.iloc[0])).to_numpy(), r.ewm(span=scale,adjust=False).mean().to_numpy()])
        return np.column_stack(parts).astype(np.float32), names
    end, start, count, previous, mask, ewm_weights = bounded_ms_plan(length, scales)
    rc = np.r_[0., np.cumsum(risk)]; r2 = np.r_[0., np.cumsum(risk*risk)]
    ac = np.r_[0., np.cumsum(attack)]
    rm = (rc[end]-rc[start])/count
    rs = np.sqrt(np.maximum(0., (r2[end]-r2[start])/count-rm*rm))
    am = (ac[end]-ac[start])/count
    rx = np.where(mask, risk[None,None,:], -np.inf).max(axis=2)
    ax = np.where(mask, attack[None,None,:], -np.inf).max(axis=2)
    rd, ad = risk[None,:]-risk[previous], attack[None,:]-attack[previous]
    ewm = np.einsum("slj,j->sl",ewm_weights,risk)
    values = np.stack([rm,rs,rx,rd,am,ax,ad,ewm],axis=2)
    return values.transpose(1,0,2).reshape(length,-1).astype(np.float32), names


@lru_cache(maxsize=8)
def bounded_ms_plan(length, scales):
    position = np.arange(length)[None,:]
    scale = np.asarray(scales)[:,None]
    start = np.maximum(0,position-scale+1)
    end = np.broadcast_to(position+1,start.shape)
    count = end-start
    previous = np.maximum(0,position-scale)
    history = np.arange(length)[None,None,:]
    mask = (history>=start[:,:,None]) & (history<end[:,:,None])
    delta = np.maximum(position[:,:,None]-history,0)
    alpha = 2./(np.asarray(scales,dtype=np.float64)+1)
    weights = alpha[:,None,None]*(1-alpha[:,None,None])**delta
    weights *= history<=position[:,:,None]
    weights[:,:,0] = (1-alpha[:,None])**position
    return end,start,count,previous,mask,weights


def configure_label_contract(meta, a):
    global LABELS, RISK_WEIGHTS, STATE_IDS
    labels = meta.get("label_names", [])
    if not labels or labels[0] != "Normal" or len(labels) != len(set(labels)):
        raise ValueError("类别契约必须唯一，且Normal位于第一项")
    weights, states = dict(RISK_WEIGHTS), dict(STATE_IDS)
    if a.risk_config:
        settings = json.loads(a.risk_config.read_text(encoding="utf-8-sig"))
        weights.update(settings.get("risk_weights", {})); states.update(settings.get("state_ids", {}))
    if set(labels)-set(weights) or set(labels)-set(states):
        raise ValueError("新增类别缺少风险权重或阶段映射，请在--risk-config中显式配置")
    if any(not np.isfinite(weights[c]) or not 0<=weights[c]<=1 or states[c] not in range(len(STATES)) for c in labels):
        raise ValueError("风险权重或阶段映射不合法")
    if weights["Normal"] != 0 or states["Normal"] != 0:
        raise ValueError("Normal风险和状态必须为0")
    LABELS = list(labels); RISK_WEIGHTS = {c:float(weights[c]) for c in labels}
    STATE_IDS = {c:int(states[c]) for c in labels}
    a.label_names, a.risk_weights, a.state_ids = LABELS, RISK_WEIGHTS, STATE_IDS


class WindowDataset:
    """按索引动态取窗口，所有派生输入只访问seq_len范围。"""
    def __init__(self, segments, transition, a, train=False):
        self.segments, self.transition, self.a = segments, transition, a
        indices = []
        self.risks, self.observed_risks, self.phases = [], [], []
        for k, segment in enumerate(segments):
            segment.x, self.base_names = causal_features(segment, transition, a)
            ends = np.arange(a.seq_len, len(segment.frame)-a.pred_len+1, dtype=np.int64)
            if len(ends): indices.append(np.column_stack([np.full(len(ends), k), ends]))
            self.risks.append(segment.risk); self.observed_risks.append(segment.observed_risk)
            self.phases.append(segment.phases)
        if not indices:
            raise ValueError(f"没有有效预测样本：每段至少需要{a.seq_len+a.pred_len}个连续窗口")
        self.indices = np.concatenate(indices)
        self.total_before_cap = len(self.indices)
        if train and a.max_train_samples and len(self.indices)>a.max_train_samples:
            chosen = np.sort(np.random.default_rng(a.seed).choice(len(self.indices), a.max_train_samples, replace=False))
            self.indices = self.indices[chosen]
        self.powers = np.stack([np.linalg.matrix_power(transition, h) for h in range(1, a.pred_len+1)])
        self.sample_weights = None
        if train and a.risk_sampler_alpha>0 and not a.no_ra:
            weights = [1.+a.risk_sampler_alpha*float(self.risks[k][end:end+a.pred_len].max()>=a.risk_threshold)
                       for k, end in self.indices]
            self.sample_weights = torch.as_tensor(weights, dtype=torch.double)
        self.mean, self.scale = None, None
        extra = [] if a.no_ms else bounded_ms_features(np.zeros(a.seq_len), np.zeros(a.seq_len), a.scales)[1]
        self.feature_names = self.base_names+extra

    def __len__(self): return len(self.indices)

    def raw_features(self, item):
        k, end = self.indices[item]
        start = end-self.a.seq_len
        base = self.segments[k].x[start:end]
        if self.a.no_ms: return base.copy()
        ms, _ = bounded_ms_features(self.observed_risks[k][start:end],
                        base[:, self.base_names.index("attack_ratio")], self.a.scales)
        return np.concatenate([base, ms], axis=1)

    def __getitem__(self, item):
        k, end = self.indices[item]
        x = self.raw_features(item)
        if self.mean is not None: x = np.clip((x-self.mean)/self.scale, -10., 10.).astype(np.float32)
        decoder = np.zeros((self.a.label_len+self.a.pred_len, 2+len(STATES)), dtype=np.float32)
        decoder[:self.a.label_len, 0] = self.observed_risks[k][end-self.a.label_len:end]
        decoder[:self.a.label_len, 1] = 1.
        if not self.a.no_tep:
            decoder[:self.a.label_len, 2:] = self.phases[k][end-self.a.label_len:end]
            decoder[self.a.label_len:, 2:] = np.einsum("i,hij->hj", self.phases[k][end-1], self.powers)
        return (np.ascontiguousarray(x), decoder, self.risks[k][end:end+self.a.pred_len].copy(),
                np.float32(self.observed_risks[k][end-1]), np.float32(self.risks[k][end-1]))


if torch is not None:
    class ProbSparseAttention(nn.Module):
        """历史编码器用非因果稀疏注意力；整个编码器只见预测起点之前的数据。"""
        def __init__(self, width, heads, factor, dropout, legacy_score=False, sampling="uniform"):
            super().__init__()
            self.heads, self.head_dim, self.factor = heads, width // heads, factor
            self.legacy_score, self.sampling = legacy_score, sampling
            self.qkv = nn.Linear(width, 3 * width)
            self.proj = nn.Linear(width, width)
            self.dropout = nn.Dropout(dropout)

        def forward(self, x):
            batch, length, width = x.shape
            q, k, v = self.qkv(x).reshape(batch, length, 3, self.heads, self.head_dim).permute(2, 0, 3, 1, 4).unbind(0)
            sample_n = min(length, self.factor * math.ceil(math.log(length + 1)))
            top_n = min(length, self.factor * math.ceil(math.log(length + 1)))
            # 训练/评估采用相同的均匀键采样，消除随机稀疏结构的分布差异。
            base = torch.div(torch.arange(sample_n, device=x.device) * length, sample_n, rounding_mode="floor")
            sample_idx = ((torch.arange(length, device=x.device)[:, None] + base[None, :]) % length
                          if self.sampling == "uniform" else torch.randint(length, (length, sample_n), device=x.device))
            # 评分用float32，防止混合精度下注意力点积溢出。
            sampled = (q.float().unsqueeze(-2) * k[:, :, sample_idx, :].float()).sum(-1)
            # paper沿用官方分母；sample_mean为算法变体，不能自动称为修复。
            sparsity = (sampled.max(-1).values - sampled.sum(-1)/length
                        if self.legacy_score else sampled.max(-1).values - sampled.mean(-1))
            top = sparsity.topk(top_n, dim=-1, sorted=False).indices
            q_top = q.gather(2, top.unsqueeze(-1).expand(-1, -1, -1, self.head_dim))
            scores = torch.matmul(q_top.float(), k.float().transpose(-2, -1)) / math.sqrt(self.head_dim)
            probs = self.dropout(scores.softmax(-1)).to(v.dtype)
            selected = torch.matmul(probs, v)
            context = v.mean(2, keepdim=True).expand(-1, -1, length, -1).clone()
            context.scatter_(2, top.unsqueeze(-1).expand_as(selected), selected)
            return self.proj(context.transpose(1, 2).reshape(batch, length, width))


    class EncoderBlock(nn.Module):
        def __init__(self, a):
            super().__init__()
            self.norm1 = nn.LayerNorm(a.d_model)
            self.norm2 = nn.LayerNorm(a.d_model)
            self.attn = ProbSparseAttention(a.d_model, a.heads, a.factor, a.dropout,
                                            a.legacy_probsparse_score or a.sparse_score == "paper",
                                            a.sparse_sampling)
            self.ff = nn.Sequential(nn.Linear(a.d_model, 4*a.d_model), nn.GELU(), nn.Dropout(a.dropout),
                                    nn.Linear(4*a.d_model, a.d_model))
            self.drop = nn.Dropout(a.dropout)

        def forward(self, x):
            x = x + self.drop(self.attn(self.norm1(x)))
            return x + self.drop(self.ff(self.norm2(x)))


    class MS_TEP_RA_Informer_Optimized(nn.Module):
        def __init__(self, feature_dim, a):
            super().__init__()
            self.pred_len, self.label_len = a.pred_len, a.label_len
            self.use_feature_gate = not a.no_feature_gate
            self.use_local_mixer = not a.no_local_mixer
            self.use_residual_head = not a.no_residual_head
            self.input_norm = nn.LayerNorm(feature_dim) if a.legacy_input_norm else nn.Identity()
            self.feature_gate = nn.Sequential(nn.Linear(feature_dim, feature_dim), nn.Sigmoid())
            self.encoder_input = nn.Linear(feature_dim, a.d_model)
            self.local_mixer = nn.Sequential(
                nn.Conv1d(a.d_model, a.d_model, 3, padding=1, groups=a.d_model),
                nn.Conv1d(a.d_model, a.d_model, 1), nn.GELU(), nn.Dropout(a.dropout))
            self.decoder_input = nn.Linear(2 + len(STATES), a.d_model)
            self.encoders = nn.ModuleList([EncoderBlock(a) for _ in range(a.encoder_layers)])
            self.distillers = nn.ModuleList([
                nn.Sequential(nn.Conv1d(a.d_model, a.d_model, 3, padding=1), nn.ELU(),
                              nn.MaxPool1d(kernel_size=3, stride=2, padding=1))
                for _ in range(a.encoder_layers - 1)])
            layer = nn.TransformerDecoderLayer(a.d_model, a.heads, 4*a.d_model, a.dropout,
                                               activation="gelu", batch_first=True, norm_first=True)
            self.decoder = nn.TransformerDecoder(layer, a.decoder_layers, norm=nn.LayerNorm(a.d_model))
            self.encoder_norm = nn.LayerNorm(a.d_model)
            self.dropout = nn.Dropout(a.dropout)
            self.output = nn.Linear(a.d_model, 1)
            # 小非零初始化：仍接近Persistence，但首批编码器即可得到梯度。
            nn.init.normal_(self.output.weight, std=.01)
            nn.init.zeros_(self.output.bias)
            self.history_head = nn.Sequential(nn.Linear(3*a.d_model, a.d_model), nn.GELU(),
                                              nn.Dropout(a.dropout), nn.Linear(a.d_model, 1))
            nn.init.normal_(self.history_head[-1].weight, std=.01)
            nn.init.zeros_(self.history_head[-1].bias)
            positions = torch.arange(a.seq_len + a.pred_len, dtype=torch.float32)[:, None]
            frequencies = torch.exp(torch.arange(0, a.d_model, 2) * (-math.log(10000.) / a.d_model))
            pe = torch.zeros(a.seq_len + a.pred_len, a.d_model)
            pe[:, 0::2], pe[:, 1::2] = torch.sin(positions*frequencies), torch.cos(positions*frequencies)
            self.register_buffer("position", pe[None], persistent=True)
            dlen = a.label_len + a.pred_len
            self.register_buffer("causal_mask", torch.triu(torch.ones(dlen, dlen, dtype=torch.bool), diagonal=1),
                                 persistent=False)

        def forward(self, x, decoder, return_aux=False):
            length = x.shape[1]
            normalized = self.input_norm(x)
            gated = normalized * (1. + self.feature_gate(normalized)) if self.use_feature_gate else normalized
            memory = self.encoder_input(gated)
            if self.use_local_mixer:
                memory = memory + self.local_mixer(memory.transpose(1, 2)).transpose(1, 2)
            memory = self.dropout(memory + self.position[:, :length])
            for i, block in enumerate(self.encoders):
                memory = block(memory)
                if i < len(self.distillers):
                    memory = self.distillers[i](memory.transpose(1, 2)).transpose(1, 2)
            memory = self.encoder_norm(memory)
            context = torch.cat([memory[:, -1], memory[:, -min(6, memory.shape[1]):].mean(1),
                                 memory.mean(1)], dim=1)
            anchor = decoder[:, self.label_len-1, 0].float()
            current_risk = anchor + self.history_head(context).squeeze(-1).float()
            dlen = decoder.shape[1]
            offset = length - (dlen - self.pred_len)
            query = self.dropout(self.decoder_input(decoder) + self.position[:, offset:offset+dlen])
            hidden = self.decoder(query, memory, tgt_mask=self.causal_mask[:dlen, :dlen])
            residual = self.output(hidden[:, -self.pred_len:]).squeeze(-1)
            if self.use_residual_head:
                # 训练使用未截断风险回归值，越界时仍能反向传播并返回有效区间。
                # 在这里clamp会令负预测梯度为0，可能造成整批输出长期锁死在0。
                # evaluate统一裁剪到[0,1]，训练损失始终约束原始预测。
                prediction = current_risk[:, None] + residual.float()
            else:
                prediction = torch.sigmoid(residual.float())
            return (prediction, current_risk) if return_aux else prediction


if torch is not None:
    class DirectMSInformer(nn.Module):
        """历史MS/TEP特征→稀疏编码器→直接12步输出，避免未来空占位解码器。

        风险滞后线性分支保留幅度，编码器补充非线性多源上下文；两者只见历史。
        """
        def __init__(self, feature_dim, a):
            super().__init__()
            self.label_len, self.pred_len = a.label_len, a.pred_len
            self.scale = a.deep_residual_scale
            self.use_tep = not a.no_tep
            self.use_local = not a.no_local_mixer
            self.use_multiscale = not a.no_ms
            self.use_residual = not a.no_residual_head
            self.input = nn.Linear(feature_dim, a.d_model)
            self.local = nn.Conv1d(a.d_model, a.d_model, 3, padding=1, groups=a.d_model)
            self.blocks = nn.ModuleList([EncoderBlock(a) for _ in range(a.encoder_layers)])
            self.norm = nn.LayerNorm(a.d_model)
            self.dropout = nn.Dropout(a.dropout)
            positions = torch.arange(a.seq_len, dtype=torch.float32)[:, None]
            frequencies = torch.exp(torch.arange(0, a.d_model, 2) * (-math.log(10000.)/a.d_model))
            pe = torch.zeros(a.seq_len, a.d_model)
            pe[:, 0::2], pe[:, 1::2] = torch.sin(positions*frequencies), torch.cos(positions*frequencies)
            self.register_buffer("position", pe[None])
            self.context_head = nn.Sequential(nn.Linear(3*a.d_model, a.d_model), nn.GELU(),
                                              nn.Dropout(a.dropout), nn.Linear(a.d_model, a.pred_len))
            self.lag_head = nn.Linear(a.label_len, a.pred_len, bias=False)
            self.phase_head = nn.Linear(a.pred_len*len(STATES), a.pred_len, bias=False) if self.use_tep else None
            nn.init.zeros_(self.context_head[-1].weight)
            nn.init.zeros_(self.context_head[-1].bias)
            nn.init.zeros_(self.lag_head.weight)
            if self.phase_head is not None:
                nn.init.zeros_(self.phase_head.weight)

        def forward(self, x, decoder):
            # x已由训练集统计量标准化；不再跨不同特征做输入LayerNorm而丢失绝对幅度。
            z = self.input(x)
            if self.use_local:
                z = z + F.gelu(self.local(z.transpose(1, 2)).transpose(1, 2))
            z = self.dropout(z + self.position[:, :x.shape[1]])
            for block in self.blocks:
                z = block(z)
            z = self.norm(z)
            context = (torch.cat([z[:, -1], z[:, -min(12, z.shape[1]):].mean(1), z.mean(1)], dim=1)
                       if self.use_multiscale else torch.cat([z[:, -1]]*3, dim=1))
            history = decoder[:, :self.label_len, 0].float()
            anchor = history[:, -1:]
            residual = self.scale*self.context_head(context).float()
            residual = residual + self.lag_head(history-anchor).float()
            if self.phase_head is not None:
                phase = decoder[:, self.label_len:, 2:].flatten(1)
                residual = residual + self.scale*self.phase_head(phase).float()
            return anchor + residual if self.use_residual else residual


def build_forecaster(feature_dim, a):
    cls = DirectMSInformer if a.forecast_method == "direct" else MS_TEP_RA_Informer_Optimized
    return cls(feature_dim, a)


def risk_loss(pred, target, anchor, a):
    pred, target, anchor = pred.float(), target.float(), anchor.float()
    if a.no_ra or a.loss_only_mse or a.loss_mode == "mse":
        return F.mse_loss(pred, target)
    if getattr(a, "loss_mode", "legacy") in ("balanced", "mse"):
        # 在全部预测对上归一化，避免稀少高风险样本的条件均值主导每个batch。
        horizon = torch.linspace(1., 1.+a.horizon_weight, target.shape[1], device=target.device)[None]
        weights = (1. + a.risk_alpha * (target >= a.risk_threshold).float()) * horizon
        error = (pred-target).square()
        if a.loss_mode == "balanced":
            error = .5 * F.smooth_l1_loss(pred, target, reduction="none", beta=a.huber_delta) + .5*error
        under = F.relu(target-pred).square() * (target >= a.risk_threshold).float()
        # 只约束未来相邻步的变化；不把有偏的上游观测当成真实历史监督值。
        trend = F.smooth_l1_loss(torch.diff(pred, dim=1), torch.diff(target, dim=1),
                                beta=a.huber_delta) if target.shape[1] > 1 else pred.sum()*0.
        return (error*weights).sum()/weights.sum() + a.underprediction_weight*under.mean() + a.trend_weight*trend
    high = (target >= a.risk_threshold).float()
    horizon = torch.linspace(1., 1.+a.horizon_weight, target.shape[1], device=target.device)[None]
    weights = (1 + a.risk_alpha*high) * horizon
    robust = F.smooth_l1_loss(pred, target, reduction="none", beta=a.huber_delta)
    weighted = (robust*weights).sum()/weights.sum()
    # 高风险真实值被低估比同幅度高估更危险；只在高风险区域施加单边项。
    under = (F.relu(target-pred).square()*high*horizon).sum()/(high*horizon).sum().clamp_min(1.)
    delta_true = torch.diff(torch.cat([anchor[:, None], target], dim=1), dim=1)
    delta_pred = torch.diff(torch.cat([anchor[:, None], pred], dim=1), dim=1)
    active = (delta_true.abs() > a.trend_epsilon).float()
    # 方向相反/平坦、或上升下降幅度不足时施加约束；首步以最后已知风险为锚。
    penalty = F.relu(delta_true.abs() - delta_true.sign()*delta_pred).square()
    trend = (penalty*active).sum()/active.sum().clamp_min(1.)
    return weighted + a.underprediction_weight*under + a.trend_weight*trend


def metric_row(truth, pred, anchor, threshold, epsilon, pred_anchor=None):
    truth, pred = np.asarray(truth, dtype=float), np.asarray(pred, dtype=float)
    err = pred-truth
    variance = np.sum((truth-truth.mean())**2)
    true_high, pred_high = truth >= threshold, pred >= threshold
    tp = np.sum(true_high & pred_high)
    ph, th = int(pred_high.sum()), int(true_high.sum())
    dt = np.diff(np.column_stack([anchor, truth]), axis=1)
    dp = np.diff(np.column_stack([anchor if pred_anchor is None else pred_anchor, pred]), axis=1)
    nonflat = np.abs(dt) > epsilon
    direction = np.where(dp > epsilon, 1, np.where(dp < -epsilon, -1, 0))
    denom = np.abs(truth)+np.abs(pred)
    return {"MAE": np.abs(err).mean(), "RMSE": np.sqrt(np.mean(err**2)),
            "SMAPE": np.mean(np.where(denom > 1e-8, 2*np.abs(err)/np.maximum(denom, 1e-8), 0.)),
            "R2": 1-np.sum(err**2)/variance if variance > 1e-12 else np.nan,
            "Direction_accuracy": np.mean(direction[nonflat] == np.sign(dt[nonflat])) if nonflat.any() else np.nan,
            "Direction_support": int(nonflat.sum()), "High_risk_MAE": np.abs(err[true_high]).mean() if th else np.nan,
            "High_risk_RMSE": np.sqrt(np.mean(err[true_high]**2)) if th else np.nan,
            "High_risk_underestimation_rate": np.mean(pred[true_high] < truth[true_high]) if th else np.nan,
            "High_risk_precision": tp/ph if ph else np.nan, "High_risk_recall": tp/th if th else np.nan,
            "High_risk_F1": 2*tp/(ph+th) if ph+th else np.nan,
            "High_risk_support": th, "Prediction_pairs": truth.size}


def setup_distributed(a):
    """torchrun环境启用NCCL DDP；普通python运行保持单进程兼容。"""
    a.rank = int(os.environ.get("RANK", "0"))
    a.local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    a.world_size = int(os.environ.get("WORLD_SIZE", "1"))
    a.distributed = a.world_size > 1
    if a.distributed:
        if torch is None or not torch.cuda.is_available():
            raise RuntimeError("torchrun多进程训练需要CUDA/NCCL环境")
        torch.cuda.set_device(a.local_rank)
        dist.init_process_group(backend="nccl", init_method="env://")


def is_main(a):
    return getattr(a, "rank", 0) == 0


def finish_distributed(a):
    if getattr(a, "distributed", False) and dist.is_initialized():
        dist.destroy_process_group()


def select_device(a):
    if torch is None:
        raise RuntimeError("缺少PyTorch。请在已有GPU版torch的服务器环境运行；--prepare-only不需要torch。")
    if getattr(a, "distributed", False):
        return torch.device(f"cuda:{a.local_rank}"), list(range(a.world_size))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if a.device == "auto" else torch.device(a.device)
    if device.type not in ("cpu", "cuda"):
        raise ValueError("本首版支持cpu/cuda")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("指定CUDA但当前环境不可用；检查GPU环境，或使用 --device cpu")
    if device.type == "cpu":
        return device, []
    count = torch.cuda.device_count()
    if not count:
        raise RuntimeError("没有检测到可用GPU")
    if a.gpu_ids == "auto":
        # 显式cuda:N表示固定单卡；auto/cuda默认使用全部可见GPU。
        gpu_ids = [device.index] if device.index is not None else list(range(count))
    else:
        try:
            gpu_ids = [int(x.strip()) for x in a.gpu_ids.split(",") if x.strip()]
        except ValueError as exc:
            raise ValueError("--gpu-ids格式错误，应为auto、0或0,1") from exc
        if not gpu_ids or len(gpu_ids) != len(set(gpu_ids)):
            raise ValueError("--gpu-ids不能为空或包含重复编号")
        if min(gpu_ids) < 0 or max(gpu_ids) >= count:
            raise ValueError(f"--gpu-ids超出范围；当前仅检测到{count}张GPU")
    primary = torch.device(f"cuda:{gpu_ids[0]}")
    torch.cuda.set_device(primary)
    return primary, gpu_ids


def unwrap_model(model):
    """去除DataParallel/DDP/torch.compile包装，检查点可跨设备复现。"""
    while True:
        if isinstance(model, (nn.DataParallel, DDP)):
            model = model.module
        elif hasattr(model, "_orig_mod"):
            model = model._orig_mod
        else:
            return model


def seed_all(seed, threads, deterministic=False):
    if deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    if torch is not None:
        torch.use_deterministic_algorithms(deterministic)
        torch.manual_seed(seed)
        torch.set_num_threads(threads)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.benchmark = not deterministic
            torch.backends.cudnn.deterministic = deterministic
            torch.backends.cuda.matmul.allow_tf32 = not deterministic
            torch.backends.cudnn.allow_tf32 = not deterministic
            if hasattr(torch, "set_float32_matmul_precision"):
                torch.set_float32_matmul_precision("highest" if deterministic else "high")


if torch is not None:
    class DistributedWeightedSampler(Sampler):
        """全局按风险权重抽样后均匀分片，避免DDP进程重复抽取同一批序列。"""
        def __init__(self, weights, replicas, rank, seed):
            self.weights = torch.as_tensor(weights, dtype=torch.double)
            self.replicas, self.rank, self.seed, self.epoch = replicas, rank, seed, 0
            self.num_samples = math.ceil(len(self.weights)/replicas)
            self.total_size = self.num_samples*replicas

        def set_epoch(self, epoch):
            self.epoch = epoch

        def __iter__(self):
            generator = torch.Generator().manual_seed(self.seed+self.epoch)
            selected = torch.multinomial(self.weights, self.total_size, replacement=True, generator=generator)
            return iter(selected[self.rank:self.total_size:self.replicas].tolist())

        def __len__(self):
            return self.num_samples


no_grad = torch.no_grad if torch is not None else (lambda: (lambda fn: fn))


class EMA:
    """仅保存可训练参数的指数滑动平均；验证和测试使用平滑权重。"""
    def __init__(self, model, decay):
        self.decay = decay
        self.shadow = {n: p.detach().clone() for n, p in unwrap_model(model).named_parameters() if p.requires_grad}
        self.backup = {}

    @no_grad()
    def update(self, model):
        for name, param in unwrap_model(model).named_parameters():
            if name in self.shadow:
                self.shadow[name].lerp_(param.detach(), 1.-self.decay)

    @no_grad()
    def apply(self, model):
        core = unwrap_model(model)
        self.backup = {n: p.detach().clone() for n, p in core.named_parameters() if n in self.shadow}
        for name, param in core.named_parameters():
            if name in self.shadow:
                param.copy_(self.shadow[name])

    @no_grad()
    def restore(self, model):
        for name, param in unwrap_model(model).named_parameters():
            if name in self.backup:
                param.copy_(self.backup[name])
        self.backup = {}


class EnergyMeter:
    """记录nvidia-smi返回的全部GPU板卡功率（含空闲/其他进程），不是模型专属能耗。"""
    def __init__(self, enabled, interval=1.0):
        self.enabled, self.interval = enabled, interval
        self.samples = []
        self.stop_event, self.thread = threading.Event(), None
        self.energy_joule, self.last_time, self.last_power = 0., None, None

    def _power(self):
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=index,power.draw,utilization.gpu,memory.used", "--format=csv,noheader,nounits"],
                check=True, capture_output=True, text=True, timeout=3)
            power = 0.
            for line in result.stdout.splitlines():
                idx, watts, util, memory = [float(v.strip()) for v in line.split(",")]
                self.samples.append({"timestamp": datetime.now().isoformat(), "gpu": int(idx),
                                     "power_W": watts, "utilization_percent": util, "memory_MiB": memory})
                power += watts
            return power
        except (OSError, ValueError, subprocess.SubprocessError):
            return None

    def _sample(self):
        now, power = time.perf_counter(), self._power()
        if power is not None and self.last_power is not None:
            self.energy_joule += .5*(power+self.last_power)*(now-self.last_time)
        if power is not None:
            self.last_time, self.last_power = now, power

    def _loop(self):
        while not self.stop_event.wait(self.interval):
            self._sample()

    def start(self):
        if self.enabled:
            self._sample()
            self.thread = threading.Thread(target=self._loop, daemon=True)
            self.thread.start()
        return self

    def stop(self):
        if not self.enabled:
            return float("nan")
        self.stop_event.set()
        if self.thread is not None:
            self.thread.join(timeout=2.)
        self._sample()
        return self.energy_joule/3_600_000.

def amp_settings(a, device):
    enabled = device.type == "cuda" and not a.no_amp
    if not enabled:
        return False, torch.float32
    if a.amp_dtype == "bf16" or (a.amp_dtype == "auto" and torch.cuda.is_bf16_supported()):
        return True, torch.bfloat16
    return True, torch.float16


def loader_for(dataset, a, device, train=False):
    sampler = None
    if train and getattr(a, "distributed", False):
        if dataset.sample_weights is not None:
            sampler = DistributedWeightedSampler(dataset.sample_weights, a.world_size, a.rank, a.seed)
        else:
            sampler = DistributedSampler(dataset, num_replicas=a.world_size, rank=a.rank,
                                         shuffle=True, seed=a.seed, drop_last=False)
    elif train and dataset.sample_weights is not None:
        sampler = WeightedRandomSampler(dataset.sample_weights, len(dataset), replacement=True,
                                        generator=torch.Generator().manual_seed(a.seed))
    kwargs = dict(dataset=dataset, batch_size=a.batch_size, shuffle=train and sampler is None,
                  sampler=sampler, num_workers=a.workers, pin_memory=device.type == "cuda",
                  persistent_workers=a.workers > 0,
                  generator=torch.Generator().manual_seed(a.seed+getattr(a, "rank", 0)))
    if a.workers > 0:
        kwargs["prefetch_factor"] = a.prefetch_factor
        if getattr(a, "distributed", False):
            kwargs["multiprocessing_context"] = "spawn"
    return torch.utils.data.DataLoader(**kwargs)


def evaluate(model, loader, device, a):
    model.eval()
    truth, preds, anchors = [], [], []
    amp, amp_dtype = amp_settings(a, device)
    with torch.inference_mode():
        for x, dec, y, last, _true_history in loader:
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp):
                output = model(x.to(device, non_blocking=True), dec.to(device, non_blocking=True))
            if not torch.isfinite(output).all():
                raise RuntimeError("模型产生非有限预测值")
            preds.append(output.float().clamp(0., 1.).cpu().numpy())
            truth.append(y.numpy())
            anchors.append(last.numpy())
    return np.concatenate(truth), np.concatenate(preds), np.concatenate(anchors)


def macro_skill_score(dataset, truth, prediction, baseline):
    """来源等权的 MAE/RMSE 相对持久性比值；1等于基线，低于1更优。"""
    sources = np.asarray([dataset.segments[k].source for k, _ in dataset.indices])
    scores = []
    for source in sorted(set(sources)):
        mask = sources == source
        error, base = prediction[mask]-truth[mask], baseline[mask]-truth[mask]
        scores.append(.5*((np.abs(error).mean()+.01)/(np.abs(base).mean()+.01)
                          + (np.sqrt(np.mean(error**2))+.01)/(np.sqrt(np.mean(base**2))+.01)))
    return float(np.mean(scores))


def train_model(model, datasets, a, out, device, signature, preprocessing):
    train_loader = loader_for(datasets["train"], a, device, True)
    val_loader = loader_for(datasets["val"], a, device) if is_main(a) else None
    optimizer = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.weight_decay, fused=device.type == "cuda")
    minimum = min(1., a.min_lr/a.lr)
    def lr_factor(step):
        if a.warmup_epochs and step < a.warmup_epochs:
            return max(minimum, (step+1)/a.warmup_epochs)
        span = max(1, a.epochs-a.warmup_epochs)
        progress = min(1., max(0., (step-a.warmup_epochs)/span))
        return minimum + (1.-minimum)*.5*(1.+math.cos(math.pi*progress))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_factor)
    amp, amp_dtype = amp_settings(a, device)
    use_scaler = amp and amp_dtype == torch.float16
    scaler = (torch.amp.GradScaler("cuda", enabled=use_scaler) if hasattr(torch.amp, "GradScaler")
              else torch.cuda.amp.GradScaler(enabled=use_scaler))
    ema = EMA(model, a.ema_decay) if a.ema_decay > 0 else None
    LOG.info("模型选择指标=%s；损失使用原始回归值，验证/推断限制到[0,1]", a.selection_metric)
    best, stale, history = float("inf"), 0, []
    for epoch in range(1, a.epochs+1):
        if hasattr(train_loader.sampler, "set_epoch"):
            train_loader.sampler.set_epoch(epoch)
        model.train()
        total, n = 0., 0
        started = time.perf_counter()
        energy_meter = EnergyMeter(is_main(a) and device.type == "cuda").start()
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        aux_total = 0.
        for batch, (x, dec, y, last, true_history) in enumerate(train_loader, 1):
            x, dec, y, last, true_history = [v.to(device, non_blocking=True) for v in (x, dec, y, last, true_history)]
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp):
                if a.forecast_method == "decoder":
                    pred, reconstructed = model(x, dec, return_aux=True)
                    auxiliary = F.mse_loss(reconstructed, true_history)
                else:
                    pred = model(x, dec)
                    auxiliary = pred.sum()*0.
                loss = risk_loss(pred, y, last, a)
                loss = loss + a.history_aux_weight*auxiliary
            if not torch.isfinite(loss):
                raise RuntimeError("训练损失非有限；可尝试 --no-amp 或降低 --lr")
            if epoch == 1 and batch == 1 and is_main(a):
                LOG.info("实际训练张量设备：参数=%s 输入=%s 输出=%s 损失=%s AMP=%s",
                         next(model.parameters()).device, x.device, pred.device, loss.device, amp_dtype)
                dump_json(out/"training_device_proof.json", {
                    "parameter_device": str(next(model.parameters()).device), "input_device": str(x.device),
                    "prediction_device": str(pred.device), "loss_device": str(loss.device),
                    "cuda_memory_allocated_bytes": torch.cuda.memory_allocated(device) if device.type == "cuda" else 0,
                    "amp_dtype": str(amp_dtype), "deterministic": a.deterministic})
            if use_scaler:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=False)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
                optimizer.step()
            if ema is not None:
                ema.update(model)
            total += loss.item()*len(y)
            aux_total += auxiliary.item()*len(y)
            n += len(y)
            if is_main(a) and batch % 100 == 0:
                LOG.info("epoch %d batch %d/%d loss %.6f", epoch, batch, len(train_loader), total/n)
        scheduler.step()
        if getattr(a, "distributed", False):
            stats = torch.tensor([total, n, aux_total], dtype=torch.float64, device=device)
            dist.all_reduce(stats, op=dist.ReduceOp.SUM)
            total, n, aux_total = stats.tolist()
        stop = False
        if is_main(a):
            if ema is not None:
                ema.apply(model)
            truth, pred, anchor = evaluate(unwrap_model(model), val_loader, device, a)
            if ema is not None:
                ema.restore(model)
            val_mse = float(np.mean((truth-pred)**2))
            val_objective = float(risk_loss(torch.from_numpy(pred), torch.from_numpy(truth),
                                            torch.from_numpy(anchor), a))
            val_mae = float(np.abs(truth-pred).mean())
            baseline = np.repeat(anchor[:, None], a.pred_len, axis=1)
            macro_score = macro_skill_score(datasets["val"], truth, pred, baseline)
            selection_score = {"mae": val_mae, "rmse": math.sqrt(val_mse),
                               "objective": val_objective, "macro_skill": macro_score}[a.selection_metric]
            elapsed = time.perf_counter()-started
            energy_kwh = energy_meter.stop()
            if energy_meter.samples:
                gpu_rows = pd.DataFrame(energy_meter.samples).assign(epoch=epoch, stage="forecast_train_and_val")
                gpu_path = out/"gpu_usage.csv"
                gpu_rows.to_csv(gpu_path, mode="a", header=not gpu_path.exists(), index=False, encoding="utf-8")
            row = {"epoch": epoch, "train_objective": total/n, "train_auxiliary_MSE": aux_total/n,
                   "val_macro_skill": macro_score, "val_objective": val_objective,
                   "val_MSE": val_mse, "val_RMSE": math.sqrt(val_mse),
                   "val_MAE": val_mae, "selection_score": selection_score,
                   "val_persistence_MAE": float(np.abs(truth-baseline).mean()),
                   "val_persistence_RMSE": float(np.sqrt(np.mean((truth-baseline)**2))),
                   "val_zero_fraction": float(np.mean(pred == 0.)),
                   "learning_rate": optimizer.param_groups[0]["lr"], "seconds": elapsed,
                   "train_samples_per_second": n/max(elapsed, 1e-9),
                   "gpu_energy_kWh": energy_kwh,
                   "peak_gpu_utilization_percent": max((v["utilization_percent"] for v in energy_meter.samples), default=float("nan")),
                   "train_samples_per_kWh": n/energy_kwh if energy_kwh > 0 else float("nan"),
                   "peak_gpu_memory_GB": (torch.cuda.max_memory_allocated(device)/1024**3
                                          if device.type == "cuda" else 0.)}
            history.append(row)
            pd.DataFrame(history).to_csv(out/"training_history.csv", index=False, encoding="utf-8-sig")
            LOG.info("epoch %d/%d | train %.6f | val objective %.6f MSE %.6f MAE %.6f | %.1fs",
                     epoch, a.epochs, row["train_objective"], val_objective, val_mse, row["val_MAE"], elapsed)
            if selection_score < best - 1e-9:
                best, stale = selection_score, 0
                if ema is not None:
                    ema.apply(model)
                state = {k: v.detach().cpu().clone() for k, v in unwrap_model(model).state_dict().items()}
                if ema is not None:
                    ema.restore(model)
                torch.save({"model": state, "signature": signature, "epoch": epoch,
                            "selection_metric": a.selection_metric, "selection_score": best,
                            "val_macro_skill": macro_score, "val_MAE": val_mae,
                            "val_objective": val_objective, "val_MSE": val_mse, "preprocessing": preprocessing,
                            "config": {k: str(v) if isinstance(v, Path) else v for k, v in vars(a).items()}},
                           out/"best_model.pt")
            else:
                stale += 1
            stop = stale >= a.patience
        if getattr(a, "distributed", False):
            flag = torch.tensor(int(stop), device=device)
            dist.broadcast(flag, src=0)
            stop = bool(flag.item())
        if stop:
            if is_main(a):
                LOG.info("正常早停：验证集%s连续%d轮未改善；接下来加载最佳模型并完成测试与输出", a.selection_metric, a.patience)
            break
    return history


def export_predictions(dataset, truth, pred, anchor, out, a, status):
    n, horizon = truth.shape
    rows = []
    for i, (k, end) in enumerate(dataset.indices):
        seg = dataset.segments[k]
        frame = seg.frame
        for h in range(horizon):
            bucket = int(frame["bucket"].iloc[end+h])
            origin_bucket = int(frame["bucket"].iloc[end-1])
            rows.append((status, seg.source, seg.series, i, h+1, origin_bucket, bucket,
                         float(truth[i, h]), float(pred[i, h]), float(anchor[i])))
    df = pd.DataFrame(rows, columns=["data_status", "source", "sequence", "forecast_id", "horizon",
                                    "origin_bucket", "target_bucket", "true_risk", "predicted_risk", "persistence"])
    if getattr(a, "time_axis_mode", "absolute_timestamp") == "absolute_timestamp":
        df["forecast_origin_time"] = pd.to_datetime(df["origin_bucket"] + a.window_seconds*10**9, unit="ns")
        df["target_window_start"] = pd.to_datetime(df["target_bucket"], unit="ns")
    elif a.time_axis_mode == "relative_flow_order":
        step = a.window_seconds*10**9
        df["forecast_origin_window"] = df["origin_bucket"] // step
        df["target_window"] = df["target_bucket"] // step
    else:
        step = a.window_seconds*10**9
        proxy = df["source"].astype(str).map(a.time_basis_by_source).eq("row_order_proxy")
        df["forecast_origin_time"], df["target_window_start"] = "", ""
        df["forecast_origin_window"], df["target_window"] = pd.NA, pd.NA
        real = ~proxy
        df.loc[real, "forecast_origin_time"] = pd.to_datetime(
            df.loc[real, "origin_bucket"] + step, unit="ns").dt.strftime("%Y-%m-%dT%H:%M:%S")
        df.loc[real, "target_window_start"] = pd.to_datetime(
            df.loc[real, "target_bucket"], unit="ns").dt.strftime("%Y-%m-%dT%H:%M:%S")
        df.loc[proxy, "forecast_origin_window"] = df.loc[proxy, "origin_bucket"] // step
        df.loc[proxy, "target_window"] = df.loc[proxy, "target_bucket"] // step
    df.to_csv(out/"test_predictions.csv", index=False, encoding="utf-8-sig")


def set_run_stage(out, stage):
    global ACTIVE_OUTPUT, ACTIVE_STAGE
    ACTIVE_OUTPUT, ACTIVE_STAGE = out, stage
    dump_json(out/"run_status.json", {"status": "running", "stage": stage,
              "updated_at": datetime.now().isoformat(), "revision": REVISION, "pid": os.getpid()})
    LOG.info("阶段：%s", stage)


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4*1024*1024), b""):
            digest.update(block)
    return digest.hexdigest()


def cached_history(tables, run, a, out):
    """只复用内容匹配、训练历史为OOF的已完成估计；未来目标继续来自原始输入。"""
    raw_hashes = {s: sha256_file(run/FILES[s]) for s in SPLITS}
    if a.history_run:
        candidates = [a.history_run.expanduser().resolve()]
    else:
        roots = [Path.cwd()/"experiment_results/MS_TEP_RA_Informer_Optimized",
                 Path(__file__).resolve().parent/"experiment_results/MS_TEP_RA_Informer_Optimized",
                 Path("/root/shared-nvme/project/experiment_results/MS_TEP_RA_Informer_Optimized")]
        candidates = sorted({p for root in roots if root.exists() for p in root.glob("run_*")}, reverse=True)
    failures = []
    for candidate in candidates:
        try:
            summary = json.loads((candidate/"run_summary.json").read_text(encoding="utf-8"))
            fingerprint = json.loads((candidate/"input_fingerprints.json").read_text(encoding="utf-8"))["sha256"]
            bridge_dir = candidate/"history_diagnostics" if (candidate/"history_diagnostics"/"xgboost_bridge_audit.json").is_file() else candidate
            audit = json.loads((bridge_dir/"xgboost_bridge_audit.json").read_text(encoding="utf-8"))
            if summary.get("status") not in {"completed", "validation_completed"} or any(fingerprint.get(s) != raw_hashes[s] for s in SPLITS):
                raise ValueError("输入SHA256不同或上游未完成")
            if audit.get("bridge_revision") != BRIDGE_REVISION or audit.get("state_ids") != STATE_IDS:
                raise ValueError("历史缓存的隔离/校准协议或状态定义不匹配，需重建")
            if audit.get("train_probability_mode") != "OOF" or audit.get("oof_folds", 0) < 2:
                raise ValueError("训练历史不是OOF估计，拒绝使用训练内概率")
            if audit.get("risk_bridge") != a.xgb_risk_bridge:
                raise ValueError("历史风险定义不同")
            old_data_audit = json.loads((bridge_dir/"data_audit.json").read_text(encoding="utf-8"))
            if old_data_audit.get("risk_weights") != RISK_WEIGHTS:
                raise ValueError("历史严重度定义不同")
            cache = load_xgboost_augmented_tables(bridge_dir)
            keys = ["source", "series", "bucket"]
            inject = ["observed_risk", "attack_ratio"] + [f"state_{i}" for i in range(len(STATES))]
            for split in SPLITS:
                current = tables[split].sort_values(keys, kind="stable").reset_index(drop=True)
                stored = cache[split].sort_values(keys, kind="stable").reset_index(drop=True)
                if len(current) != len(stored) or not np.array_equal(current[keys].astype(str), stored[keys].astype(str)):
                    raise ValueError(f"{split}窗口身份不同")
                compare = ["count", "oracle_risk"] + [c for c in current if c.startswith("flow_mean_")]
                if not np.allclose(current[compare].to_numpy(float), stored[compare].to_numpy(float), atol=2e-7, rtol=1e-7):
                    raise ValueError(f"{split}特征/监督内容不同")
                values = stored[inject].to_numpy(dtype=float)
                if not np.isfinite(values).all() or np.any(values < -1e-7) or np.any(values > 1+1e-7):
                    raise ValueError(f"{split}历史风险/状态范围无效")
                if not np.allclose(stored[[f"state_{i}" for i in range(len(STATES))]].sum(axis=1), 1., atol=1e-5):
                    raise ValueError(f"{split}历史状态概率未归一化")
                # 只导入历史可观测估计；不从缓存替换监督目标或原始流特征。
                current[inject] = values
                cache[split] = current
            hashes = {f"history_{s}": sha256_file(bridge_dir/f"xgboost_{s}_windows.csv") for s in SPLITS}
            manifest = {"mode": "reused_frozen_history", "source_run": str(candidate),
                        "raw_input_sha256": raw_hashes, "history_sha256": hashes,
                        "injected_columns": inject, "train_estimation": "OOF",
                        "oof_folds": audit["oof_folds"], "history_refitted": False,
                        "future_targets": "original temporal CSV labels; never injected as historical inputs"}
            if is_main(a):
                dump_json(out/"history_manifest.json", manifest)
                LOG.info("已固定历史输入：%s；本次只训练后续预测器", candidate)
            return cache, audit, manifest
        except (OSError, ValueError, KeyError, TypeError) as exc:
            failures.append(f"{candidate.name}: {exc}")
    raise ValueError("没有可安全复用的历史风险估计。指定 --history-run 完成的旧run，或显式 --rebuild-history。"
                     + "；".join(failures[:3]))


def forecast_arrays(dataset):
    features, target, observed, true_previous = [], [], [], []
    for i in range(len(dataset)):
        x, dec, y, anchor, _true_history = dataset[i]
        risk = dec[:dataset.a.label_len, 0]
        features.append(np.concatenate([risk-anchor, x[-1], x[-3:].mean(0), x[-12:].mean(0), x.mean(0)]))
        target.append(y)
        observed.append(anchor)
        k, end = dataset.indices[i]
        true_previous.append(dataset.risks[k][end-1])
    return (np.asarray(features), np.asarray(target), np.asarray(observed), np.asarray(true_previous))


def fit_ridge_forecast(train, val, a, out, dataset=None):
    """直接多步Ridge仅作预测基线，超参数仅根据验证MAE选择。"""
    from sklearn.linear_model import Ridge
    scale = StandardScaler().fit(train[0])
    tx, vx = scale.transform(train[0]), scale.transform(val[0])
    scores, models = [], []
    for alpha in [10., 100., 1000., 10000.]:
        model = Ridge(alpha=alpha).fit(tx, train[1]-train[2][:, None])
        prediction = np.clip(model.predict(vx)+val[2][:, None], 0., 1.)
        scores.append({"alpha": alpha, "val_MAE": float(np.abs(prediction-val[1]).mean()),
                       "val_RMSE": float(np.sqrt(np.mean((prediction-val[1])**2)))})
        models.append(model)
    for i, candidate in enumerate(models):
        prediction = np.clip(candidate.predict(vx)+val[2][:, None], 0., 1.)
        if a.selection_metric == "macro_skill":
            scores[i]["selection_score"] = macro_skill_score(dataset, val[1], prediction,
                                        np.repeat(val[2][:, None], a.pred_len, axis=1))
        elif a.selection_metric == "objective":
            scores[i]["selection_score"] = float(risk_loss(torch.from_numpy(prediction),
                                    torch.from_numpy(val[1]), torch.from_numpy(val[2]), a))
        else: scores[i]["selection_score"] = scores[i]["val_RMSE" if a.selection_metric == "rmse" else "val_MAE"]
    chosen = min(range(len(scores)), key=lambda i: scores[i]["selection_score"])
    model = models[chosen]
    dump_json(out/"ridge_validation_selection.json", {"selection_metric": a.selection_metric, "candidates": scores,
               "chosen_alpha": scores[chosen]["alpha"], "trained_on": "train_only"})
    np.savez(out/"ridge_baseline.npz", coef=model.coef_, intercept=model.intercept_,
             mean=scale.mean_, scale=scale.scale_, alpha=scores[chosen]["alpha"])
    return model, scale


def simple_forecasts(dataset):
    output = {"Persistence": [], "MovingAverage12": [], "Drift12": [], "TrainMean": []}
    horizon = np.arange(1, dataset.a.pred_len+1, dtype=np.float32)
    for k, end in dataset.indices:
        history = dataset.observed_risks[k][max(0, end-min(12, dataset.a.seq_len)):end]
        last = history[-1]
        output["Persistence"].append(np.repeat(last, dataset.a.pred_len))
        output["MovingAverage12"].append(np.repeat(history.mean(), dataset.a.pred_len))
        slope = (last-history[0])/max(1, len(history)-1)
        output["Drift12"].append(np.clip(last+slope*horizon, 0., 1.))
        output["TrainMean"].append(np.repeat(dataset.a.global_train_risk_mean,
                                                  dataset.a.pred_len))
    return {k: np.asarray(v) for k, v in output.items()}


def prediction_metrics(dataset, truth, predictions, observed, a):
    true_previous = np.asarray([dataset.risks[k][end-1] for k, end in dataset.indices])
    rows = []
    sources = np.asarray([dataset.segments[k].source for k, end in dataset.indices])
    for name, pred in predictions.items():
        for source in ["all"] + sorted(set(sources)):
            mask = np.ones(len(truth), dtype=bool) if source == "all" else sources == source
            for horizon in ["all"] + list(range(1, a.pred_len+1)):
                if horizon == "all":
                    y, pr = truth[mask], pred[mask]
                    tprev, pprev = true_previous[mask], observed[mask]
                else:
                    h = horizon-1
                    y, pr = truth[mask, h:h+1], pred[mask, h:h+1]
                    tprev = true_previous[mask] if h == 0 else truth[mask, h-1]
                    pprev = observed[mask] if h == 0 else pred[mask, h-1]
                rows.append({"model": name, "source": source, "horizon": horizon,
                             **metric_row(y, pr, tprev, a.risk_threshold, a.trend_epsilon, pprev)})
    return rows


def evidence_diagnostics(tables, dataset, truth, predictions, out, a):
    """分离上游偏差、预测增益与重叠样本相关性；不参与参数选择。"""
    bridge_rows = []
    for split, table in tables.items():
        for source, part in table.groupby("source"):
            target, observed = part["risk"].to_numpy(), part["observed_risk"].to_numpy()
            bridge_rows.append({"split": split, "source": str(source), "windows": len(part),
                               "target_mean": target.mean(), "observed_mean": observed.mean(),
                               "history_MAE": np.abs(target-observed).mean(),
                               "high_risk_windows": int(np.sum(target >= a.risk_threshold))})
    pd.DataFrame(bridge_rows).to_csv(out/"history_bias_by_source.csv", index=False)
    segment_rows, supports, blocks = [], [], []
    for k, segment in enumerate(dataset.segments):
        ids = np.flatnonzero(dataset.indices[:, 0] == k)
        if not len(ids):
            continue
        blocks.append(ids)
        ends = dataset.indices[ids, 1]
        targets = np.unique(np.concatenate([np.arange(end, end+a.pred_len) for end in ends]))
        supports.append({"source": segment.source, "sequence": segment.series, "origins": len(ids),
                         "unique_targets": len(targets),
                         "unique_high_risk_targets": int(np.sum(segment.risk[targets] >= a.risk_threshold))})
        for name, prediction in predictions.items():
            for protocol, chosen in [("all_origins", ids), ("nonoverlapping_targets", ids[::a.pred_len])]:
                error = prediction[chosen] - truth[chosen]
                segment_rows.append({"model": name, "source": segment.source, "sequence": segment.series,
                                     "protocol": protocol, "origins": len(chosen), "MAE": np.abs(error).mean(),
                                     "RMSE": np.sqrt(np.mean(error**2)), "bias": error.mean()})
    pd.DataFrame(segment_rows).to_csv(out/"segment_metrics.csv", index=False)
    # 固定既有片段的移动块bootstrap：不将12个重叠目标当作独立样本。
    # 此区间仅描述当前片段内不确定性，不能外推为新日期/新数据集的泛化区间。
    rng = np.random.default_rng(20260922)
    block_size = max(a.seq_len, a.pred_len)
    base = np.abs(predictions["Persistence"]-truth).mean(axis=1)
    deltas = {name: np.abs(pred-truth).mean(axis=1)-base for name, pred in predictions.items()
              if name != "Persistence"}
    draws = {name: [] for name in deltas}
    for _ in range(500):
        sampled = []
        for ids in blocks:
            starts = rng.integers(0, len(ids), size=math.ceil(len(ids)/block_size))
            positions = np.concatenate([(start+np.arange(block_size)) % len(ids) for start in starts])[:len(ids)]
            sampled.append(ids[positions])
        take = np.concatenate(sampled)
        for name, delta in deltas.items():
            draws[name].append(float(delta[take].mean()))
    intervals = {name: {"MAE_minus_persistence": float(delta.mean()),
                        "conditional_95pct_interval": np.quantile(draws[name], [.025,.975]).tolist()}
                 for name, delta in deltas.items()}
    selected = predictions[a.forecast_name]
    observed = predictions["Persistence"]
    diagnostic = {"risk_threshold": a.risk_threshold, "target_support": supports,
                  "bootstrap": {"method": "within_segment_circular_moving_blocks", "replicates": 500,
                                "block_length_origins": block_size, "comparisons": intervals,
                                "limitation": "conditional on existing segments; not new-source/day uncertainty"},
                  "mean_abs_correction_to_persistence": float(np.abs(selected-observed).mean()),
                  "mean_predicted_horizon_range": float(np.ptp(selected, axis=1).mean()),
                  "mean_target_horizon_range": float(np.ptp(truth, axis=1).mean()),
                  "test_status": a.test_role+"_holdout_declared_by_caller; no automatic new-scenario claim",
                  "cached_history_caveat": "grouped cross-fitting is offline; it is not forward-only online provenance",
                  "historical_auxiliary_labels_in_model_input": False}
    dump_json(out/"evaluation_evidence.json", diagnostic)


def forecast_plots(history, truth, predictions, dataset, rows, out, a):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "savefig.dpi": 180, "pdf.fonttype": 42})
    method = a.forecast_name
    pred, baseline = predictions[method], predictions["Persistence"]
    def save(fig, name):
        fig.savefig(out/f"{name}.png", bbox_inches="tight")
        fig.savefig(out/f"{name}.pdf", bbox_inches="tight")
        plt.close(fig)
    if history:
        hist = pd.DataFrame(history)
        fig, axes = plt.subplots(1, 3, figsize=(13, 3.3), constrained_layout=True)
        axes[0].plot(hist.epoch, hist.train_objective, label="Train objective")
        axes[0].set(title="Training objective", xlabel="Epoch")
        for ax, key in zip(axes[1:], ["MAE", "RMSE"]):
            ax.plot(hist.epoch, hist[f"val_{key}"], label="Forecast")
            ax.plot(hist.epoch, hist[f"val_persistence_{key}"], ls="--", label="Persistence")
            ax.set(title=f"Validation {key}", xlabel="Epoch")
            ax.legend()
        save(fig, "01_forecast_training")
    groups = sorted(set(int(k) for k, _ in dataset.indices))
    # 每段独立绘图，禁止跨来源或缺测区间连接曲线。
    for page, offset in enumerate(range(0, len(groups), 6), 1):
        selected = groups[offset:offset+6]
        fig, axes = plt.subplots(len(selected), 1, figsize=(12, 2.6*len(selected)), squeeze=False,
                                 constrained_layout=True)
        for ax, k in zip(axes[:, 0], selected):
            idx = np.flatnonzero(dataset.indices[:, 0] == k)
            seg = dataset.segments[k]
            ax.plot(truth[idx, 0], c="#222222", label="True next-window risk", lw=1.3)
            ax.plot(pred[idx, 0], c="#2166ac", label=method, lw=1.1)
            ax.plot(baseline[idx, 0], c="#d98e32", label="Persistence", alpha=.8, lw=.8)
            ax.axhline(a.risk_threshold, c="#b2182b", ls=":", lw=.8)
            ax.set(title=f"Source {seg.source} | {seg.series}", ylabel="Risk [0,1]",
                   xlabel="Ordered forecast origins within this segment", ylim=(-.02, 1.02))
            ax.legend(loc="upper right", ncol=3, fontsize=7)
        save(fig, f"02_forecast_all_segments_{page}")
    # 示例起点按片段位置固定选择，避免挑选预测好的位置。
    for source in sorted({dataset.segments[k].source for k in groups}):
        source_groups = [k for k in groups if dataset.segments[k].source == source]
        for page, offset in enumerate(range(0, len(source_groups), 4), 1):
            chosen = source_groups[offset:offset+4]
            fig, axes = plt.subplots(len(chosen), 3, figsize=(13, 3*len(chosen)), squeeze=False, constrained_layout=True)
            for j, k in enumerate(chosen):
                idx = np.flatnonzero(dataset.indices[:, 0] == k)
                for ax, q in zip(axes[j], [.2, .5, .8]):
                    i = int(idx[round((len(idx)-1)*q)])
                    end = int(dataset.indices[i, 1]); seg = dataset.segments[k]
                    hist = seg.observed_risk[max(0, end-a.label_len):end]
                    ax.plot(np.arange(-len(hist)+1, 1), hist, c="#777777", label="Observed history")
                    h = np.arange(1, a.pred_len+1)
                    ax.plot(h, truth[i], "o-", c="#222222", ms=2, label="True future")
                    ax.plot(h, pred[i], "o-", c="#2166ac", ms=2, label="Forecast")
                    ax.plot(h, baseline[i], "--", c="#d98e32", label="Persistence")
                    ax.axvline(.5, c="#888888", lw=.8)
                    ax.axhline(a.risk_threshold, c="#b2182b", ls=":", lw=.8)
                    ax.set(title=f"Source {source} | segment {k} | origin {i} ({q:.0%})",
                           xlabel="Relative windows (forecast starts at 1)", ylabel="Risk", ylim=(-.02, 1.02))
                    ax.legend(fontsize=6, ncol=2)
            safe_source = "".join(c if c.isalnum() else "_" for c in str(source))
            save(fig, f"03_multistep_source_{safe_source}_{page}")
    metrics = pd.DataFrame(rows)
    fig, axes = plt.subplots(1, 2, figsize=(12, 3.8), constrained_layout=True)
    for name in predictions:
        data = metrics[(metrics.model == name) & (metrics.source == "all") & (metrics.horizon != "all")]
        for ax, key in zip(axes, ["MAE", "RMSE"]):
            ax.plot(data.horizon.astype(int), data[key], marker="o", ms=3, label=name)
            ax.set(xlabel="Forecast horizon (windows)", ylabel=key)
            ax.legend(fontsize=7)
    save(fig, "04_multistep_errors")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    overall = metrics[(metrics.source == "all") & (metrics.horizon == "all")]
    for ax, key in zip(axes, ["MAE", "RMSE"]):
        ax.barh(overall.model, overall[key], color=["#2166ac"]+["#aaaaaa"]*(len(overall)-1))
        ax.set(xlabel=key, title="Same inputs / same targets")
    save(fig, "05_forecasting_baselines")
    fig, axes = plt.subplots(1, 2, figsize=(9, 4), constrained_layout=True)
    for source in sorted({dataset.segments[k].source for k in groups}):
        mask = np.asarray([dataset.segments[k].source == source for k, end in dataset.indices])
        y, pr = truth[mask].ravel(), pred[mask].ravel()
        take = np.linspace(0, len(y)-1, min(2500, len(y)), dtype=int)
        axes[0].scatter(y[take], pr[take], s=6, alpha=.25, label=f"Source {source}")
        axes[1].hist(pr-y, bins=35, alpha=.45, label=f"Source {source}")
    axes[0].plot([0, 1], [0, 1], "k--", lw=.8)
    axes[0].set(xlabel="True future risk", ylabel="Predicted future risk", xlim=(0,1), ylim=(0,1))
    axes[1].set(xlabel="Prediction minus truth", ylabel="Forecast pairs")
    for ax in axes: ax.legend()
    save(fig, "06_forecast_bias_by_source")


def write_forecast_report(out, summary, rows, history):
    overall = [r for r in rows if r["source"] == "all" and r["horizon"] == "all"]
    model, baseline = overall[0], next(r for r in overall if r["model"] == "Persistence")
    lines = ["# 多步态势预测结果", "", f"状态：{summary['status']}；模型：{model['model']}。",
             f"历史输入：{summary['history_mode']}；本轮高风险阈值：{summary['risk_threshold']}。", "",
             "## 预测效果（相同历史输入、相同测试目标）", "",
             "| 方法 | MAE | RMSE | R² | 高风险召回 |", "|---|---:|---:|---:|---:|"]
    for row in overall:
        recall = row['High_risk_recall']
        high_text = f"{recall:.2%}" if np.isfinite(recall) else "不适用"
        lines.append(f"| {row['model']} | {row['MAE']:.6f} | {row['RMSE']:.6f} | {row['R2']:.6f} | {high_text} |")
    lines += ["", f"相对 Persistence 的 MAE 变化：{(model['MAE']/baseline['MAE']-1)*100:+.2f}%（负值为改善）。",
              f"最佳轮次：{summary['best_epoch']}；训练实际轮次：{len(history)}。",
              f"测试高风险独立目标窗口：{summary['unique_high_risk_test_targets']}；滚动预测对数：{model['High_risk_support']}。",
              "", "## 分来源", "", "| 来源 | MAE | RMSE | 预测偏差需结合图06判断 |",
              "|---|---:|---:|---|"]
    for row in rows:
        if row['model'] == model['model'] and row['source'] != 'all' and row['horizon'] == 'all':
            lines.append(f"| {row['source']} | {row['MAE']:.6f} | {row['RMSE']:.6f} | 查看完整误差图 |")
    lines += ["", "## 预测图与明细", "",
              "- 01：训练曲线和验证 MAE/RMSE，与 Persistence 对照。",
              "- 02：所有可预测片段的一步真实/预测曲线，片段不拼接。",
              "- 03：各来源/片段在固定20%、50%、80%位置起点的完整多步预测。",
              "- 04：每个预测步长的误差，与持久性、均值、漂移和Ridge等基线对比。",
              "- 05：总体预测基线比较；06：按来源的散点和系统性偏差。",
              "- test_predictions.csv：每个起点全部未来步的预测和基线值。",
              "- test_metrics.csv：整体、各来源、各预测步长的指标。",
              "- segment_metrics.csv：分片段指标，以及预测目标互不重叠的起点子集结果。",
              "- history_bias_by_source.csv：历史风险估计与监督风险的分来源偏差。",
              "- evaluation_evidence.json：独立高风险窗口数、固定片段内块bootstrap及锚点依赖诊断。", "",
              "## 评估范围", "",
              "未来真实标签只生成监督和评估，不进入预测输入。方向指标首步的真实变化使用真实上一窗，预测变化使用可观测上一窗；该定义修正后不可直接与旧版方向指标比较。",
              "输入为主流程处理后的融合特征；步长是固定完成流条数，不能表述为分钟。无时间戳的原CSV顺序仅在显式允许后使用。",
              "滚动窗口相互重叠，高风险预测对不能当成独立事件。留出用途需结合test_role和实验记录确认；程序不自动宣称新场景泛化。",
              "历史桥接器使用分组交叉拟合和训练内部校准；这是离线协议，不宣称全过程前向在线评估。板卡能耗包含设备空闲和其他进程开销，不是模型独占能耗。",
              "本报告没有通过测试集结果选择轮次或超参数；预测方法未超过基线时应如实报告。", ""]
    (out/"00_预测结果报告.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    a = parse_args()
    if torch is None:
        raise RuntimeError("需要服务器已有的PyTorch环境")
    if a.history_run and a.rebuild_history:
        raise ValueError("--history-run 与 --rebuild-history 不能同时指定")
    if a.deep_residual_scale <= 0:
        raise ValueError("--deep-residual-scale必须为正数")
    setup_distributed(a)
    seed_all(a.seed+a.rank, a.threads, a.deterministic)
    run = resolve_run(a.data_dir)
    parent = a.output_dir or Path.cwd()/"forecast_results"/"MS_TEP_RA_Informer"
    out_text = str(parent.resolve()/datetime.now().strftime("run_%Y%m%d_%H%M%S_%f")) if is_main(a) else ""
    if a.distributed:
        shared = [out_text]; dist.broadcast_object_list(shared, src=0); out_text = shared[0]
    out = Path(out_text)
    if is_main(a): out.mkdir(parents=True, exist_ok=False)
    if a.distributed: dist.barrier()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s",
                        handlers=([logging.StreamHandler(), logging.FileHandler(out/"train_log.txt", encoding="utf-8")]
                                  if is_main(a) else [logging.NullHandler()]), force=True)
    logging.getLogger("fontTools").setLevel(logging.WARNING)
    if is_main(a): set_run_stage(out, "validate_and_freeze_history")
    meta = temporal_contract(run, a)
    tables, audit = {}, {}
    for split in SPLITS: tables[split], audit[split] = read_temporal_windows(run, split, meta)
    check_chronology(tables, meta)
    counts = {split: sum(max(0, len(seg.frame)-a.seq_len-a.pred_len+1)
                        for seg in make_segments(table, split, a)) for split, table in tables.items()}
    if any(n == 0 for n in counts.values()):
        raise ValueError(f"聚合后没有足够的连续预测窗口：{counts}；在训练前停止")
    if a.prepare_only:
        if is_main(a):
            dump_json(out/"prepared_data_audit.json", {"metadata": meta, "files": audit,
                       "predictor_or_bridge_trained": False, "forecast_samples": counts})
            dump_json(out/"run_status.json", {"status": "prepared", "stage": "complete"})
        finish_distributed(a); return
    diagnostics = out/"history_diagnostics"
    if is_main(a): diagnostics.mkdir(exist_ok=True)
    if a.risk_source == "oracle":
        manifest = {"mode": "oracle_upper_bound", "history_refitted": False}
        bridge = {"risk_input_formula": "true historical labels: oracle upper bound"}
    elif a.rebuild_history or not a.history_run:
        if is_main(a):
            set_run_stage(out, "explicit_history_rebuild")
            bridge = fit_xgboost_risk_bridge(tables, diagnostics, a)
        if a.distributed:
            dist.barrier(); tables = load_xgboost_augmented_tables(diagnostics)
            bridge = json.loads((diagnostics/"xgboost_bridge_audit.json").read_text(encoding="utf-8"))
        manifest = {"mode": "explicit_history_rebuild", "history_refitted": True,
                    "source_run": str(diagnostics), "history_sha256":
                    {f"history_{s}": sha256_file(diagnostics/f"xgboost_{s}_windows.csv") for s in SPLITS}}
    else:
        tables, bridge, manifest = cached_history(tables, run, a, out)
    if is_main(a): dump_json(out/"history_manifest.json", manifest)
    segments = {s: make_segments(tables[s], s, a) for s in SPLITS}
    a.source_names = sorted({s.source for s in segments["train"]})
    a.train_risk_means = {source: float(tables["train"].loc[tables["train"].source.astype(str)==source, "risk"].mean())
                         for source in a.source_names}
    if any(s.source not in a.source_names for split in ("val", "test") for s in segments[split]):
        raise ValueError("验证/测试出现训练中不存在的数据来源，需独立设计跨域评估")
    a.global_train_risk_mean = float(tables["train"]["risk"].mean())
    transition, pairs = ((np.eye(len(STATES), dtype=np.float32), 0)
                         if a.no_tep or a.transition_mode == "identity"
                         else fit_transition(segments["train"], a.transition_smoothing))
    datasets = {s: WindowDataset(segments[s], transition, a, train=s == "train") for s in SPLITS}
    names = datasets["train"].feature_names
    if any(d.feature_names != names for d in datasets.values()): raise ValueError("划分间特征顺序不一致")
    scaler = StandardScaler()
    for start in range(0, len(datasets["train"]), 128):
        block = np.concatenate([datasets["train"].raw_features(i)
                   for i in range(start, min(start+128, len(datasets["train"])))])
        scaler.partial_fit(block)
    for dataset in datasets.values(): dataset.mean, dataset.scale = scaler.mean_.astype(np.float32), scaler.scale_.astype(np.float32)
    preprocessing = {"feature_names": names, "mean": scaler.mean_.tolist(), "scale": scaler.scale_.tolist(),
                     "transition": transition.tolist(), "risk_weights": RISK_WEIGHTS,
                     "history_mode": "crossfit_estimated_history" if a.risk_source == "xgboost" else "oracle_diagnostic_history", "clip": [-10,10],
                     "forecast_head_output": "learned_history_correction_plus_decoder_residual",
                     "source_names": a.source_names, "auxiliary_target": "train historical label risk; loss only",
                      "raw_history_scope": "all encoder/MS inputs restricted to seq_len",
                      "data_product_type": meta["data_product_type"]}
    fingerprints = {s: sha256_file(run/FILES[s]) for s in SPLITS}
    fingerprints.update(manifest.get("history_sha256", {}))
    keys = ["forecast_method", "loss_mode", "deep_residual_scale", "seq_len", "label_len", "pred_len",
            "d_model", "heads", "encoder_layers", "decoder_layers", "factor", "dropout", "scales",
            "no_ms", "no_tep", "no_ra", "no_time_features", "no_feature_gate", "no_local_mixer", "no_residual_head",
            "legacy_probsparse_score", "risk_source", "risk_threshold", "risk_alpha", "huber_delta",
            "trend_weight", "underprediction_weight", "seed", "horizon_weight", "history_aux_weight",
            "no_source_features", "no_source_balance", "legacy_input_norm",
            "sparse_score", "sparse_sampling", "loss_only_mse", "transition_mode"]
    signature = hashlib.sha256(json.dumps({"revision": REVISION, "inputs": fingerprints, "preprocessing": preprocessing,
                               "meta": meta, "script_sha256": sha256_file(Path(__file__)), "settings": {k: getattr(a,k) for k in keys}}, sort_keys=True).encode()).hexdigest()
    a.forecast_name = "MS_TEP_RA_Informer_Direct" if a.forecast_method == "direct" else "MS_TEP_RA_Informer_Decoder"
    if is_main(a):
        dump_json(out/"config.json", vars(a)); dump_json(out/"preprocessing.json", preprocessing)
        dump_json(out/"input_fingerprints.json", {"sha256": fingerprints, "model_data_signature": signature})
        dump_json(diagnostics/"data_audit.json", {"temporal_metadata": meta, "risk_weights": RISK_WEIGHTS,
                  "history": manifest, "input_run": str(run), "train_transition_pairs": pairs,
                  "forecast_samples": {s:len(datasets[s]) for s in SPLITS},
                  "chronology_checked": True, "future_labels_in_input": False})
    if a.prepare_only:
        if is_main(a): dump_json(out/"run_status.json", {"status": "prepared", "stage": "complete"})
        finish_distributed(a); return
    device, gpu_ids = select_device(a)
    if is_main(a):
        set_run_stage(out, "forecast_training")
        dump_json(out/"runtime_environment.json", {"python": sys.executable, "torch": str(torch.__version__),
                  "cuda_runtime": torch.version.cuda, "device": str(device), "gpu_ids": gpu_ids,
                  "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                  "energy_scope": "all boards returned by nvidia-smi, includes idle and other processes",
                  "script_sha256": sha256_file(Path(__file__)), "deterministic": a.deterministic})
    model = build_forecaster(len(names), a).to(device)
    LOG.info("预测模型=%s 参数量=%d；设备=%s；预测窗口=%s", a.forecast_name,
             sum(p.numel() for p in model.parameters()), device, {s:len(datasets[s]) for s in SPLITS})
    if a.distributed:
        model = DDP(model, device_ids=[a.local_rank], output_device=a.local_rank,
                    broadcast_buffers=False, find_unused_parameters=True)
    elif len(gpu_ids) > 1: model = nn.DataParallel(model, device_ids=gpu_ids)
    if a.compile: model = torch.compile(model, mode="reduce-overhead")
    if a.checkpoint:
        checkpoint_path = a.checkpoint.resolve()
        history_file = checkpoint_path.parent/"training_history.csv"
        history = pd.read_csv(history_file).to_dict("records") if history_file.exists() else []
        if history and is_main(a):
            pd.DataFrame(history).to_csv(out/"training_history.csv", index=False, encoding="utf-8-sig")
    else:
        history = train_model(model, datasets, a, out, device, signature, preprocessing)
        checkpoint_path = out/"best_model.pt"
    if a.distributed:
        dist.barrier()
        if not is_main(a): finish_distributed(a); return
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if checkpoint.get("signature") != signature: raise ValueError("检查点的输入/历史估计/预测配置不匹配")
    unwrap_model(model).load_state_dict(checkpoint["model"])
    set_run_stage(out, "validation_and_forecasting_baselines")
    train_arrays, val_arrays = forecast_arrays(datasets["train"]), forecast_arrays(datasets["val"])
    ridge, ridge_scale = fit_ridge_forecast(train_arrays, val_arrays, a, out, datasets["val"])
    val_y, val_pred, val_anchor = evaluate(unwrap_model(model), loader_for(datasets["val"], a, device), device, a)
    val_predictions = {a.forecast_name: val_pred, **simple_forecasts(datasets["val"])}
    val_predictions["Ridge_MultiStep"] = np.clip(ridge.predict(ridge_scale.transform(val_arrays[0]))+val_anchor[:,None],0,1)
    val_rows = prediction_metrics(datasets["val"], val_y, val_predictions, val_anchor, a)
    dump_json(out/"validation_metrics.json", val_rows)
    if a.validation_only:
        dump_json(out/"run_summary.json", {"status": "validation_completed", "best_epoch": checkpoint["epoch"],
                  "selection_metric": a.selection_metric, "selection_score": checkpoint["selection_score"],
                  "history_mode": manifest["mode"], "metrics": [r for r in val_rows if r["source"] == "all" and r["horizon"] == "all"],
                  "test_evaluated": False})
        dump_json(out/"run_status.json", {"status": "validation_completed", "stage": "complete"})
        finish_distributed(a); return
    set_run_stage(out, "test_multistep_forecasting")
    truth, pred, observed = evaluate(unwrap_model(model), loader_for(datasets["test"], a, device), device, a)
    predictions = {a.forecast_name: pred, **simple_forecasts(datasets["test"])}
    test_arrays = forecast_arrays(datasets["test"])
    predictions["Ridge_MultiStep"] = np.clip(ridge.predict(ridge_scale.transform(test_arrays[0]))+observed[:,None],0,1)
    rows = prediction_metrics(datasets["test"], truth, predictions, observed, a)
    pd.DataFrame(rows).to_csv(out/"test_metrics.csv", index=False, encoding="utf-8-sig")
    dump_json(out/"test_metrics.json", rows)
    export_predictions(datasets["test"], truth, pred, observed, out, a,
                       f"{manifest['mode'].upper()}_MULTISTEP_{a.time_axis_mode.upper()}")
    frame = pd.read_csv(out/"test_predictions.csv")
    for name, prediction in predictions.items(): frame[f"prediction_{name}"] = prediction.ravel()
    frame["true_previous_risk"] = np.repeat(test_arrays[3], a.pred_len)
    frame.to_csv(out/"test_predictions.csv", index=False, encoding="utf-8-sig")
    unique_high = frame[frame.true_risk >= a.risk_threshold].drop_duplicates(["source", "sequence", "target_bucket"])
    evidence_diagnostics(tables, datasets["test"], truth, predictions, out, a)
    set_run_stage(out, "forecast_figures_and_report")
    forecast_plots(history, truth, predictions, datasets["test"], rows, out, a)
    overall = [r for r in rows if r["source"] == "all" and r["horizon"] == "all"]
    summary = {"status": "completed", "revision": REVISION, "history_mode": manifest["mode"],
               "input_run": str(run), "output_dir": str(out), "history_run": manifest.get("source_run"),
               "forecast_method": a.forecast_method, "risk_threshold": a.risk_threshold,
               "best_epoch": checkpoint["epoch"], "best_val_MAE": checkpoint.get("val_MAE"),
               "selection_metric": a.selection_metric, "selection_score": checkpoint["selection_score"],
               "epochs_run": len(history), "parameters": sum(v.numel() for v in model.parameters()),
               "evaluation_only": bool(a.checkpoint), "checkpoint_source": str(checkpoint_path),
               "unique_high_risk_test_targets": len(unique_high), "metrics": overall,
               "test_zero_fraction": float(np.mean(pred == 0)), "device": str(device), "gpu_ids": gpu_ids,
               "forecast_samples": {s:len(datasets[s]) for s in SPLITS},
               "test_role": a.test_role, "has_order_proxy": meta.get("has_order_proxy", False),
                "shared_split_identity": meta["shared_split_identity"],
                "data_product_type": meta["data_product_type"],
                "direction_definition": "true change from true history; predicted change from observed history"}
    write_forecast_report(out, summary, rows, history)
    dump_json(out/"run_summary.json", summary)
    dump_json(out/"run_status.json", {"status": "completed", "stage": "complete", "updated_at":datetime.now().isoformat()})
    for row in overall: LOG.info("%s MAE=%.6f RMSE=%.6f", row["model"], row["MAE"], row["RMSE"])
    LOG.info("多步预测全部完成：%s；先查看00_预测结果报告.md", out)
    finish_distributed(a)
    print(f"\n多步预测完成：{out}\n主报告：{out/'00_预测结果报告.md'}")


if __name__ == "__main__":
    try:
        main()
    except BaseException as exc:
        if isinstance(exc, SystemExit) and exc.code in (0, None):
            raise
        logging.exception("运行未完成；失败阶段=%s", ACTIVE_STAGE)
        if ACTIVE_OUTPUT is not None:
            failure = {"status": "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed",
                       "stage": ACTIVE_STAGE, "exception": repr(exc), "traceback": traceback.format_exc(),
                       "revision": REVISION, "updated_at": datetime.now().isoformat()}
            failure_path = ("run_status.json" if int(os.environ.get("RANK", "0")) == 0
                            else f"failure_rank{os.environ.get('RANK')}.json")
            dump_json(ACTIVE_OUTPUT/failure_path, failure)
        sys.exit(130 if isinstance(exc, KeyboardInterrupt) else 1)
