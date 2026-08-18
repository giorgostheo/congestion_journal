import heapq
import math
import os
import pickle
import re
import time
import warnings
from collections import Counter, defaultdict
from datetime import datetime
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import f1_score, precision_score, recall_score
from tqdm.auto import tqdm

warnings.simplefilter(action='ignore', category=(FutureWarning, UserWarning))

METHOD_DIR = Path(__file__).resolve().parent
DATA_DIR = METHOD_DIR.parent / 'data'

DATASET = os.environ.get('DATASET', 'pems')

_DATASET_CONFIGS = {
    'pems': {'data_file': 'PEMS-BAY.csv',   'adj_file': 'adj_mx_bay.pkl'},
    'metr': {'data_file': 'METR-LA.csv',    'adj_file': 'adj_mx_METR-LA.pkl'},
}

if DATASET not in _DATASET_CONFIGS:
    raise ValueError(
        f'Unknown DATASET={DATASET!r}. Valid options: {sorted(_DATASET_CONFIGS)}.'
    )

DATASET_DIR = DATA_DIR / DATASET
RESULTS_DIR = METHOD_DIR / 'results' / DATASET
FEATURE_IMPORTANCE_DIR = RESULTS_DIR / 'feature_importance'
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
FEATURE_IMPORTANCE_DIR.mkdir(parents=True, exist_ok=True)
RUN_TIMESTAMP = os.environ.get('RUN_TIMESTAMP') or datetime.now().strftime('%Y%m%d_%H%M%S_%f')
OUTPUT_DIR = METHOD_DIR.parent / 'results'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TARGET_SENSOR_IDS_BY_DATASET = {
    'pems': [
        "400104",
        "403404",
        "400654",
        "400065",
        "400414",
        "400995",
        "407373",
        "400832",
        "401906",
        "401541"
    ],
    'metr': [
        "717513",
        "717825",
        "717816",
        "716953",
        "769867",
        "717493",
        "717595",
        "718204",
        "717504",
        "767609"
    ],
}


def dataset_data_path():
    return DATASET_DIR / _DATASET_CONFIGS[DATASET]['data_file']


def dataset_sensor_columns():
    df = pd.read_csv(dataset_data_path(), nrows=0)
    columns = list(df.columns)
    if columns and columns[0] in ('Unnamed: 0', 'timestamp'):
        columns = columns[1:]
    return [str(column) for column in columns]


def dataset_target_locations():
    """Return target sensors for the active dataset.

    Each experiment trains target-specific XGBoost models and reports errors only
    for the explicit subset in TARGET_SENSOR_IDS_BY_DATASET.
    """
    target_sensor_ids = [str(sensor_id) for sensor_id in TARGET_SENSOR_IDS_BY_DATASET[DATASET]]
    available_sensor_ids = set(dataset_sensor_columns())
    missing_sensor_ids = [sensor_id for sensor_id in target_sensor_ids if sensor_id not in available_sensor_ids]
    if missing_sensor_ids:
        raise ValueError(
            f'TARGET_SENSOR_IDS_BY_DATASET[{DATASET!r}] includes sensors not present '
            f'in {_DATASET_CONFIGS[DATASET]["data_file"]}: {missing_sensor_ids}'
        )
    return [[sensor_id] for sensor_id in target_sensor_ids]


LOCATIONS = dataset_target_locations()

SAMPLING_RATE_MINUTES = 5
HORIZONS_MINUTES = [10, 20, 30, 40, 50, 60]
ROLLING_WINDOWS_MINUTES = [15, 30, 60]
ROLLING_MIN_WINDOWS_MINUTES = [30]
SINGLE_SENSOR_LAG_MINUTES = [5, 10, 15, 30, 60, 180]
SINGLE_SENSOR_TRANSITION_WINDOWS_MINUTES = [10, 20, 30]
DELTA_1_STEPS = 1
DELTA_5_STEPS = 5
EARLY_STOPPING_ROUNDS = 50
MAX_BOOST_ROUNDS = 2000
RANDOM_SEED = 42

MODEL_SEARCH_GRID = {
    'max_depth': [3, 5, 7],
    'eta': [0.01, 0.03, 0.05],
    'subsample': [0.6, 0.8],
    'colsample_bytree': [0.6, 0.8],
    'min_child_weight': [1],
    'gamma': [0],
    'reg_lambda': [0, 1],
    'reg_alpha': [0],
}

SEARCH_STAGE_GROUPS = [
    ('max_depth', 'eta'),
    ('subsample', 'colsample_bytree'),
    ('min_child_weight', 'gamma'),
    ('reg_lambda', 'reg_alpha'),
]

DEFAULT_DEVICE = 'cuda' if os.environ.get('CUDA_VISIBLE_DEVICES') not in (None, '', '-1') else 'cpu'
TRSP_WINDOW = int(os.environ.get('TRSP_WINDOW', '1'))
TRSP_TRANSITION_WEIGHT = float(os.environ.get('TRSP_TRANSITION_WEIGHT', '0.5'))
XGB_SELECTION_METRIC = os.environ.get('XGB_SELECTION_METRIC', 'trsp').strip().lower()
VALID_SELECTION_METRICS = {'f1', 'transition_f1', 'trsp'}
if XGB_SELECTION_METRIC not in VALID_SELECTION_METRICS:
    raise ValueError(
        f'Unknown XGB_SELECTION_METRIC={XGB_SELECTION_METRIC!r}. '
        f'Valid options: {sorted(VALID_SELECTION_METRICS)}.'
    )


def parse_float_list_env(name, default):
    raw = os.environ.get(name)
    if not raw:
        return default
    values = []
    for item in raw.split(','):
        item = item.strip()
        if item:
            values.append(float(item))
    if not values:
        raise ValueError(f'{name} must contain at least one numeric value.')
    return values


XGB_TRANSITION_WEIGHTS = parse_float_list_env('XGB_TRANSITION_WEIGHTS', [1.0, 3.0])


def _parse_flag_env(name, default='1'):
    return os.environ.get(name, default).strip().lower() not in ('0', 'false', 'no', 'off')


# Feature families can be disabled individually for ablation experiments.
FEATURE_FLAGS = {
    'profile': _parse_flag_env('XGB_FEAT_PROFILE'),        # time-of-week congestion probability (train-split table)
    'state': _parse_flag_env('XGB_FEAT_STATE'),            # time-in-state / episode-duration counters
    'margin': _parse_flag_env('XGB_FEAT_MARGIN'),          # speed margin to the congestion threshold (train-split q90)
    'wave': _parse_flag_env('XGB_FEAT_WAVE'),              # wave-aligned upstream spatial lags (global model only)
    'prof_resid': _parse_flag_env('XGB_FEAT_PROF_RESID'),  # speed residual vs time-of-week speed profile
    'naive': _parse_flag_env('XGB_FEAT_NAIVE'),            # yesterday/last-week congestion labels as features
    'ema': _parse_flag_env('XGB_FEAT_EMA'),                # exponential moving averages of speed
    'holiday': _parse_flag_env('XGB_FEAT_HOLIDAY'),        # public-holiday calendar flag
}

# Must match the 70% train share used by prepare_train_eval_test_split and
# build_prefilter_context: every train-split-derived feature table uses these rows only.
TRAIN_FRACTION = 0.7
PROFILE_BIN_MINUTES = 15
EMA_HALFLIFE_MINUTES = [15, 60]
CONGESTION_THRESHOLD_TAU = 0.65
CONGESTION_QUANTILE = 0.9
WAVE_STEPS_PER_HOP = int(os.environ.get('XGB_WAVE_STEPS_PER_HOP', '1'))

# Data safeguards used by the reported experiments.
FIX_FLAGS = {
    # Derive the congestion threshold from the training period only.
    'leak_threshold': _parse_flag_env('XGB_FIX_LEAK_THRESHOLD'),
    # Treat exact-zero speeds as missing observations.
    'zero_missing': _parse_flag_env('XGB_FIX_ZERO_MISSING'),
    # Remove rows without a complete future target window.
    'tail_drop': _parse_flag_env('XGB_FIX_TAIL_DROP'),
}


def describe_fix_flags():
    return ', '.join(f'{name}={"on" if enabled else "off"}' for name, enabled in sorted(FIX_FLAGS.items()))

# Public holidays inside each dataset's collection window (California).
DATASET_HOLIDAYS = {
    'pems': [  # PEMS-BAY: Jan-Jun 2017
        '2017-01-01', '2017-01-02', '2017-01-16', '2017-02-20', '2017-03-31', '2017-05-29',
    ],
    'metr': [  # METR-LA: Mar-Jun 2012
        '2012-03-31', '2012-05-28',
    ],
}


def describe_feature_flags():
    return ', '.join(f'{name}={"on" if enabled else "off"}' for name, enabled in sorted(FEATURE_FLAGS.items()))


EXPERIMENT_NAME = os.environ.get('EXPERIMENT_NAME', 'paper').strip()


RESULT_COLUMNS = [
    'sid',
    'horizon_min',
    'f1',
    'transition_f1',
    'trsp',
    'precision',
    'recall',
    'inference_us',
    'n_features',
    'n_boost_rounds',
]

SUMMARY_RESULT_COLUMNS = [
    'dataset',
    'experiment',
    'method',
    'horizon_min',
    'f1_mean',
    'transition_f1_mean',
    'trsp',
    'prec_mean',
    'rec_mean',
    'inference_us_all_nodes',
    'inference_us_per_node',
]

SUMMARY_TT_COLUMNS = [
    'dataset',
    'experiment',
    'method',
    'tt_mins',
]

time_feature_names = frozenset({
    'day_of_week_sin', 'day_of_week_cos',
    'hour_of_day_sin', 'hour_of_day_cos',
    'minute_of_day_sin', 'minute_of_day_cos',
    'is_weekend', 'rushhour', 'is_holiday',
})


def save_pickle(obj, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'wb') as f:
        pickle.dump(obj, f)


def load_pickle(path):
    with open(path, 'rb') as f:
        return pickle.load(f)


def append_result_row(path, row):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    header = not path.exists()
    pd.DataFrame([row], columns=RESULT_COLUMNS).to_csv(path, mode='a', header=header, index=False)


def append_specialized_result_row(path, row, result_columns):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    header = not path.exists()
    pd.DataFrame([row], columns=result_columns).to_csv(path, mode='a', header=header, index=False)


def timestamped_output_path(directory, filename):
    path = Path(filename)
    return Path(directory) / f'{path.stem}_{RUN_TIMESTAMP}{path.suffix}'


SUMMARY_RESULTS_PATH = Path(os.environ.get(
    'XGB_SUMMARY_RESULTS_PATH',
    timestamped_output_path(OUTPUT_DIR, 'results.csv'),
))
SUMMARY_TT_PATH = Path(os.environ.get(
    'XGB_SUMMARY_TT_PATH',
    timestamped_output_path(OUTPUT_DIR, 'training_times.csv'),
))


def append_csv_row(path, row, columns):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    header = not path.exists()
    pd.DataFrame([row], columns=columns).to_csv(path, mode='a', header=header, index=False)


def safe_nanmean(values):
    values = np.asarray(values, dtype=float)
    finite_values = values[~np.isnan(values)]
    if len(finite_values) == 0:
        return np.nan
    return float(np.mean(finite_values))


def aggregate_metric_entries(entries):
    return {
        'f1_mean': float(np.mean([entry['f1'] for entry in entries])),
        'transition_f1_mean': safe_nanmean([entry['transition_f1'] for entry in entries]),
        'trsp': safe_nanmean([entry['trsp'] for entry in entries]),
        'prec_mean': float(np.mean([entry['precision'] for entry in entries])),
        'rec_mean': float(np.mean([entry['recall'] for entry in entries])),
        'inference_us_all_nodes': float(np.sum([entry['inference_us'] for entry in entries])),
        'inference_us_per_node': float(np.mean([entry['inference_us'] for entry in entries])),
    }


def append_summary_result_row(method, horizon_min, entries):
    row = {
        'dataset': DATASET,
        'experiment': EXPERIMENT_NAME,
        'method': method,
        'horizon_min': horizon_min,
        **aggregate_metric_entries(entries),
    }
    append_csv_row(SUMMARY_RESULTS_PATH, row, SUMMARY_RESULT_COLUMNS)
    return row


def append_training_time_row(method, elapsed_seconds):
    row = {
        'dataset': DATASET,
        'experiment': EXPERIMENT_NAME,
        'method': method,
        'tt_mins': round(float(elapsed_seconds) / 60, 2),
    }
    append_csv_row(SUMMARY_TT_PATH, row, SUMMARY_TT_COLUMNS)
    return row


def normalize_feature_group(feature_name, sensor_ids=None):
    feature_name = str(feature_name)
    sensor_id_set = {str(sensor_id) for sensor_id in sensor_ids} if sensor_ids is not None else set()

    if feature_name in sensor_id_set:
        return 'sensor_current_speed'

    spatial_match = re.fullmatch(r'(spd|cong|trans_on|trans_off|wspd|wcong)_h(\d+)_(\d+)', feature_name)
    if spatial_match:
        metric, hop, _ = spatial_match.groups()
        return f'spatial_{metric}_h{hop}'

    if sensor_id_set:
        for sensor_id in sensor_id_set:
            suffix = f'_{sensor_id}'
            if feature_name.endswith(suffix):
                prefix = feature_name[:-len(suffix)]
                return f'sensor_{prefix}'

    numeric_suffix_match = re.fullmatch(r'(.+)_\d+', feature_name)
    if numeric_suffix_match and feature_name.startswith(('lag_', 'roll_', 'delta_', 'acceleration_')):
        return f'sensor_{numeric_suffix_match.group(1)}'

    if re.fullmatch(r'\d+', feature_name):
        return 'sensor_current_speed'

    return feature_name


def collect_grouped_feature_importance(rows, variant, horizon_min, model, feature_columns, sensor_ids=None, importance_type='gain'):
    raw_importance = model.get_score(importance_type=importance_type)
    grouped_values = defaultdict(list)
    for feature_name in feature_columns:
        if feature_name not in raw_importance:
            continue
        group = normalize_feature_group(feature_name, sensor_ids=sensor_ids)
        grouped_values[group].append(float(raw_importance[feature_name]))

    for feature_group, values in grouped_values.items():
        rows.append({
            'variant': variant,
            'horizon_min': horizon_min,
            'importance_type': importance_type,
            'feature_group': feature_group,
            'sum_importance': float(np.sum(values)),
            'n_columns': len(values),
            'model_mean_importance': float(np.mean(values)),
        })


def collect_grouped_selection_counts(rows, variant, horizon_min, selected_columns, sensor_ids=None):
    grouped_counts = Counter(
        normalize_feature_group(feature_name, sensor_ids=sensor_ids)
        for feature_name in selected_columns
    )
    for feature_group, selected_count in grouped_counts.items():
        rows.append({
            'variant': variant,
            'horizon_min': horizon_min,
            'feature_group': feature_group,
            'selected_count': int(selected_count),
        })


def write_aggregated_feature_importance(path, importance_rows, selection_rows=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = [
        'variant',
        'horizon_min',
        'importance_type',
        'feature_group',
        'mean_importance',
        'sum_importance',
        'n_columns',
        'n_models_with_feature',
        'selected_count',
        'selected_count_norm',
    ]

    if not importance_rows:
        pd.DataFrame(columns=columns).to_csv(path, index=False)
        return

    df = pd.DataFrame(importance_rows)
    grouped = df.groupby(['variant', 'horizon_min', 'importance_type', 'feature_group'], as_index=False).agg(
        sum_importance=('sum_importance', 'sum'),
        n_columns=('n_columns', 'sum'),
        n_models_with_feature=('model_mean_importance', 'count'),
    )
    grouped['mean_importance'] = grouped['sum_importance'] / grouped['n_columns']

    if selection_rows:
        selection_df = pd.DataFrame(selection_rows)
        selection_grouped = selection_df.groupby(['variant', 'horizon_min', 'feature_group'], as_index=False).agg(
            selected_count=('selected_count', 'sum'),
        )
        selection_grouped['selected_count_norm'] = selection_grouped.groupby(
            ['variant', 'horizon_min']
        )['selected_count'].transform(lambda values: values / values.max() if values.max() else 0.0)
        grouped = grouped.merge(selection_grouped, on=['variant', 'horizon_min', 'feature_group'], how='left')
    else:
        grouped['selected_count'] = np.nan
        grouped['selected_count_norm'] = np.nan

    grouped['selected_count'] = grouped['selected_count'].fillna(0).astype(int)
    grouped['selected_count_norm'] = grouped['selected_count_norm'].fillna(0.0)
    grouped = grouped[columns].sort_values(
        ['variant', 'horizon_min', 'importance_type', 'mean_importance'],
        ascending=[True, True, True, False],
    )
    grouped.to_csv(path, index=False)


def load_adjacency(base_dir=None):
    base_dir = DATASET_DIR if base_dir is None else Path(base_dir)
    adj_path = base_dir / _DATASET_CONFIGS[DATASET]['adj_file']
    if not adj_path.exists():
        print(f'WARNING: Adjacency file not found at {adj_path}, using identity (no spatial features)')
        return None, None, None
    try:
        with open(adj_path, 'rb') as f:
            sensor_ids, sensor_id_to_ind, adj = pickle.load(f, encoding='latin1')
    except Exception as e:
        print(f'WARNING: Failed to load adjacency ({e}), using identity (no spatial features)')
        return None, None, None
    adj_arr = np.array(adj)
    return sensor_ids, sensor_id_to_ind, adj_arr


def compute_hop_sets(sensor_ids, sensor_id_to_ind, adj_arr, k_max=3):
    n = len(sensor_ids)
    adj_bin = (adj_arr > 0).astype(int)
    hop_sets = {}
    for idx in range(n):
        visited = {idx: 0}
        queue = [(idx, 0)]
        while queue:
            node, d = queue.pop(0)
            for nb in np.where(adj_bin[node] > 0)[0]:
                if nb not in visited and d + 1 <= k_max:
                    visited[nb] = d + 1
                    queue.append((nb, d + 1))
        hop_sets[idx] = visited
    return hop_sets


def compute_upstream_hop_sets(sensor_ids, sensor_id_to_ind, adj_arr, k_max=3):
    """BFS in the upstream direction.

    adj_arr[i, j] > 0 encodes i -> j, so upstream neighbors of `node` are found
    by traversing column `node` of the adjacency matrix.
    """
    n = len(sensor_ids)
    adj_bin = (adj_arr > 0).astype(int)
    hop_sets = {}
    for idx in range(n):
        visited = {idx: 0}
        queue = [(idx, 0)]
        while queue:
            node, d = queue.pop(0)
            for nb in np.where(adj_bin[:, node] > 0)[0]:
                if nb not in visited and d + 1 <= k_max:
                    visited[nb] = d + 1
                    queue.append((nb, d + 1))
        hop_sets[idx] = visited
    return hop_sets


def minutes_to_steps(minutes):
    return max(1, int(round(minutes / SAMPLING_RATE_MINUTES)))


def add_time_features(df):
    enriched = df.copy()
    tm = pd.to_datetime(enriched.index)
    day_of_week = tm.day_of_week.to_numpy()
    hour_of_day = tm.hour.to_numpy()
    minute_of_day = (tm.hour * 60 + tm.minute).to_numpy()

    enriched['day_of_week_sin'] = np.sin(2 * np.pi * day_of_week / 7)
    enriched['day_of_week_cos'] = np.cos(2 * np.pi * day_of_week / 7)
    enriched['hour_of_day_sin'] = np.sin(2 * np.pi * hour_of_day / 24)
    enriched['hour_of_day_cos'] = np.cos(2 * np.pi * hour_of_day / 24)
    enriched['minute_of_day_sin'] = np.sin(2 * np.pi * minute_of_day / 1440)
    enriched['minute_of_day_cos'] = np.cos(2 * np.pi * minute_of_day / 1440)
    enriched['is_weekend'] = np.isin(day_of_week, [5, 6]).astype(int)
    enriched['rushhour'] = np.isin(hour_of_day, [7, 8, 9, 16, 17, 18]).astype(int)
    if FEATURE_FLAGS['holiday']:
        holiday_dates = pd.to_datetime(DATASET_HOLIDAYS.get(DATASET, []))
        enriched['is_holiday'] = tm.normalize().isin(holiday_dates).astype(int)
    return enriched


def shift_columns(df, steps, prefix):
    return pd.DataFrame(
        {f'{prefix}_{col}': df[col].shift(steps) for col in df.columns},
        index=df.index,
    )


def rolling_columns(df, steps, prefix, reducer):
    if reducer == 'mean':
        rolled = df.rolling(window=steps, min_periods=steps).mean()
    elif reducer == 'std':
        rolled = df.rolling(window=steps, min_periods=steps).std()
    elif reducer == 'min':
        rolled = df.rolling(window=steps, min_periods=steps).min()
    else:
        raise ValueError(f'Unsupported reducer: {reducer}')

    rolled.columns = [f'{prefix}_{col}' for col in rolled.columns]
    return rolled


def rate_of_change_columns(df):
    delta_1 = df.diff(DELTA_1_STEPS)
    delta_1.columns = [f'delta_1_{col}' for col in delta_1.columns]

    delta_5 = df.diff(DELTA_5_STEPS)
    delta_5.columns = [f'delta_5_{col}' for col in delta_5.columns]

    acceleration = df.diff(DELTA_1_STEPS).diff(DELTA_1_STEPS)
    acceleration.columns = [f'acceleration_{col}' for col in acceleration.columns]

    return [delta_1, delta_5, acceleration]


def build_spatial_aggregation_features(speed_df, congestion_df, hop_sets):
    """Build graph-informed spatial features using k-hop neighbor aggregation."""
    if hop_sets is None:
        return pd.DataFrame(index=speed_df.index)

    all_sensor_ids = list(speed_df.columns)
    n = len(all_sensor_ids)
    spatial_frames = []

    for idx in range(n):
        sid = all_sensor_ids[idx]
        if idx not in hop_sets:
            continue
        target_hops = hop_sets[idx]

        sensor_features = {}
        for hop_dist in [1, 2, 3]:
            neighbor_indices = [(nb_idx, dist) for nb_idx, dist in target_hops.items()
                                if dist > 0 and nb_idx < n]
            neighbors_at_hop = [(nb_i, d) for nb_i, d in neighbor_indices if d == hop_dist]
            if not neighbors_at_hop:
                continue

            nb_ids = [all_sensor_ids[nb_i] for nb_i, _ in neighbors_at_hop]
            weights = np.array([1.0 / d for _, d in neighbors_at_hop], dtype=float)

            speed_block = speed_df[nb_ids].values
            cong_block = congestion_df[nb_ids].values if all(c in congestion_df.columns for c in nb_ids) else None

            weight_row = weights.reshape(1, -1)
            weight_sum = weights.sum()
            weighted_speed_mean = (speed_block * weight_row).sum(axis=1) / weight_sum
            sensor_features[f'spd_h{hop_dist}'] = pd.Series(weighted_speed_mean, index=speed_df.index)

            if cong_block is not None:
                weighted_cong = (cong_block * weight_row).sum(axis=1) / weight_sum
                sensor_features[f'cong_h{hop_dist}'] = pd.Series(weighted_cong, index=speed_df.index)

        if sensor_features:
            frame = pd.DataFrame(sensor_features, index=speed_df.index)
            frame.columns = [f'{col}_{idx}' for col in frame.columns]
            spatial_frames.append(frame)

    if not spatial_frames:
        return pd.DataFrame(index=speed_df.index)

    return pd.concat(spatial_frames, axis=1)


def find_best_threshold(y_true, y_pred_prob, current_labels=None, metric='f1'):
    thresholds = np.arange(0.02, 0.98, 0.02)
    best_threshold = 0.99
    best_score = -1
    best_f1 = 0
    best_transition_f1 = np.nan
    best_rec = 0
    best_prec = 0
    for threshold in thresholds:
        try:
            y_pred = (y_pred_prob >= threshold).astype(int)
            prec = precision_score(y_true, y_pred, zero_division=0)
            rec = recall_score(y_true, y_pred, zero_division=0)
            f1 = f1_score(y_true, y_pred, zero_division=0)
            transition_f1 = compute_transition_f1(y_true, y_pred, current_labels)
            if metric == 'transition_f1':
                score = transition_f1 if not np.isnan(transition_f1) else f1
            elif metric == 'trsp':
                score = compute_trsp(y_true, y_pred)
            else:
                score = f1
            if score > best_score:
                best_score = score
                best_threshold = threshold
                best_f1 = f1
                best_transition_f1 = transition_f1
                best_rec = rec
                best_prec = prec
        except Exception:
            pass
    return best_threshold, best_f1, best_prec, best_rec, best_transition_f1, best_score


def predict_with_best_iteration(model, dmatrix):
    best_iteration = getattr(model, 'best_iteration', None)
    if best_iteration is not None and best_iteration >= 0:
        return model.predict(dmatrix, iteration_range=(0, best_iteration + 1))
    return model.predict(dmatrix)


def evaluate_model(model, X_eval, y_eval, current_eval_labels=None, selection_metric='f1'):
    deval = xgb.DMatrix(data=X_eval, label=y_eval)
    y_pred_prob = predict_with_best_iteration(model, deval)
    return find_best_threshold(
        y_eval.values,
        y_pred_prob,
        current_labels=current_eval_labels,
        metric=selection_metric,
    )


def labels_for_index(label_series, index):
    return label_series.reindex(index).fillna(0).astype(int)


def compute_transition_f1(y_true, y_pred, current_labels):
    """F1 on records where current label differs from the true future target."""
    if current_labels is None:
        return np.nan

    y_true_arr = np.asarray(y_true, dtype=int)
    y_pred_arr = np.asarray(y_pred, dtype=int)
    current_arr = np.asarray(current_labels, dtype=int)
    changed_mask = current_arr != y_true_arr
    if not np.any(changed_mask):
        return np.nan
    return f1_score(y_true_arr[changed_mask], y_pred_arr[changed_mask], zero_division=0)


def _region_f1_or_acc(y_true, y_pred):
    if len(y_true) == 0:
        return 1.0
    if len(np.unique(y_true)) < 2:
        return float(np.mean(np.asarray(y_true) == np.asarray(y_pred)))
    return float(f1_score(y_true, y_pred, zero_division=0))


def trsp_score(y_true, y_pred, window=TRSP_WINDOW, transition_weight=TRSP_TRANSITION_WEIGHT):
    """Score transition-band recall and stable-region precision with a weighted harmonic mean."""
    y_true_arr = np.asarray(y_true, dtype=int)
    y_pred_arr = np.asarray(y_pred, dtype=int)
    if len(y_true_arr) != len(y_pred_arr):
        raise ValueError('TRSP requires y_true and y_pred to have the same length.')
    if len(y_true_arr) == 0:
        return np.nan, np.nan, np.nan

    window = max(0, int(window))
    transition_weight = float(np.clip(transition_weight, 0.0, 1.0))

    transition_mask = np.zeros(len(y_true_arr), dtype=bool)
    change_points = np.flatnonzero(np.diff(y_true_arr) != 0) + 1
    for change_idx in change_points:
        start = max(0, change_idx - window)
        stop = min(len(y_true_arr), change_idx + window + 1)
        transition_mask[start:stop] = True

    tr = _region_f1_or_acc(y_true_arr[transition_mask], y_pred_arr[transition_mask])
    sp = _region_f1_or_acc(y_true_arr[~transition_mask], y_pred_arr[~transition_mask])

    if transition_weight <= 0:
        return sp, tr, sp
    if transition_weight >= 1:
        return tr, tr, sp
    if tr <= 0 or sp <= 0:
        return 0.0, tr, sp
    score = 1.0 / ((transition_weight / tr) + ((1.0 - transition_weight) / sp))
    return float(score), tr, sp


def compute_trsp(y_true, y_pred):
    score, _, _ = trsp_score(y_true, y_pred)
    return score


def time_series_validation_split(X_train, y_train, validation_fraction=0.1):
    if len(X_train) < 2:
        raise ValueError('Need at least two training rows for time-based validation splitting.')

    split_idx = int(len(X_train) * (1 - validation_fraction))
    split_idx = min(max(split_idx, 1), len(X_train) - 1)

    return (
        X_train.iloc[:split_idx],
        X_train.iloc[split_idx:],
        y_train.iloc[:split_idx],
        y_train.iloc[split_idx:],
    )


def compute_scale_pos_weight(y):
    y_array = np.asarray(y)
    pos_count = float(np.sum(y_array == 1))
    neg_count = float(np.sum(y_array == 0))
    if pos_count == 0 or neg_count == 0:
        return 1.0
    return neg_count / pos_count


def is_device_error(error):
    message = str(error).lower()
    tokens = ['cuda', 'gpu', 'device', 'nvidia', 'visible device']
    return any(token in message for token in tokens)


def train_booster(params, dtrain, num_boost_round, evals=None, early_stopping_rounds=None):
    global DEFAULT_DEVICE

    evals = [] if evals is None else evals
    candidate_devices = [DEFAULT_DEVICE]
    if DEFAULT_DEVICE != 'cpu':
        candidate_devices.append('cpu')

    last_error = None
    for device in candidate_devices:
        train_params = params.copy()
        train_params['device'] = device
        train_params.setdefault('tree_method', 'hist')

        try:
            booster = xgb.train(
                params=train_params,
                dtrain=dtrain,
                num_boost_round=num_boost_round,
                evals=evals,
                early_stopping_rounds=early_stopping_rounds,
                verbose_eval=False,
            )
            DEFAULT_DEVICE = device
            return booster
        except xgb.core.XGBoostError as error:
            last_error = error
            if device == 'cpu' or not is_device_error(error):
                raise
            print(f'XGBoost device={device} unavailable ({error}); retrying on CPU...')

    if last_error is not None:
        raise last_error

    raise RuntimeError('XGBoost training failed without producing an error object.')


def build_xgb_params(candidate_params, scale_pos_weight):
    params = {
        'objective': 'binary:logistic',
        'eval_metric': 'logloss',
        'seed': RANDOM_SEED,
        'scale_pos_weight': scale_pos_weight,
    }
    params.update(candidate_params)
    return params


def train_final_model(best_params, X_train, y_train):
    dtrain = xgb.DMatrix(data=X_train, label=y_train)
    scale_pos_weight = compute_scale_pos_weight(y_train)

    final_params = build_xgb_params(
        {key: value for key, value in best_params.items() if key != 'num_boost_round'},
        scale_pos_weight=scale_pos_weight,
    )

    num_boost_round = max(1, int(best_params['num_boost_round']))
    return train_booster(final_params, dtrain, num_boost_round=num_boost_round)


def train_final_model_weighted(best_params, X_train, y_train, sample_weight):
    dtrain = xgb.DMatrix(data=X_train, label=y_train, weight=sample_weight)
    scale_pos_weight = compute_scale_pos_weight(y_train)

    final_params = build_xgb_params(
        {key: value for key, value in best_params.items() if key != 'num_boost_round'},
        scale_pos_weight=scale_pos_weight,
    )

    num_boost_round = max(1, int(best_params['num_boost_round']))
    return train_booster(final_params, dtrain, num_boost_round=num_boost_round)


def transition_sample_weights(current_labels, targets, transition_weight):
    current_arr = np.asarray(current_labels, dtype=int)
    target_arr = np.asarray(targets, dtype=int)
    return pd.Series(
        np.where(current_arr != target_arr, transition_weight, 1.0),
        index=targets.index,
        name='sample_weight',
    )


def selection_metric_for_weight(transition_weight):
    return XGB_SELECTION_METRIC


def iter_stage_param_combinations(best_candidate, param_grid, stage_keys):
    values = [param_grid[key] for key in stage_keys]
    for stage_values in product(*values):
        candidate = best_candidate.copy()
        candidate.update(dict(zip(stage_keys, stage_values)))
        yield candidate


def evaluate_candidate_params(
    candidate_params,
    dtrain,
    dval,
    X_eval,
    y_eval,
    scale_pos_weight,
    current_eval_labels=None,
    selection_metric='f1',
):
    params = build_xgb_params(candidate_params, scale_pos_weight=scale_pos_weight)
    model = train_booster(
        params=params,
        dtrain=dtrain,
        num_boost_round=MAX_BOOST_ROUNDS,
        evals=[(dtrain, 'train'), (dval, 'validation')],
        early_stopping_rounds=EARLY_STOPPING_ROUNDS,
    )

    rounds = evaluate_model(
        model,
        X_eval,
        y_eval,
        current_eval_labels=current_eval_labels,
        selection_metric=selection_metric,
    )
    best_iteration = getattr(model, 'best_iteration', None)
    num_boost_round = MAX_BOOST_ROUNDS if best_iteration is None else best_iteration + 1
    return model, num_boost_round, rounds


def evaluate_candidate_params_weighted(
    candidate_params,
    dtrain,
    dval,
    X_eval,
    y_eval,
    scale_pos_weight,
    max_boost_rounds,
    early_stopping_rounds,
    current_eval_labels=None,
    selection_metric='f1',
):
    params = build_xgb_params(candidate_params, scale_pos_weight=scale_pos_weight)
    model = train_booster(
        params=params,
        dtrain=dtrain,
        num_boost_round=max_boost_rounds,
        evals=[(dtrain, 'train'), (dval, 'validation')],
        early_stopping_rounds=early_stopping_rounds,
    )

    rounds = evaluate_model(
        model,
        X_eval,
        y_eval,
        current_eval_labels=current_eval_labels,
        selection_metric=selection_metric,
    )
    best_iteration = getattr(model, 'best_iteration', None)
    num_boost_round = max_boost_rounds if best_iteration is None else best_iteration + 1
    return model, num_boost_round, rounds


def find_best_xgboost_model(param_grid, X_train, y_train, X_eval, y_eval):
    X_inner_train, X_val, y_inner_train, y_val = time_series_validation_split(X_train, y_train, validation_fraction=0.1)

    dtrain = xgb.DMatrix(data=X_inner_train, label=y_inner_train)
    dval = xgb.DMatrix(data=X_val, label=y_val)
    scale_pos_weight = compute_scale_pos_weight(y_inner_train)

    best_candidate = {key: values[0] for key, values in param_grid.items()}
    best_f1 = -1
    best_rounds = None
    best_num_boost_round = None

    for stage_keys in SEARCH_STAGE_GROUPS:
        stage_best_candidate = best_candidate.copy()
        stage_best_rounds = best_rounds
        stage_best_num_boost_round = best_num_boost_round
        stage_best_f1 = best_f1

        for candidate_params in iter_stage_param_combinations(best_candidate, param_grid, stage_keys):
            model, num_boost_round, rounds = evaluate_candidate_params(
                candidate_params,
                dtrain,
                dval,
                X_eval,
                y_eval,
                scale_pos_weight,
            )
            f1 = rounds[1]

            if f1 > stage_best_f1:
                stage_best_f1 = f1
                stage_best_candidate = candidate_params.copy()
                stage_best_rounds = rounds
                stage_best_num_boost_round = num_boost_round

        best_candidate = stage_best_candidate
        best_rounds = stage_best_rounds
        best_num_boost_round = stage_best_num_boost_round
        best_f1 = stage_best_f1

    if best_num_boost_round is None:
        raise RuntimeError('Hyperparameter search did not produce a valid XGBoost model.')

    best_params = best_candidate.copy()
    best_params['num_boost_round'] = best_num_boost_round
    final_model = train_final_model(best_params, X_train, y_train)
    return final_model, best_params, best_rounds


def find_best_xgboost_model_weighted(
    param_grid,
    X_train,
    y_train,
    sample_weight,
    X_eval,
    y_eval,
    search_stage_groups,
    max_boost_rounds,
    early_stopping_rounds,
    current_eval_labels=None,
    selection_metric='f1',
):
    X_inner_train, X_val, y_inner_train, y_val = time_series_validation_split(
        X_train,
        y_train,
        validation_fraction=0.1,
    )
    weight_inner_train = sample_weight.loc[X_inner_train.index]

    dtrain = xgb.DMatrix(data=X_inner_train, label=y_inner_train, weight=weight_inner_train)
    dval = xgb.DMatrix(data=X_val, label=y_val)
    scale_pos_weight = compute_scale_pos_weight(y_inner_train)

    best_candidate = {key: values[0] for key, values in param_grid.items()}
    best_score = -1
    best_rounds = None
    best_num_boost_round = None

    for stage_keys in active_search_stage_groups(param_grid, search_stage_groups):
        stage_best_candidate = best_candidate.copy()
        stage_best_rounds = best_rounds
        stage_best_num_boost_round = best_num_boost_round
        stage_best_score = best_score

        for candidate_params in iter_stage_param_combinations(best_candidate, param_grid, stage_keys):
            model, num_boost_round, rounds = evaluate_candidate_params_weighted(
                candidate_params,
                dtrain,
                dval,
                X_eval,
                y_eval,
                scale_pos_weight,
                max_boost_rounds,
                early_stopping_rounds,
                current_eval_labels=current_eval_labels,
                selection_metric=selection_metric,
            )
            score = rounds[-1]

            if score > stage_best_score:
                stage_best_score = score
                stage_best_candidate = candidate_params.copy()
                stage_best_rounds = rounds
                stage_best_num_boost_round = num_boost_round

        best_candidate = stage_best_candidate
        best_rounds = stage_best_rounds
        best_num_boost_round = stage_best_num_boost_round
        best_score = stage_best_score

    if best_num_boost_round is None:
        model, best_num_boost_round, best_rounds = evaluate_candidate_params_weighted(
            best_candidate,
            dtrain,
            dval,
            X_eval,
            y_eval,
            scale_pos_weight,
            max_boost_rounds,
            early_stopping_rounds,
            current_eval_labels=current_eval_labels,
            selection_metric=selection_metric,
        )
        del model

    best_params = best_candidate.copy()
    best_params['num_boost_round'] = best_num_boost_round
    final_model = train_final_model_weighted(best_params, X_train, y_train, sample_weight)
    return final_model, best_params, best_rounds


def select_top_features(df, importance_map, nfeat):
    if nfeat <= 0:
        raise ValueError('nfeat must be positive.')

    ranked = heapq.nlargest(min(nfeat, len(importance_map)), importance_map, key=importance_map.get)
    if len(ranked) < nfeat:
        remaining = [column for column in df.columns if column not in ranked]
        ranked.extend(remaining[: max(0, nfeat - len(ranked))])

    if not ranked:
        ranked = list(df.columns[: min(nfeat, len(df.columns))])

    return ranked


def validate_spatial_sensor_alignment(speed_df, sensor_id_to_ind):
    """Raise ValueError if speed_df column order doesn't match adjacency matrix sensor order."""
    adj_ordered = sorted(sensor_id_to_ind, key=sensor_id_to_ind.get)
    csv_sensors = [str(c) for c in speed_df.columns]
    adj_sensors = [str(s) for s in adj_ordered]

    adj_set = set(adj_sensors)
    csv_set = set(csv_sensors)
    if adj_set != csv_set:
        missing_from_csv = adj_set - csv_set
        missing_from_adj = csv_set - adj_set
        raise ValueError(
            f'Sensor sets differ between CSV and adjacency matrix. '
            f'Missing from CSV: {missing_from_csv or "none"}. '
            f'Missing from adj: {missing_from_adj or "none"}.'
        )

    for i, (csv_sid, adj_sid) in enumerate(zip(csv_sensors, adj_sensors)):
        if csv_sid != adj_sid:
            raise ValueError(
                f'Sensor ordering mismatch at position {i}: CSV has {csv_sid!r} but '
                f'adjacency matrix has {adj_sid!r}. hop_sets is indexed by adjacency '
                f'position, so spatial features would be silently computed for the wrong '
                f'sensors. Ensure CSV columns are in the same order as sensor_ids in '
                f'the adjacency file for the active dataset ({DATASET}).'
            )


def build_engineered_feature_frame(speed_features, congestion_features, horizon_min, sensor_id_to_ind, hop_sets):
    """Build features from all sensors and graph-based spatial aggregates."""
    feature_frames = [speed_features.copy()]

    lag_minutes = sorted({
        horizon_min, 5, 10, 15, 30, 60, 180,
        12 * 60,
        24 * 60 - horizon_min,
        2 * 24 * 60 - horizon_min,
        7 * 24 * 60 - horizon_min,
    })

    for lag_min in lag_minutes:
        lag_steps = minutes_to_steps(lag_min)
        feature_frames.append(shift_columns(speed_features, lag_steps, f'lag_{lag_min}min'))

    for window_min in ROLLING_WINDOWS_MINUTES:
        window_steps = minutes_to_steps(window_min)
        feature_frames.append(rolling_columns(speed_features, window_steps, f'roll_mean_{window_min}min', 'mean'))
        feature_frames.append(rolling_columns(speed_features, window_steps, f'roll_std_{window_min}min', 'std'))

    for window_min in ROLLING_MIN_WINDOWS_MINUTES:
        window_steps = minutes_to_steps(window_min)
        feature_frames.append(rolling_columns(speed_features, window_steps, f'roll_min_{window_min}min', 'min'))

    feature_frames.extend(rate_of_change_columns(speed_features))

    if sensor_id_to_ind is not None and hop_sets is not None:
        spatial = build_spatial_aggregation_features(speed_features, congestion_features, hop_sets)
        if len(spatial.columns) > 0:
            feature_frames.append(spatial)

    feature_frame = pd.concat(feature_frames, axis=1)
    feature_frame = add_time_features(feature_frame)
    return feature_frame


def _train_row_count(n_rows):
    return int(n_rows * TRAIN_FRACTION)


def time_of_week_minutes(index):
    tm = pd.to_datetime(index)
    return tm.day_of_week.to_numpy() * 1440 + tm.hour.to_numpy() * 60 + tm.minute.to_numpy()


def _bin_time_of_week(time_of_week):
    return (time_of_week // PROFILE_BIN_MINUTES) * PROFILE_BIN_MINUTES


def _time_of_week_train_table(series, bins):
    """Mean of `series` per time-of-week bin, computed on the train split only (no leakage)."""
    n_train = _train_row_count(len(series))
    train_values = pd.Series(series.to_numpy()[:n_train], dtype=float)
    return train_values.groupby(pd.Series(bins[:n_train])).mean()


def congestion_profile_features(congestion_series, horizon_min, sid):
    """Empirical congestion probability per time-of-week bin, at t and at the target time t+h.

    The bin->probability table is the recurring-pattern signal that the persistence
    baselines (Naive-Yesterday and Naive-LastWeek) exploit, exposed as a single O(1) lookup.
    """
    time_of_week = time_of_week_minutes(congestion_series.index)
    bins = _bin_time_of_week(time_of_week)
    table = _time_of_week_train_table(congestion_series, bins)
    target_bins = _bin_time_of_week((time_of_week + horizon_min) % (7 * 1440))
    index = congestion_series.index
    return pd.DataFrame({
        f'tow_cong_prob_{sid}': pd.Series(bins, index=index).map(table),
        f'tow_cong_prob_t{horizon_min}_{sid}': pd.Series(target_bins, index=index).map(table),
    }, index=index)


def speed_profile_residual_features(speed_series, sid):
    """Current speed minus the time-of-week mean speed (train-split table)."""
    time_of_week = time_of_week_minutes(speed_series.index)
    bins = _bin_time_of_week(time_of_week)
    table = _time_of_week_train_table(speed_series, bins)
    expected = pd.Series(bins, index=speed_series.index).map(table)
    return pd.DataFrame(
        {f'prof_resid_{sid}': speed_series.astype(float) - expected},
        index=speed_series.index,
    )


def threshold_margin_features(speed_series, sid):
    """Signed speed margin to the congestion threshold (tau * train-split q90).

    The label threshold itself is defined in load_speed_data; this feature recomputes it
    from train rows only so the feature stays leakage-free even though the labels use
    the full series.
    """
    n_train = _train_row_count(len(speed_series))
    threshold = speed_series.iloc[:n_train].quantile(CONGESTION_QUANTILE) * CONGESTION_THRESHOLD_TAU
    columns = {f'margin_{sid}': speed_series.astype(float) - threshold}
    if threshold > 0:
        columns[f'rel_margin_{sid}'] = speed_series.astype(float) / threshold
    return pd.DataFrame(columns, index=speed_series.index)


def time_in_state_features(congestion_series, sid):
    """Backward-looking congestion state-machine counters.

    mins_in_state: minutes since the current binary state began.
    mins_since_onset: minutes since the most recent 0->1 flip (NaN before the first).
    prev_episode_mins: duration of the most recently COMPLETED congestion episode
    (NaN before the first completed episode); the ongoing episode is excluded.
    """
    state = congestion_series.fillna(0).astype(int).to_numpy()
    n = len(state)
    index = congestion_series.index
    if n == 0:
        return pd.DataFrame(index=index)

    pos = np.arange(n)
    is_run_start = np.r_[True, state[1:] != state[:-1]]
    run_starts = pos[is_run_start]
    run_id = np.cumsum(is_run_start) - 1
    mins_in_state = (pos - run_starts[run_id]) * SAMPLING_RATE_MINUTES

    is_onset = np.r_[False, (state[1:] == 1) & (state[:-1] == 0)]
    onset_marks = np.where(is_onset, pos, -1)
    last_onset = np.maximum.accumulate(onset_marks)
    mins_since_onset = np.where(last_onset >= 0, (pos - last_onset) * SAMPLING_RATE_MINUTES, np.nan)

    run_values = state[run_starts]
    run_lengths = np.r_[run_starts[1:], n] - run_starts
    prev_episode = np.full(len(run_starts), np.nan)
    last_completed_minutes = np.nan
    for i in range(len(run_starts)):
        prev_episode[i] = last_completed_minutes
        if run_values[i] == 1 and run_starts[i] + run_lengths[i] < n:
            last_completed_minutes = run_lengths[i] * SAMPLING_RATE_MINUTES

    return pd.DataFrame({
        f'mins_in_state_{sid}': mins_in_state.astype(float),
        f'mins_since_onset_{sid}': mins_since_onset,
        f'prev_episode_mins_{sid}': prev_episode[run_id],
    }, index=index)


def naive_label_features(congestion_series, horizon_min, sid):
    """The Naive-Yesterday and Naive-LastWeek predictions as binary features."""
    horizon_steps = minutes_to_steps(horizon_min)
    yesterday_steps = minutes_to_steps(24 * 60) - horizon_steps
    last_week_steps = minutes_to_steps(7 * 24 * 60) - horizon_steps
    return pd.DataFrame({
        f'naive_yesterday_t{horizon_min}_{sid}': congestion_series.shift(yesterday_steps),
        f'naive_lastweek_t{horizon_min}_{sid}': congestion_series.shift(last_week_steps),
    }, index=congestion_series.index)


def ema_features(speed_series, sid):
    """Exponential moving averages of speed (O(1) streaming state on the edge)."""
    columns = {}
    for halflife_min in EMA_HALFLIFE_MINUTES:
        steps = minutes_to_steps(halflife_min)
        columns[f'ema_{halflife_min}min_{sid}'] = speed_series.ewm(halflife=steps, min_periods=steps).mean()
    return pd.DataFrame(columns, index=speed_series.index)


def build_extra_sensor_features(speed_series, congestion_series, horizon_min, sid):
    """Build the configurable feature blocks for one sensor.

    Train-split-derived tables (profile, margin, residual) use the first TRAIN_FRACTION
    rows only; everything else is backward-looking, so all blocks are leakage-free.
    """
    frames = []
    if congestion_series is not None:
        if FEATURE_FLAGS['profile']:
            frames.append(congestion_profile_features(congestion_series, horizon_min, sid))
        if FEATURE_FLAGS['state']:
            frames.append(time_in_state_features(congestion_series, sid))
        if FEATURE_FLAGS['naive']:
            frames.append(naive_label_features(congestion_series, horizon_min, sid))
    if FEATURE_FLAGS['margin']:
        frames.append(threshold_margin_features(speed_series, sid))
    if FEATURE_FLAGS['prof_resid']:
        frames.append(speed_profile_residual_features(speed_series, sid))
    if FEATURE_FLAGS['ema']:
        frames.append(ema_features(speed_series, sid))
    if not frames:
        return pd.DataFrame(index=speed_series.index)
    return pd.concat(frames, axis=1)


def build_single_sensor_features(speed_series, horizon_min, congestion_series=None):
    """Build lag, rolling, delta, and time features for one target sensor only."""
    df = speed_series.to_frame()
    frames = [df.copy()]
    if congestion_series is None:
        lag_minutes = sorted({
            horizon_min, 5, 10, 15, 30, 60, 180,
            12 * 60, 24 * 60 - horizon_min, 2 * 24 * 60 - horizon_min, 7 * 24 * 60 - horizon_min,
        })
    else:
        lag_minutes = sorted({horizon_min, *SINGLE_SENSOR_LAG_MINUTES})
    for lag_min in lag_minutes:
        frames.append(shift_columns(df, minutes_to_steps(lag_min), f'lag_{lag_min}min'))
    for w in ROLLING_WINDOWS_MINUTES:
        frames.append(rolling_columns(df, minutes_to_steps(w), f'roll_mean_{w}min', 'mean'))
        frames.append(rolling_columns(df, minutes_to_steps(w), f'roll_std_{w}min', 'std'))
    for w in ROLLING_MIN_WINDOWS_MINUTES:
        frames.append(rolling_columns(df, minutes_to_steps(w), f'roll_min_{w}min', 'min'))
    frames.extend(rate_of_change_columns(df))

    if congestion_series is not None:
        cdf = congestion_series.to_frame()
        cdf.columns = [f'cong_{congestion_series.name}']
        frames.append(cdf.copy())
        for w in ROLLING_WINDOWS_MINUTES:
            rolled = cdf.rolling(window=minutes_to_steps(w), min_periods=minutes_to_steps(w)).mean()
            rolled.columns = [f'roll_mean_{w}min_{cdf.columns[0]}']
            frames.append(rolled)
        sid_name = congestion_series.name
        for w_min in sorted({horizon_min, *SINGLE_SENSOR_TRANSITION_WINDOWS_MINUTES}):
            steps = minutes_to_steps(w_min)
            diff = cdf.diff(steps)
            is_nan = diff.isna()
            onset = (diff == 1).astype(float).mask(is_nan)
            recovery = (diff == -1).astype(float).mask(is_nan)
            onset.columns = [f'trans_on_{w_min}min_{sid_name}']
            recovery.columns = [f'trans_off_{w_min}min_{sid_name}']
            frames.append(onset)
            frames.append(recovery)

    extra = build_extra_sensor_features(speed_series, congestion_series, horizon_min, speed_series.name)
    if len(extra.columns) > 0:
        frames.append(extra)

    return add_time_features(pd.concat(frames, axis=1))


def prepare_train_eval_test_split(features, target_series, horizon_min):
    """70/10/20 chronological split. Returns X_train, X_eval, X_test, y_train, y_eval, y_test."""
    horizon_steps = minutes_to_steps(horizon_min)
    future_targets = pd.concat(
        [target_series.shift(-step) for step in range(1, horizon_steps + 1)],
        axis=1,
    )
    target = future_targets.max(axis=1).fillna(0).astype(int).rename('target')
    dataset = features.join(target)

    if FIX_FLAGS['tail_drop'] and len(dataset) > horizon_steps:
        # Rows without a complete future window cannot be assigned a target.
        dataset = dataset.iloc[:len(dataset) - horizon_steps]

    X = dataset.drop(columns=['target'])
    y = dataset['target']

    n = len(X)
    n_train = int(n * 0.7)
    n_eval = int(n * 0.1)

    X_train = X.iloc[:n_train]
    X_eval = X.iloc[n_train:n_train + n_eval]
    X_test = X.iloc[n_train + n_eval:]
    y_train = y.iloc[:n_train]
    y_eval = y.iloc[n_train:n_train + n_eval]
    y_test = y.iloc[n_train + n_eval:]

    return X_train, X_eval, X_test, y_train, y_eval, y_test


def load_speed_data():
    data_file = _DATASET_CONFIGS[DATASET]['data_file']
    data_path = DATASET_DIR / data_file
    print(f'Loading {data_file} (dataset={DATASET})...')
    print(f'Data safeguards: {describe_fix_flags()}')
    df = pd.read_csv(data_path)
    df.rename(columns={'Unnamed: 0': 'timestamp'}, inplace=True)
    smooth = df.set_index('timestamp')

    if FIX_FLAGS['zero_missing']:
        # Exact-zero speeds are missing-data sentinels in METR-LA. XGBoost handles the
        # resulting NaNs natively.
        values = smooth.to_numpy()
        n_cells = values.size
        n_zero = int(np.count_nonzero(values == 0))
        frac = (100.0 * n_zero / n_cells) if n_cells else 0.0
        print(f'Zero-as-missing: {n_zero}/{n_cells} speed cells are exactly 0 ({frac:.2f}%); '
              f'converting to NaN.')
        smooth = smooth.replace(0.0, np.nan).dropna(how='all')
    else:
        smooth = smooth.dropna()

    print(f'Data loaded: {smooth.shape[0]} rows, {smooth.shape[1]} columns. Computing congestion labels...')

    if FIX_FLAGS['leak_threshold']:
        # Compute congestion thresholds on the training split only.
        def _congested(col):
            n_train = _train_row_count(len(col))
            threshold = col.iloc[:n_train].quantile(CONGESTION_QUANTILE) * CONGESTION_THRESHOLD_TAU
            return (col < threshold).astype(int)
        smooth_congested = smooth.apply(_congested)
    else:
        smooth_congested = smooth.apply(
            lambda col: (col < col.quantile(CONGESTION_QUANTILE) * CONGESTION_THRESHOLD_TAU).astype(int)
        )
    return smooth, smooth_congested


def evaluate_on_test(
    model,
    X_eval,
    y_eval,
    X_test,
    y_test,
    current_test_labels=None,
    current_eval_labels=None,
    selection_metric='f1',
):
    deval = xgb.DMatrix(data=X_eval, label=y_eval)
    y_eval_pred_prob = predict_with_best_iteration(model, deval)
    best_threshold, _, _, _, _, _ = find_best_threshold(
        y_eval.values,
        y_eval_pred_prob,
        current_labels=current_eval_labels,
        metric=selection_metric,
    )

    dtest = xgb.DMatrix(data=X_test, label=y_test)
    start_inference_time = time.time()
    y_pred_prob = predict_with_best_iteration(model, dtest)
    inference_time = time.time() - start_inference_time
    inference_time_per_sample_us = (inference_time / len(X_test)) * 1_000_000

    y_pred = (y_pred_prob >= best_threshold).astype(int)
    f1 = f1_score(y_test.values, y_pred, zero_division=0)
    transition_f1 = compute_transition_f1(y_test.values, y_pred, current_test_labels)
    trsp = compute_trsp(y_test.values, y_pred)
    precision = precision_score(y_test.values, y_pred, zero_division=0)
    recall = recall_score(y_test.values, y_pred, zero_division=0)
    return f1, precision, recall, transition_f1, trsp, inference_time_per_sample_us


def local_feature_suffix(feature_name, sid):
    sid = str(sid)
    if feature_name == sid:
        return '__current_speed__'
    if feature_name.endswith(f'_{sid}'):
        return feature_name[:-(len(sid) + 1)]
    if sid in feature_name:
        return feature_name.replace(sid, '{sid}')
    return None


def most_common_params(params_list):
    if not params_list:
        return {}
    counter = Counter(tuple(sorted(params.items())) for params in params_list)
    return dict(counter.most_common(1)[0][0])


def build_horizon_meta(importance_by_horizon, best_params_by_horizon, horizons_minutes):
    horizon_meta = {}
    for hor in horizons_minutes:
        suffix_scores = {}
        for sid, importance_map in importance_by_horizon.get(hor, []):
            for feature_name, score in importance_map.items():
                suffix = local_feature_suffix(feature_name, sid)
                if suffix is None:
                    continue
                suffix_scores[suffix] = suffix_scores.get(suffix, 0) + score

        ranked_local_feature_name_suffixes = [
            suffix for suffix, _ in sorted(suffix_scores.items(), key=lambda item: item[1], reverse=True)
        ]

        horizon_meta[hor] = {
            'top_local_feature_suffixes': ranked_local_feature_name_suffixes,
            'best_params_mode': most_common_params(best_params_by_horizon.get(hor, [])),
        }
    return horizon_meta


def get_own_sensor_columns(feature_columns, sid):
    """Return columns belonging to the target sensor or shared time features."""
    sid_str = str(sid)
    suffix = f'_{sid_str}'
    return [
        col for col in feature_columns
        if col == sid_str or col.endswith(suffix) or col in time_feature_names
    ]


def select_features(feature_columns, sid, importance_map, nfeat):
    """Always keep own-sensor/time features, then fill remaining slots by gain."""
    own_cols = get_own_sensor_columns(feature_columns, sid)
    own_set = set(own_cols)

    extra_budget = max(0, nfeat - len(own_cols))
    if extra_budget == 0:
        return own_cols

    cross_cols = [col for col in feature_columns if col not in own_set]
    cross_ranked = sorted(cross_cols, key=lambda col: importance_map.get(col, 0.0), reverse=True)
    return own_cols + cross_ranked[:extra_budget]


def train_feature_sweep_model(
    X_train,
    y_train,
    X_eval,
    y_eval,
    feature_sweep_params,
    feature_sweep_max_boost_rounds,
    early_stopping_rounds,
    current_eval_labels=None,
    selection_metric='f1',
):
    """Train one fixed-parameter model for cheap feature-count selection."""
    X_inner_train, X_val, y_inner_train, y_val = time_series_validation_split(
        X_train,
        y_train,
        validation_fraction=0.1,
    )

    dtrain = xgb.DMatrix(data=X_inner_train, label=y_inner_train)
    dval = xgb.DMatrix(data=X_val, label=y_val)
    scale_pos_weight = compute_scale_pos_weight(y_inner_train)
    params = build_xgb_params(feature_sweep_params, scale_pos_weight=scale_pos_weight)

    model = train_booster(
        params=params,
        dtrain=dtrain,
        num_boost_round=feature_sweep_max_boost_rounds,
        evals=[(dtrain, 'train'), (dval, 'validation')],
        early_stopping_rounds=early_stopping_rounds,
    )
    return model, evaluate_model(
        model,
        X_eval,
        y_eval,
        current_eval_labels=current_eval_labels,
        selection_metric=selection_metric,
    )


def train_base_ranking_model(
    X_train,
    y_train,
    X_eval,
    y_eval,
    base_ranking_params,
    base_ranking_max_boost_rounds,
    early_stopping_rounds,
):
    """Train one fixed-parameter full-feature model for gain-based ranking."""
    X_inner_train, X_val, y_inner_train, y_val = time_series_validation_split(
        X_train,
        y_train,
        validation_fraction=0.1,
    )

    dtrain = xgb.DMatrix(data=X_inner_train, label=y_inner_train)
    dval = xgb.DMatrix(data=X_val, label=y_val)
    scale_pos_weight = compute_scale_pos_weight(y_inner_train)
    params = build_xgb_params(base_ranking_params, scale_pos_weight=scale_pos_weight)

    model = train_booster(
        params=params,
        dtrain=dtrain,
        num_boost_round=base_ranking_max_boost_rounds,
        evals=[(dtrain, 'train'), (dval, 'validation')],
        early_stopping_rounds=early_stopping_rounds,
    )
    best_iteration = getattr(model, 'best_iteration', None)
    num_boost_round = base_ranking_max_boost_rounds if best_iteration is None else best_iteration + 1
    threshold, f1, precision, recall, transition_f1, score = evaluate_model(model, X_eval, y_eval)

    best_params = base_ranking_params.copy()
    best_params['num_boost_round'] = num_boost_round
    return model, best_params, (threshold, f1, precision, recall, transition_f1, score)


def to_builtin(value):
    if isinstance(value, dict):
        return {key: to_builtin(item) for key, item in value.items()}
    if isinstance(value, list):
        return [to_builtin(item) for item in value]
    if isinstance(value, tuple):
        return [to_builtin(item) for item in value]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    return value


def active_search_stage_groups(param_grid, search_stage_groups):
    return [
        stage_keys for stage_keys in search_stage_groups
        if np.prod([len(param_grid[key]) for key in stage_keys]) > 1
    ]


def congestion_state_columns(congestion_features):
    """Current binary congestion state per sensor, renamed cong_<sensor>."""
    state = congestion_features.copy()
    state.columns = [f'cong_{col}' for col in congestion_features.columns]
    return state


def congestion_rolling_mean_columns(congestion_features, steps, prefix):
    """Rolling fraction-of-time-congested (congestion persistence/intensity)."""
    rolled = congestion_features.rolling(window=steps, min_periods=steps).mean()
    rolled.columns = [f'{prefix}_cong_{col}' for col in congestion_features.columns]
    return rolled


def congestion_transition_columns(congestion_features, steps, window_min):
    """Onset (0->1) and recovery (1->0) congestion flips over the trailing window."""
    diff = congestion_features.diff(steps)
    is_nan = diff.isna()
    onset = (diff == 1).astype(float).mask(is_nan)
    recovery = (diff == -1).astype(float).mask(is_nan)
    onset.columns = [f'trans_on_{window_min}min_{col}' for col in congestion_features.columns]
    recovery.columns = [f'trans_off_{window_min}min_{col}' for col in congestion_features.columns]
    return [onset, recovery]
