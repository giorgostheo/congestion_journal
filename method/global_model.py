import argparse
import json
from shared import *


HORIZONS_MINUTES = [10, 20, 30, 40, 50, 60]
ROLLING_WINDOWS_MINUTES = [15, 30, 60]
ROLLING_MIN_WINDOWS_MINUTES = [30]
EARLY_STOPPING_ROUNDS = 50
MAX_BOOST_ROUNDS = 2000

SEARCH_STAGE_GROUPS = [
    ('max_depth', 'eta'),
    ('subsample', 'colsample_bytree'),
    ('min_child_weight', 'gamma'),
    ('reg_lambda', 'reg_alpha'),
]

# Cross-sensor budgets are added to the always-retained own-sensor columns.
FEATURE_SWEEP_CROSS_BUDGETS = [15, 30]
FEATURE_SWEEP_MAX_BOOST_ROUNDS = 600
FEATURE_SWEEP_PARAMS = {
    'max_depth': 3,
    'eta': 0.05,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'min_child_weight': 1,
    'gamma': 0,
    'reg_lambda': 1,
    'reg_alpha': 0,
}

BASE_RANKING_MAX_BOOST_ROUNDS = 600
BASE_RANKING_PARAMS = {
    'max_depth': 3,
    'eta': 0.05,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'min_child_weight': 1,
    'gamma': 0,
    'reg_lambda': 1,
    'reg_alpha': 0,
}

TRANSITION_WEIGHTS = XGB_TRANSITION_WEIGHTS

SPECIALIZED_RESULT_COLUMNS = [
    'transition_weight',
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

TRANSITION_WINDOWS_MINUTES = [10, 20, 30]

# Trimmed speed-lag set: drop the long-range (12h/24h/48h/7d) lags that add little at a
# 10-60min horizon but cost ~4 columns per sensor.
SPEED_LAG_MINUTES = [5, 10, 15, 30, 60, 180]

# Univariate pre-filter: only the top-K columns by |corr with target| (plus own-sensor and
# time columns, always kept) reach the gain-based base ranker, which can only split on a few
# thousand features anyway (max_depth=3 x 600 rounds ~ 4.2k split slots).
PREFILTER_TOP_K = 1000


def build_spatial_transition_features(onset_df, recovery_df, hop_sets):
    """Hop-aggregated upstream congestion transitions (leading indicators).

    For each sensor, computes the inverse-distance-weighted fraction of its upstream
    neighbors (at hop 1, 2, 3) that had an onset (0->1) or a recovery (1->0) over the
    trailing window encoded in onset_df/recovery_df. Bounded by the number of hops, not
    by neighbor count, so it stays small regardless of graph connectivity. Columns are
    named trans_on_h{hop}_{sensor_idx} / trans_off_h{hop}_{sensor_idx}, matching the
    cong_h{hop}_{idx} convention.
    """
    if hop_sets is None:
        return pd.DataFrame(index=onset_df.index)

    all_sensor_ids = list(onset_df.columns)
    n = len(all_sensor_ids)
    spatial_frames = []

    for idx in range(n):
        if idx not in hop_sets:
            continue
        target_hops = hop_sets[idx]

        sensor_features = {}
        for hop_dist in [1, 2, 3]:
            neighbors_at_hop = [(nb_i, d) for nb_i, d in target_hops.items()
                                if d == hop_dist and d > 0 and nb_i < n]
            if not neighbors_at_hop:
                continue

            nb_ids = [all_sensor_ids[nb_i] for nb_i, _ in neighbors_at_hop]
            weights = np.array([1.0 / d for _, d in neighbors_at_hop], dtype=float)
            weight_row = weights.reshape(1, -1)
            weight_sum = weights.sum()

            onset_block = onset_df[nb_ids].values
            recovery_block = recovery_df[nb_ids].values
            sensor_features[f'trans_on_h{hop_dist}'] = pd.Series(
                (onset_block * weight_row).sum(axis=1) / weight_sum, index=onset_df.index)
            sensor_features[f'trans_off_h{hop_dist}'] = pd.Series(
                (recovery_block * weight_row).sum(axis=1) / weight_sum, index=onset_df.index)

        if sensor_features:
            frame = pd.DataFrame(sensor_features, index=onset_df.index)
            frame.columns = [f'{col}_{idx}' for col in frame.columns]
            spatial_frames.append(frame)

    if not spatial_frames:
        return pd.DataFrame(index=onset_df.index)

    return pd.concat(spatial_frames, axis=1)


def build_wave_aligned_spatial_features(speed_df, congestion_df, hop_sets):
    """Upstream hop aggregates lagged by the expected shockwave travel time.

    Congestion shockwaves propagate upstream at roughly one hop per
    WAVE_STEPS_PER_HOP sampling steps, so each hop-d neighbor block is shifted
    back d*WAVE_STEPS_PER_HOP steps before inverse-distance aggregation. This
    turns the spatial aggregates into leading indicators aligned with the wave
    instead of a snapshot of the current network state. Columns are named
    wspd_h{hop}_{idx} / wcong_h{hop}_{idx}, mirroring spd_h/cong_h.
    """
    if hop_sets is None:
        return pd.DataFrame(index=speed_df.index)

    all_sensor_ids = list(speed_df.columns)
    n = len(all_sensor_ids)
    spatial_frames = []

    for idx in range(n):
        if idx not in hop_sets:
            continue
        target_hops = hop_sets[idx]

        sensor_features = {}
        for hop_dist in [1, 2, 3]:
            neighbors_at_hop = [(nb_i, d) for nb_i, d in target_hops.items()
                                if d == hop_dist and d > 0 and nb_i < n]
            if not neighbors_at_hop:
                continue

            nb_ids = [all_sensor_ids[nb_i] for nb_i, _ in neighbors_at_hop]
            weights = np.array([1.0 / d for _, d in neighbors_at_hop], dtype=float)
            weight_row = weights.reshape(1, -1)
            weight_sum = weights.sum()
            wave_lag_steps = hop_dist * WAVE_STEPS_PER_HOP

            speed_block = speed_df[nb_ids].shift(wave_lag_steps).values
            sensor_features[f'wspd_h{hop_dist}'] = pd.Series(
                (speed_block * weight_row).sum(axis=1) / weight_sum, index=speed_df.index)

            if all(c in congestion_df.columns for c in nb_ids):
                cong_block = congestion_df[nb_ids].shift(wave_lag_steps).values
                sensor_features[f'wcong_h{hop_dist}'] = pd.Series(
                    (cong_block * weight_row).sum(axis=1) / weight_sum, index=speed_df.index)

        if sensor_features:
            frame = pd.DataFrame(sensor_features, index=speed_df.index)
            frame.columns = [f'{col}_{idx}' for col in frame.columns]
            spatial_frames.append(frame)

    if not spatial_frames:
        return pd.DataFrame(index=speed_df.index)

    return pd.concat(spatial_frames, axis=1)


def build_engineered_feature_frame(
    speed_features, congestion_features, horizon_min, sensor_id_to_ind, upstream_hop_sets,
    target_sensor_ids,
):
    """Build the global model's speed, congestion, and graph feature frame."""
    feature_frames = [speed_features.copy()]

    for lag_min in sorted({horizon_min, *SPEED_LAG_MINUTES}):
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

    # Current upstream spatial aggregates.
    if sensor_id_to_ind is not None and upstream_hop_sets is not None:
        validate_spatial_sensor_alignment(speed_features, sensor_id_to_ind)
        spatial = build_spatial_aggregation_features(speed_features, congestion_features, upstream_hop_sets)
        if len(spatial.columns) > 0:
            feature_frames.append(spatial)

        # Wave-aligned variant: hop-d aggregates lagged by the shockwave travel time.
        if FEATURE_FLAGS['wave']:
            wave_spatial = build_wave_aligned_spatial_features(
                speed_features, congestion_features, upstream_hop_sets,
            )
            if len(wave_spatial.columns) > 0:
                feature_frames.append(wave_spatial)

    # Own-sensor congestion features for target sensors only.
    target_cols = [col for col in target_sensor_ids if col in congestion_features.columns]
    target_cong = congestion_features[target_cols]
    feature_frames.append(congestion_state_columns(target_cong))
    for window_min in ROLLING_WINDOWS_MINUTES:
        window_steps = minutes_to_steps(window_min)
        feature_frames.append(
            congestion_rolling_mean_columns(target_cong, window_steps, f'roll_mean_{window_min}min')
        )
    for window_min in sorted({horizon_min, *TRANSITION_WINDOWS_MINUTES}):
        window_steps = minutes_to_steps(window_min)
        feature_frames.extend(
            congestion_transition_columns(target_cong, window_steps, window_min)
        )

    # Configurable own-sensor profile, state, margin, persistence, and EMA features.
    for sid in target_cols:
        extra = build_extra_sensor_features(
            speed_features[sid], congestion_features[sid], horizon_min, sid,
        )
        if len(extra.columns) > 0:
            feature_frames.append(extra)

    # Hop-aggregated upstream transitions over the trailing horizon window.
    if sensor_id_to_ind is not None and upstream_hop_sets is not None:
        horizon_steps = minutes_to_steps(horizon_min)
        diff = congestion_features.diff(horizon_steps)
        is_nan = diff.isna()
        onset_all = (diff == 1).astype(float).mask(is_nan)
        recovery_all = (diff == -1).astype(float).mask(is_nan)
        spatial_trans = build_spatial_transition_features(onset_all, recovery_all, upstream_hop_sets)
        if len(spatial_trans.columns) > 0:
            feature_frames.append(spatial_trans)

    feature_frame = pd.concat(feature_frames, axis=1)
    feature_frame = add_time_features(feature_frame)
    return feature_frame


def build_prefilter_context(features):
    """Precompute the train-split centered feature matrix and per-column norms ONCE.

    The pre-filter's correlation only depends on the target via y; the feature matrix and
    the 70% train-row split are identical across all sensors of a horizon. Doing the heavy
    DataFrame->numpy conversion + centering here (once per horizon) instead of per-sensor
    turns each sensor's pre-filter into a single matrix-vector product. Returns None when
    there are fewer columns than any sensible top-K (caller falls back to keeping all).
    """
    columns = list(features.columns)
    n_train = int(len(features) * 0.7)

    train_block = features.iloc[:n_train]
    col_means = train_block.mean().to_numpy(dtype=np.float32)
    centered = train_block.to_numpy(dtype=np.float32, copy=True)
    nan_rows, nan_cols = np.where(np.isnan(centered))
    centered[nan_rows, nan_cols] = np.take(col_means, nan_cols)
    centered -= centered.mean(axis=0, keepdims=True)
    col_sumsq = np.einsum('ij,ij->j', centered, centered).astype(np.float64)

    return {
        'columns': columns,
        'n_train': n_train,
        'centered': centered,        # float32, (n_train, n_cols), mean-removed
        'col_sumsq': col_sumsq,      # float64, (n_cols,)
    }


def univariate_prefilter_columns(features, target_series, horizon_min, sid, top_k, context=None):
    """Return own-sensor/time columns (always kept) + the top-K columns by |point-biserial
    correlation| with the target, computed on the TRAIN split only (no leakage).

    This caps what the gain-based base ranker has to chew through. With max_depth=3 and
    ~600 rounds the ranker can only assign splits to a few thousand features at most, so
    feeding it ~15k columns is wasteful and lets noise features randomly score high gain.

    Pass a `context` from build_prefilter_context to reuse the (sensor-independent) centered
    feature matrix across all sensors of a horizon; otherwise it is built on the fly.
    """
    columns = list(features.columns)
    if len(columns) <= top_k:
        return columns

    if context is None:
        context = build_prefilter_context(features)
    n_train = context['n_train']
    centered = context['centered']
    col_sumsq = context['col_sumsq']

    always_keep = set(get_own_sensor_columns(columns, sid))

    # Build the train-split target directly (max congestion over the next horizon steps),
    # avoiding a join against the full feature frame.
    horizon_steps = minutes_to_steps(horizon_min)
    future_targets = pd.concat(
        [target_series.shift(-step) for step in range(1, horizon_steps + 1)],
        axis=1,
    )
    target = future_targets.max(axis=1).fillna(0).astype(int)
    y = target.iloc[:n_train].to_numpy(dtype=np.float64)
    y = y - y.mean()
    y_ss = float(np.dot(y, y))

    scores = np.zeros(len(columns), dtype=np.float64)
    if y_ss > 0:
        numerator = centered.T @ y.astype(np.float32)
        denominator = np.sqrt(col_sumsq * y_ss)
        with np.errstate(divide='ignore', invalid='ignore'):
            corr = np.where(denominator > 0, numerator / denominator, 0.0)
        scores = np.abs(corr)

    keep = set(always_keep)
    for j in np.argsort(scores)[::-1]:
        if len(keep) >= top_k:
            break
        keep.add(columns[j])

    return [col for col in columns if col in keep]


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            'Run the graph-aware global XGBoost model with feature selection.'
        )
    )
    parser.add_argument(
        '--save-transfer-artifacts',
        action='store_true',
        help='Save global_artifacts.pkl and global_horizon_meta.pkl for use by transfer.py.',
    )
    return parser.parse_args()


def main():
    args = parse_args()
    pipeline_start_time = time.time()
    print('=== Global model: graph-aware XGBoost with feature selection ===')
    print(f'Transition sample weights: {TRANSITION_WEIGHTS}')
    print(f'Selection metric: {XGB_SELECTION_METRIC}')
    print(f'Feature flags: {describe_feature_flags()}')
    if FEATURE_FLAGS['wave']:
        print(f'Wave-aligned spatial lag: {WAVE_STEPS_PER_HOP} step(s)/hop '
              f'({WAVE_STEPS_PER_HOP * SAMPLING_RATE_MINUTES} min/hop)')
    print(f'Speed lag minutes: {SPEED_LAG_MINUTES} (+ prediction horizon)')
    print(f'Congestion persistence windows (min): {ROLLING_WINDOWS_MINUTES}')
    print(f'Own-sensor transition windows (min): {TRANSITION_WINDOWS_MINUTES} (+ horizon)')
    print(f'Univariate pre-filter top-K into base ranker: {PREFILTER_TOP_K}')
    print(f'Base ranking params: {BASE_RANKING_PARAMS}')
    print(f'Base ranking max rounds: {BASE_RANKING_MAX_BOOST_ROUNDS}')
    print(f'Feature-count sweep cross-sensor budgets (on top of own-sensor cols): {FEATURE_SWEEP_CROSS_BUDGETS}')
    print(f'Feature-count sweep params: {FEATURE_SWEEP_PARAMS}')
    print(f'Feature-count sweep max rounds: {FEATURE_SWEEP_MAX_BOOST_ROUNDS}')
    print(f'Model search grid: {MODEL_SEARCH_GRID}')
    print(f'Active search stages: {active_search_stage_groups(MODEL_SEARCH_GRID, SEARCH_STAGE_GROUPS)}')
    if args.save_transfer_artifacts:
        print('Weighted transfer artifact saving enabled.')
    else:
        print('Weighted transfer artifact saving disabled. Use --save-transfer-artifacts to enable.')

    sensor_ids, sensor_id_to_ind, adj_arr = load_adjacency()
    upstream_hop_sets = None
    if sensor_ids is not None and sensor_id_to_ind is not None:
        symmetric = np.allclose(adj_arr, adj_arr.T)
        print(f'Loaded adjacency: {adj_arr.shape[0]} sensors  symmetric={symmetric}')
        print('Building upstream k-hop sets (traversing columns of adj matrix)...')
        upstream_hop_sets = compute_upstream_hop_sets(sensor_ids, sensor_id_to_ind, adj_arr)
        no_upstream = sum(1 for v in upstream_hop_sets.values() if len(v) == 1)
        print(f'Built upstream k-hop sets (k_max=3) for all {len(upstream_hop_sets)} sensors '
              f'({no_upstream} sensors have no upstream neighbors)')
    else:
        print('Running without spatial features (no adjacency available)')

    smooth, smooth_congested = load_speed_data()
    smooth_sub = smooth.iloc[:int(len(smooth) * 0.9999)].copy()
    smooth_congested_sub = smooth_congested.iloc[:int(len(smooth_congested) * 0.9999)].copy()

    result_path = timestamped_output_path(RESULTS_DIR, 'results_global.csv')
    choices_path = timestamped_output_path(RESULTS_DIR, 'choices_global.jsonl')
    artifacts = {}
    importance_by_weight_horizon = {}
    best_params_by_weight_horizon = {}
    importance_rows = []
    selection_rows = []
    results_by_weight_horizon = {}

    all_sids = [item for sublist in LOCATIONS for item in sublist]
    total_runs = len(TRANSITION_WEIGHTS) * len(HORIZONS_MINUTES) * len(all_sids)

    progress = tqdm(total=total_runs, desc='Global model', unit='run')
    for hor in HORIZONS_MINUTES:
        progress.set_description(f'Global hor={hor}min (building frame)')
        engineered_features = build_engineered_feature_frame(
            smooth_sub,
            smooth_congested_sub,
            hor,
            sensor_id_to_ind,
            upstream_hop_sets,
            all_sids,
        )
        n_full = engineered_features.shape[1]

        # Sensor-independent pre-filter context (centered train matrix + column norms),
        # built once per horizon and reused across all sensors.
        prefilter_context = (
            build_prefilter_context(engineered_features) if n_full > PREFILTER_TOP_K else None
        )
        progress.set_description(f'Global hor={hor}min')

        for sid in all_sids:
            # Base ranking and feature selection are independent of transition_weight — compute once per (hor, sid).
            # Pre-filter the full frame down to PREFILTER_TOP_K candidates before the gain ranker.
            feature_columns = univariate_prefilter_columns(
                engineered_features, smooth_congested_sub[sid], hor, sid, PREFILTER_TOP_K,
                context=prefilter_context,
            )
            n_before_select = len(feature_columns)
            engineered_features_pf = engineered_features[feature_columns]

            X_train_full, X_eval_full, _, y_train_full, y_eval_full, _ = prepare_train_eval_test_split(
                engineered_features_pf,
                smooth_congested_sub[sid],
                hor,
            )

            base_model, base_params, base_eval = train_base_ranking_model(
                X_train_full,
                y_train_full,
                X_eval_full,
                y_eval_full,
                BASE_RANKING_PARAMS,
                BASE_RANKING_MAX_BOOST_ROUNDS,
                EARLY_STOPPING_ROUNDS,
            )
            importance_map = base_model.get_score(importance_type='gain')

            own_cols = get_own_sensor_columns(feature_columns, sid)
            nfeat_best_n = None
            nfeat_best_score = -1
            feature_sweep_selection_metric = XGB_SELECTION_METRIC
            feature_sweep_scores = []
            for cross_budget in FEATURE_SWEEP_CROSS_BUDGETS:
                nfeat = len(own_cols) + cross_budget
                selected_columns_s = select_features(feature_columns, sid, importance_map, nfeat)
                feature_subset = engineered_features[selected_columns_s]
                X_train_s, X_eval_s, _, y_train_s, y_eval_s, _ = prepare_train_eval_test_split(
                    feature_subset,
                    smooth_congested_sub[sid],
                    hor,
                )
                current_eval_labels_s = labels_for_index(smooth_congested_sub[sid], X_eval_s.index)

                _, (threshold_s, best_f1_s, precision_s, recall_s, transition_f1_s, score_s) = train_feature_sweep_model(
                    X_train_s,
                    y_train_s,
                    X_eval_s,
                    y_eval_s,
                    FEATURE_SWEEP_PARAMS,
                    FEATURE_SWEEP_MAX_BOOST_ROUNDS,
                    EARLY_STOPPING_ROUNDS,
                    current_eval_labels=current_eval_labels_s,
                    selection_metric=feature_sweep_selection_metric,
                )
                feature_sweep_scores.append({
                    'requested_cross_budget': cross_budget,
                    'requested_nfeat': nfeat,
                    'actual_n_features': len(selected_columns_s),
                    'threshold': threshold_s,
                    'f1': best_f1_s,
                    'precision': precision_s,
                    'recall': recall_s,
                    'transition_f1': transition_f1_s,
                    'selection_metric': feature_sweep_selection_metric,
                    'selection_score': score_s,
                })
                if score_s > nfeat_best_score:
                    nfeat_best_score = score_s
                    nfeat_best_n = nfeat

            if nfeat_best_n is None:
                selected_columns = own_cols
            else:
                selected_columns = select_features(feature_columns, sid, importance_map, nfeat_best_n)

            n_own = len(own_cols)
            n_total = len(selected_columns)

            engineered_features_best = engineered_features[selected_columns]
            X_train, X_eval, X_test, y_train, y_eval, y_test = prepare_train_eval_test_split(
                engineered_features_best,
                smooth_congested_sub[sid],
                hor,
            )

            for transition_weight in TRANSITION_WEIGHTS:
                current_train_labels = labels_for_index(smooth_congested_sub[sid], X_train.index)
                current_eval_labels = labels_for_index(smooth_congested_sub[sid], X_eval.index)
                sample_weight = transition_sample_weights(current_train_labels, y_train, transition_weight)
                transition_train_count = int(np.sum(sample_weight.values > 1.0))
                selection_metric = selection_metric_for_weight(transition_weight)

                final_model, best_params, final_search_eval = find_best_xgboost_model_weighted(
                    MODEL_SEARCH_GRID,
                    X_train,
                    y_train,
                    sample_weight,
                    X_eval,
                    y_eval,
                    SEARCH_STAGE_GROUPS,
                    MAX_BOOST_ROUNDS,
                    EARLY_STOPPING_ROUNDS,
                    current_eval_labels=current_eval_labels,
                    selection_metric=selection_metric,
                )

                # Per-run selection/param diagnostics go to the choices JSONL sidecar
                # (one JSON object per line), not stdout, to keep the console clean.
                choices_record = to_builtin({
                    'transition_weight': transition_weight,
                    'sid': sid,
                    'horizon_min': hor,
                    'transition_train_count': transition_train_count,
                    'feature_frame_full_columns': n_full,
                    'feature_prefilter_top_k': PREFILTER_TOP_K,
                    'feature_columns_after_prefilter': n_before_select,
                    'base_ranking_fixed_params': BASE_RANKING_PARAMS,
                    'base_ranking_max_boost_rounds': BASE_RANKING_MAX_BOOST_ROUNDS,
                    'base_ranking_best_params': base_params,
                    'base_ranking_eval_threshold': base_eval[0],
                    'base_ranking_eval_f1': base_eval[1],
                    'base_ranking_eval_precision': base_eval[2],
                    'base_ranking_eval_recall': base_eval[3],
                    'feature_sweep_fixed_params': FEATURE_SWEEP_PARAMS,
                    'feature_sweep_max_boost_rounds': FEATURE_SWEEP_MAX_BOOST_ROUNDS,
                    'feature_sweep_scores': feature_sweep_scores,
                    'selected_requested_nfeat': nfeat_best_n,
                    'selected_actual_n_features': n_total,
                    'selected_own_sensor_features': n_own,
                    'selected_cross_sensor_features': n_total - n_own,
                    'final_search_grid': MODEL_SEARCH_GRID,
                    'final_search_active_stages': active_search_stage_groups(
                        MODEL_SEARCH_GRID,
                        SEARCH_STAGE_GROUPS,
                    ),
                    'final_search_best_params': best_params,
                    'final_search_eval_threshold': final_search_eval[0],
                    'final_search_eval_f1': final_search_eval[1],
                    'final_search_eval_precision': final_search_eval[2],
                    'final_search_eval_recall': final_search_eval[3],
                    'final_search_eval_transition_f1': final_search_eval[4],
                    'final_search_selection_metric': selection_metric,
                    'final_search_selection_score': final_search_eval[5],
                })
                with open(choices_path, 'a') as choices_file:
                    choices_file.write(json.dumps(choices_record, sort_keys=True) + '\n')

                current_test_labels = labels_for_index(smooth_congested_sub[sid], X_test.index)
                f1, precision, recall, transition_f1, trsp, inference_us = evaluate_on_test(
                    final_model,
                    X_eval,
                    y_eval,
                    X_test,
                    y_test,
                    current_test_labels,
                    current_eval_labels=current_eval_labels,
                    selection_metric=selection_metric,
                )
                variant_name = f'global_W{transition_weight:g}'
                collect_grouped_feature_importance(
                    importance_rows,
                    variant_name,
                    hor,
                    final_model,
                    selected_columns,
                    sensor_ids=smooth_sub.columns,
                )
                collect_grouped_selection_counts(
                    selection_rows,
                    variant_name,
                    hor,
                    selected_columns,
                    sensor_ids=smooth_sub.columns,
                )

                if args.save_transfer_artifacts:
                    artifacts[(transition_weight, sid, hor)] = {
                        'importance_map': importance_map,
                        'best_params': best_params,
                        'train_probs': predict_with_best_iteration(final_model, xgb.DMatrix(X_train)),
                        'train_index': X_train.index,
                    }
                    importance_by_weight_horizon.setdefault(transition_weight, {}).setdefault(hor, []).append((sid, importance_map))
                    best_params_by_weight_horizon.setdefault(transition_weight, {}).setdefault(hor, []).append(best_params)

                append_specialized_result_row(result_path, {
                    'transition_weight': transition_weight,
                    'sid': sid,
                    'horizon_min': hor,
                    'f1': f1,
                    'transition_f1': transition_f1,
                    'trsp': trsp,
                    'precision': precision,
                    'recall': recall,
                    'inference_us': inference_us,
                    'n_features': len(selected_columns),
                    'n_boost_rounds': best_params['num_boost_round'],
                }, SPECIALIZED_RESULT_COLUMNS)

                key = (transition_weight, hor)
                if key not in results_by_weight_horizon:
                    results_by_weight_horizon[key] = []
                results_by_weight_horizon[key].append({
                    'f1': f1,
                    'transition_f1': transition_f1,
                    'trsp': trsp,
                    'precision': precision,
                    'recall': recall,
                    'inference_us': inference_us,
                })

                progress.update(1)

    progress.close()

    if args.save_transfer_artifacts:
        horizon_meta = {
            transition_weight: build_horizon_meta(
                importance_by_weight_horizon.get(transition_weight, {}),
                best_params_by_weight_horizon.get(transition_weight, {}),
                HORIZONS_MINUTES,
            )
            for transition_weight in TRANSITION_WEIGHTS
        }
        artifacts_path = RESULTS_DIR / 'global_artifacts.pkl'
        horizon_meta_path = RESULTS_DIR / 'global_horizon_meta.pkl'
        save_pickle(artifacts, artifacts_path)
        save_pickle(horizon_meta, horizon_meta_path)
        print(f'Saved global artifacts to {artifacts_path}')
        print(f'Saved global horizon metadata to {horizon_meta_path}')

    importance_path = timestamped_output_path(FEATURE_IMPORTANCE_DIR, 'feature_importance_global_aggregated.csv')
    write_aggregated_feature_importance(importance_path, importance_rows, selection_rows)
    print(f'Saved aggregated feature importance to {importance_path}')
    print(f'Saved per-run selection/param diagnostics to {choices_path}')
    print(f'Saved global results to {result_path}')

    print('dataset  method  horizon_min  f1_mean  transition_f1_mean  trsp  prec_mean  rec_mean  inference_us_all_nodes  inference_us_per_node')
    for transition_weight, hor in sorted(results_by_weight_horizon.keys()):
        entries = results_by_weight_horizon[(transition_weight, hor)]
        method = f'XGB-Global-w{transition_weight:g}'
        row = append_summary_result_row(method, hor, entries)
        print(
            f'{row["dataset"]}  {method:15s} {hor:10d} '
            f'{row["f1_mean"]:.6f}  {row["transition_f1_mean"]:.6f}  {row["trsp"]:.6f}  '
            f'{row["prec_mean"]:.6f}  {row["rec_mean"]:.6f}  '
            f'{row["inference_us_all_nodes"]:22.6f}  {row["inference_us_per_node"]:21.6f}'
        )

    elapsed = time.time() - pipeline_start_time
    append_training_time_row('XGB-Global', elapsed)
    elapsed_minutes = elapsed / 60
    elapsed_hours = elapsed / 3600
    print(f'Total global training pipeline time: {elapsed_minutes:.2f} minutes ({elapsed_hours:.2f} hours)')


if __name__ == '__main__':
    main()
