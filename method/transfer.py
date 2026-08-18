from shared import *


ALPHA = 0.3
TRANSFER_N_FEATURES = 20
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


def build_warmstart_grid(global_params):
    return {
        'max_depth': [global_params['max_depth'], min(global_params['max_depth'] + 2, 7)],
        'eta': [global_params['eta']],
        'subsample': [global_params['subsample']],
        'colsample_bytree': [global_params['colsample_bytree']],
        'min_child_weight': [global_params['min_child_weight']],
        'gamma': [global_params['gamma']],
        'reg_lambda': [global_params['reg_lambda']],
        'reg_alpha': [global_params['reg_alpha']],
    }


def train_final_model_soft_labels_weighted(best_params, X_train, y_soft, y_hard, sample_weight):
    dtrain = xgb.DMatrix(data=X_train, label=y_soft, weight=sample_weight)
    scale_pos_weight = compute_scale_pos_weight(y_hard)

    final_params = build_xgb_params(
        {key: value for key, value in best_params.items() if key != 'num_boost_round'},
        scale_pos_weight=scale_pos_weight,
    )

    num_boost_round = max(1, int(best_params['num_boost_round']))
    return train_booster(final_params, dtrain, num_boost_round=num_boost_round)


def find_best_xgboost_model_soft_labels_weighted(
    param_grid,
    X_train,
    y_soft,
    y_hard,
    sample_weight,
    X_eval,
    y_eval,
    current_eval_labels=None,
    selection_metric='f1',
):
    X_inner_train, X_val, y_soft_inner, y_soft_val = time_series_validation_split(
        X_train,
        y_soft,
        validation_fraction=0.1,
    )
    y_hard_inner = y_hard.loc[y_soft_inner.index]
    y_hard_val = y_hard.loc[y_soft_val.index]
    weight_inner_train = sample_weight.loc[y_soft_inner.index]

    dtrain = xgb.DMatrix(data=X_inner_train, label=y_soft_inner, weight=weight_inner_train)
    dval = xgb.DMatrix(data=X_val, label=y_hard_val)
    scale_pos_weight = compute_scale_pos_weight(y_hard_inner)

    best_candidate = {key: values[0] for key, values in param_grid.items()}
    best_score = -1
    best_rounds = None
    best_num_boost_round = None

    for stage_keys in SEARCH_STAGE_GROUPS:
        stage_best_candidate = best_candidate.copy()
        stage_best_rounds = best_rounds
        stage_best_num_boost_round = best_num_boost_round
        stage_best_score = best_score

        for candidate_params in iter_stage_param_combinations(best_candidate, param_grid, stage_keys):
            model, num_boost_round, rounds = evaluate_candidate_params(
                candidate_params,
                dtrain,
                dval,
                X_eval,
                y_eval,
                scale_pos_weight,
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
        raise RuntimeError('Weighted soft-label hyperparameter search did not produce a valid XGBoost model.')

    best_params = best_candidate.copy()
    best_params['num_boost_round'] = best_num_boost_round
    final_model = train_final_model_soft_labels_weighted(best_params, X_train, y_soft, y_hard, sample_weight)
    return final_model, best_params, best_rounds


def load_transfer_artifacts():
    artifacts_path = RESULTS_DIR / 'global_artifacts.pkl'
    horizon_meta_path = RESULTS_DIR / 'global_horizon_meta.pkl'
    missing = [str(path) for path in (artifacts_path, horizon_meta_path) if not path.exists()]
    if missing:
        raise FileNotFoundError(
            'Transfer training requires the artifacts produced by global_model.py. '
            f'Missing: {missing}'
        )
    print(f'Loading transfer artifacts from {artifacts_path}')
    return load_pickle(artifacts_path), load_pickle(horizon_meta_path)


def artifact_for_weight(artifacts, horizon_meta, transition_weight, sid, hor):
    return artifacts[(transition_weight, sid, hor)], horizon_meta[transition_weight][hor]


def main():
    pipeline_start_time = time.time()
    print('=== Transfer model: soft-label knowledge distillation ===')
    print(f'Transition sample weights: {TRANSITION_WEIGHTS}')
    print(f'Selection metric: {XGB_SELECTION_METRIC}')
    print(f'Feature flags: {describe_feature_flags()}')
    print(f'ALPHA: {ALPHA}')
    print(f'Transfer n_features: {TRANSFER_N_FEATURES}')

    artifacts, horizon_meta = load_transfer_artifacts()

    smooth, smooth_congested = load_speed_data()
    smooth_sub = smooth.iloc[:int(len(smooth) * 0.9999)].copy()
    smooth_congested_sub = smooth_congested.iloc[:int(len(smooth_congested) * 0.9999)].copy()

    result_path = timestamped_output_path(RESULTS_DIR, 'results_transfer.csv')
    importance_rows = []
    results_by_weight_horizon = {}

    all_sids = [item for sublist in LOCATIONS for item in sublist]
    total_runs = len(TRANSITION_WEIGHTS) * len(all_sids) * len(HORIZONS_MINUTES)

    progress = tqdm(total=total_runs, desc='Transfer model', unit='run')
    for transition_weight in TRANSITION_WEIGHTS:
        for sid in all_sids:
            for hor in HORIZONS_MINUTES:
                art, _ = artifact_for_weight(artifacts, horizon_meta, transition_weight, sid, hor)

                features = build_single_sensor_features(smooth_sub[sid], hor, congestion_series=smooth_congested_sub[sid])
                X_train, X_eval, X_test, y_train, y_eval, y_test = prepare_train_eval_test_split(
                    features,
                    smooth_congested_sub[sid],
                    hor,
                )

                local_importance = {
                    feature_name: score
                    for feature_name, score in art['importance_map'].items()
                    if feature_name in features.columns
                }
                selected_columns = select_top_features(features, local_importance, nfeat=TRANSFER_N_FEATURES)
                X_train = X_train[selected_columns]
                X_eval = X_eval[selected_columns]
                X_test = X_test[selected_columns]

                global_probs = pd.Series(art['train_probs'], index=art['train_index'])
                y_soft = (
                    ALPHA * global_probs.reindex(y_train.index).fillna(0.5)
                    + (1 - ALPHA) * y_train.astype(float)
                )

                current_train_labels = labels_for_index(smooth_congested_sub[sid], X_train.index)
                current_eval_labels = labels_for_index(smooth_congested_sub[sid], X_eval.index)
                sample_weight = transition_sample_weights(current_train_labels, y_train, transition_weight)
                selection_metric = selection_metric_for_weight(transition_weight)

                model, best_params, _ = find_best_xgboost_model_soft_labels_weighted(
                    MODEL_SEARCH_GRID,
                    X_train,
                    y_soft,
                    y_train,
                    sample_weight,
                    X_eval,
                    y_eval,
                    current_eval_labels=current_eval_labels,
                    selection_metric=selection_metric,
                )

                current_test_labels = labels_for_index(smooth_congested_sub[sid], X_test.index)
                f1, precision, recall, transition_f1, trsp, inference_us = evaluate_on_test(
                    model,
                    X_eval,
                    y_eval,
                    X_test,
                    y_test,
                    current_test_labels,
                    current_eval_labels=current_eval_labels,
                    selection_metric=selection_metric,
                )
                collect_grouped_feature_importance(
                    importance_rows,
                    f'transfer_W{transition_weight:g}',
                    hor,
                    model,
                    selected_columns,
                    sensor_ids=[sid],
                )

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

    importance_path = timestamped_output_path(FEATURE_IMPORTANCE_DIR, 'feature_importance_transfer_aggregated.csv')
    write_aggregated_feature_importance(importance_path, importance_rows)
    print(f'Saved aggregated feature importance to {importance_path}')
    print(f'Saved transfer results to {result_path}')
    print('dataset  method  horizon_min  f1_mean  transition_f1_mean  trsp  prec_mean  rec_mean  inference_us_all_nodes  inference_us_per_node')
    for transition_weight, hor in sorted(results_by_weight_horizon.keys()):
        entries = results_by_weight_horizon[(transition_weight, hor)]
        method = f'XGB-Transfer-w{transition_weight:g}'
        row = append_summary_result_row(method, hor, entries)
        print(
            f'{row["dataset"]}  {method:15s} {hor:10d} '
            f'{row["f1_mean"]:.6f}  {row["transition_f1_mean"]:.6f}  {row["trsp"]:.6f}  '
            f'{row["prec_mean"]:.6f}  {row["rec_mean"]:.6f}  '
            f'{row["inference_us_all_nodes"]:22.6f}  {row["inference_us_per_node"]:21.6f}'
        )

    elapsed = time.time() - pipeline_start_time
    append_training_time_row('XGB-Transfer', elapsed)
    elapsed_minutes = elapsed / 60
    elapsed_hours = elapsed / 3600
    print(f'Total transfer training pipeline time: {elapsed_minutes:.2f} minutes ({elapsed_hours:.2f} hours)')


if __name__ == '__main__':
    main()
