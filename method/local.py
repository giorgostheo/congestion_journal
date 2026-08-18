from shared import *


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


def main():
    pipeline_start_time = time.time()
    print('=== Local model: per-sensor XGBoost ===')
    print(f'Transition sample weights: {TRANSITION_WEIGHTS}')
    print(f'Selection metric: {XGB_SELECTION_METRIC}')
    print(f'Feature flags: {describe_feature_flags()}')

    smooth, smooth_congested = load_speed_data()
    smooth_sub = smooth.iloc[:int(len(smooth) * 0.9999)].copy()
    smooth_congested_sub = smooth_congested.iloc[:int(len(smooth_congested) * 0.9999)].copy()

    result_path = timestamped_output_path(RESULTS_DIR, 'results_local.csv')
    importance_rows = []
    results_by_weight_horizon = {}

    all_sids = [item for sublist in LOCATIONS for item in sublist]
    total_runs = len(all_sids) * len(HORIZONS_MINUTES) * len(TRANSITION_WEIGHTS)

    progress = tqdm(total=total_runs, desc='Local model', unit='run')
    for transition_weight in TRANSITION_WEIGHTS:
        for sid in all_sids:
            for hor in HORIZONS_MINUTES:
                features = build_single_sensor_features(smooth_sub[sid], hor, congestion_series=smooth_congested_sub[sid])
                X_train, X_eval, X_test, y_train, y_eval, y_test = prepare_train_eval_test_split(
                    features,
                    smooth_congested_sub[sid],
                    hor,
                )

                current_train_labels = labels_for_index(smooth_congested_sub[sid], X_train.index)
                current_eval_labels = labels_for_index(smooth_congested_sub[sid], X_eval.index)
                sample_weight = transition_sample_weights(current_train_labels, y_train, transition_weight)
                selection_metric = selection_metric_for_weight(transition_weight)

                model, best_params, _ = find_best_xgboost_model_weighted(
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
                    f'local_W{transition_weight:g}',
                    hor,
                    model,
                    features.columns,
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
                    'n_features': len(features.columns),
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

    importance_path = timestamped_output_path(FEATURE_IMPORTANCE_DIR, 'feature_importance_local_aggregated.csv')
    write_aggregated_feature_importance(importance_path, importance_rows)
    print(f'Saved aggregated feature importance to {importance_path}')
    print(f'Saved local results to {result_path}')
    print('dataset  method  horizon_min  f1_mean  transition_f1_mean  trsp  prec_mean  rec_mean  inference_us_all_nodes  inference_us_per_node')
    for transition_weight, hor in sorted(results_by_weight_horizon.keys()):
        entries = results_by_weight_horizon[(transition_weight, hor)]
        method = f'XGB-Local-w{transition_weight:g}'
        row = append_summary_result_row(method, hor, entries)
        print(
            f'{row["dataset"]}  {method:15s} {hor:10d} '
            f'{row["f1_mean"]:.6f}  {row["transition_f1_mean"]:.6f}  {row["trsp"]:.6f}  '
            f'{row["prec_mean"]:.6f}  {row["rec_mean"]:.6f}  '
            f'{row["inference_us_all_nodes"]:22.6f}  {row["inference_us_per_node"]:21.6f}'
        )

    elapsed = time.time() - pipeline_start_time
    append_training_time_row('XGB-Local', elapsed)
    elapsed_minutes = elapsed / 60
    elapsed_hours = elapsed / 3600
    print(f'Total local training pipeline time: {elapsed_minutes:.2f} minutes ({elapsed_hours:.2f} hours)')


if __name__ == '__main__':
    main()
