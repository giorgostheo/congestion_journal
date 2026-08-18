from shared import *


NAIVE_RESULT_COLUMNS = [
    'method',
    'sid',
    'horizon_min',
    'f1',
    'transition_f1',
    'trsp',
    'precision',
    'recall',
    'inference_us',
]


def append_naive_result_row(path, row):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    header = not path.exists()
    pd.DataFrame([row], columns=NAIVE_RESULT_COLUMNS).to_csv(path, mode='a', header=header, index=False)


def build_future_congestion_target(congestion_series, horizon_min):
    """Match the XGBoost target: any congestion in the next horizon window."""
    horizon_steps = minutes_to_steps(horizon_min)
    future_targets = pd.concat(
        [congestion_series.shift(-step) for step in range(1, horizon_steps + 1)],
        axis=1,
    )
    return future_targets.max(axis=1).fillna(0).astype(int).rename('target')


def split_test_series(target, prediction, current_labels, horizon_min):
    dataset = pd.concat([target, prediction.rename('prediction'), current_labels.rename('current_label')], axis=1)
    dataset['prediction'] = dataset['prediction'].fillna(0).astype(int)
    dataset['current_label'] = dataset['current_label'].fillna(0).astype(int)

    horizon_steps = minutes_to_steps(horizon_min)
    if FIX_FLAGS['tail_drop'] and len(dataset) > horizon_steps:
        # Use the same complete-window test period as the learned models.
        dataset = dataset.iloc[:len(dataset) - horizon_steps]

    n = len(dataset)
    n_train = int(n * 0.7)
    n_eval = int(n * 0.1)
    test = dataset.iloc[n_train + n_eval:]
    return test['target'], test['prediction'], test['current_label']


def evaluate_naive_prediction(target, prediction, current_labels, horizon_min):
    y_test, test_prediction, test_current_labels = split_test_series(target, prediction, current_labels, horizon_min)

    start_inference_time = time.time()
    y_pred = test_prediction.to_numpy(dtype=int)
    inference_time = time.time() - start_inference_time
    inference_time_per_sample_us = (inference_time / len(y_test)) * 1_000_000 if len(y_test) else 0.0

    y_true = y_test.to_numpy(dtype=int)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    transition_f1 = compute_transition_f1(y_true, y_pred, test_current_labels)
    trsp = compute_trsp(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    return f1, precision, recall, transition_f1, trsp, inference_time_per_sample_us


def build_naive_predictions(congestion_series, horizon_min):
    horizon_steps = minutes_to_steps(horizon_min)
    yesterday_steps = minutes_to_steps(24 * 60) - horizon_steps
    last_week_steps = minutes_to_steps(7 * 24 * 60) - horizon_steps
    return {
        'Naive-Current': congestion_series,
        'Naive-Yesterday': congestion_series.shift(yesterday_steps),
        'Naive-LastWeek': congestion_series.shift(last_week_steps),
    }


def main():
    pipeline_start_time = time.time()
    print('=== Naive congestion baselines ===')

    smooth, smooth_congested = load_speed_data()
    smooth_congested_sub = smooth_congested.iloc[:int(len(smooth_congested) * 0.9999)].copy()

    result_path = timestamped_output_path(RESULTS_DIR, 'results_naive.csv')
    results_by_method_horizon = {}

    all_sids = [item for sublist in LOCATIONS for item in sublist]
    total_runs = len(all_sids) * len(HORIZONS_MINUTES)

    progress = tqdm(total=total_runs, desc='Naive baselines', unit='run')
    for sid in all_sids:
        for hor in HORIZONS_MINUTES:
            congestion_series = smooth_congested_sub[sid]
            target = build_future_congestion_target(congestion_series, hor)
            predictions = build_naive_predictions(congestion_series, hor)

            for method, prediction in predictions.items():
                f1, precision, recall, transition_f1, trsp, inference_us = evaluate_naive_prediction(
                    target,
                    prediction,
                    congestion_series,
                    hor,
                )

                append_naive_result_row(result_path, {
                    'method': method,
                    'sid': sid,
                    'horizon_min': hor,
                    'f1': f1,
                    'transition_f1': transition_f1,
                    'trsp': trsp,
                    'precision': precision,
                    'recall': recall,
                    'inference_us': inference_us,
                })

                key = (method, hor)
                if key not in results_by_method_horizon:
                    results_by_method_horizon[key] = []
                results_by_method_horizon[key].append({
                    'f1': f1,
                    'transition_f1': transition_f1,
                    'trsp': trsp,
                    'precision': precision,
                    'recall': recall,
                    'inference_us': inference_us,
                })

            progress.update(1)

    progress.close()

    print(f'Saved naive results to {result_path}')
    print('dataset  method  horizon_min  f1_mean  transition_f1_mean  trsp  prec_mean  rec_mean  inference_us_all_nodes  inference_us_per_node')
    for method, hor in sorted(results_by_method_horizon.keys()):
        entries = results_by_method_horizon[(method, hor)]
        row = append_summary_result_row(method, hor, entries)
        print(
            f'{row["dataset"]}  {method:15s} {hor:10d} '
            f'{row["f1_mean"]:.6f}  {row["transition_f1_mean"]:.6f}  {row["trsp"]:.6f}  '
            f'{row["prec_mean"]:.6f}  {row["rec_mean"]:.6f}  '
            f'{row["inference_us_all_nodes"]:22.6f}  {row["inference_us_per_node"]:21.6f}'
        )

    elapsed = time.time() - pipeline_start_time
    elapsed_minutes = elapsed / 60
    elapsed_hours = elapsed / 3600
    print(f'Total naive evaluation time: {elapsed_minutes:.2f} minutes ({elapsed_hours:.2f} hours)')


if __name__ == '__main__':
    main()
