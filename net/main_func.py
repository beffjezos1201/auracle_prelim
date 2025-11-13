import os
import h5py
import numpy as np
from tqdm import tqdm

from net.key_generator import generate_data_keys_sequential
from net.generator_ds import SequentialGenerator
from net.utils import get_metrics_scoring


def predict(config):
    """Generate predictions for test recordings using the SimpleRule.

    Assumes config.model == 'SimpleRule'. Other architectures removed.
    """
    name = config.get_name()
    model_save_path = os.path.join(config.save_dir, 'models', name)

    # Load saved config (to restore thresholds etc.) if present
    cfg_dir = os.path.join(model_save_path, 'configs')
    if os.path.isdir(cfg_dir):
        try:
            config.load_config(config_path=cfg_dir, config_name=name + '.cfg')
        except Exception:
            print('Warning: could not load saved config, using current settings.')

    # Ensure prediction directory
    pred_dir = os.path.join(config.save_dir, 'predictions', name)
    os.makedirs(pred_dir, exist_ok=True)

    # Build test recording list
    import pandas as pd
    test_pats_list = pd.read_csv(os.path.join('net', 'datasets', config.dataset + '_test.tsv'), sep='\t', header=None, skiprows=[0, 1, 2])[0].to_list()
    test_recs_list = []
    for s in test_pats_list:
        mov_path = os.path.join(config.data_path, s, 'ses-01', 'mov')
        if os.path.exists(mov_path):
            test_recs_list.extend([[s, r.split('_')[-2]] for r in os.listdir(mov_path) if 'edf' in r])
        else:
            print(f"Warning: Skipping {s} - mov folder not found")

    from net.simplerule import predict_simple_variance
    print(f"Scoring with per-recording min–max normalized variance (no fixed threshold). boundary={getattr(config,'boundary',None)}")
    gyro_thr = getattr(config, 'gyro_var_threshold', None)
    
    total_pred_pos = 0
    total_true_pos = 0
    total_windows = 0

    for rec in tqdm(test_recs_list):
        out_file = os.path.join(pred_dir, rec[0] + '_' + rec[1] + '_preds.h5')
        if os.path.isfile(out_file):
            print(rec[0] + ' ' + rec[1] + ' exists. Skipping...')
            continue

        segments = generate_data_keys_sequential(config, [rec], verbose=False)
        if len(segments) == 0:
            continue

        gen_test = SequentialGenerator(config, [rec], segments, batch_size=len(segments), shuffle=False, verbose=False)
        y_pred, y_true = predict_simple_variance(gen_test, accel_var_threshold=None, gyro_var_threshold=gyro_thr)

        with h5py.File(out_file, 'w') as f:
            f.create_dataset('y_pred', data=y_pred)
            f.create_dataset('y_true', data=y_true)

        # Logging counts
        pred_pos = int(np.sum(y_pred))
        true_pos = int(np.sum(y_true))
        windows = len(y_true)
        if windows == 0:
            print(f"WARNING: No windows produced for {rec}. Check duration and key generation.")
        total_pred_pos += pred_pos
        total_true_pos += true_pos
        total_windows += windows
        print(f"[{rec[0]} {rec[1]}] windows={windows} pred_pos={pred_pos} true_pos={true_pos}")

    print(f"Total windows={total_windows} total_pred_pos={total_pred_pos} total_true_pos={total_true_pos}")

   
#######################################################################################################################
#######################################################################################################################


def evaluate(config):

    name = config.get_name()

    pred_path = os.path.join(config.save_dir, 'predictions', name)
    pred_fs = 1.0 / float(getattr(config, 'stride', 2))  # labels sampled every "stride" seconds

    thresholds = list(np.around(np.linspace(0,1,51),2))

    x_plot = np.linspace(0, 200, 200)

    if not os.path.exists(os.path.join(config.save_dir, 'results')):
        os.mkdir(os.path.join(config.save_dir, 'results'))

    result_file = os.path.join(config.save_dir, 'results', name + '.h5')

    sens_ovlp = []
    prec_ovlp = []
    fah_ovlp = []
    sens_ovlp_plot = []
    prec_ovlp_plot = []
    f1_ovlp = []

    sens_epoch = []
    spec_epoch = []
    prec_epoch = []
    fah_epoch = []
    f1_epoch = []

    score = []

    pred_files = [x for x in os.listdir(pred_path)]
    pred_files.sort()

    for file in tqdm(pred_files):
        with h5py.File(os.path.join(pred_path, file), 'r') as f:
            y_pred = list(f['y_pred'])
            y_true = list(f['y_true'])

        sens_ovlp_th = []
        prec_ovlp_th = []
        fah_ovlp_th = []
        f1_ovlp_th = []

        sens_epoch_th = []
        spec_epoch_th = []
        prec_epoch_th = []
        fah_epoch_th = []
        f1_epoch_th = []

        score_th = []

        # rec info no longer used; removed for lean evaluation

        for th in thresholds:
            eval_score_rec, sens_ovlp_rec, prec_ovlp_rec, FA_ovlp_rec, f1_ovlp_rec, sens_epoch_rec, spec_epoch_rec, prec_epoch_rec, FA_epoch_rec, f1_epoch_rec = get_metrics_scoring(y_pred, y_true, pred_fs, th)

            sens_ovlp_th.append(sens_ovlp_rec)
            prec_ovlp_th.append(prec_ovlp_rec)
            fah_ovlp_th.append(FA_ovlp_rec)
            f1_ovlp_th.append(f1_ovlp_rec)
            sens_epoch_th.append(sens_epoch_rec)
            spec_epoch_th.append(spec_epoch_rec)
            prec_epoch_th.append(prec_epoch_rec)
            fah_epoch_th.append(FA_epoch_rec)
            f1_epoch_th.append(f1_epoch_rec)
            score_th.append(eval_score_rec)

        sens_ovlp.append(sens_ovlp_th)
        prec_ovlp.append(prec_ovlp_th)
        fah_ovlp.append(fah_ovlp_th)
        f1_ovlp.append(f1_ovlp_th)

        sens_epoch.append(sens_epoch_th)
        spec_epoch.append(spec_epoch_th)
        prec_epoch.append(prec_epoch_th)
        fah_epoch.append(fah_epoch_th)
        f1_epoch.append(f1_epoch_th)

        score.append(score_th)

        to_cut = np.argmax(fah_ovlp_th)
        fah_ovlp_plot_rec = fah_ovlp_th[to_cut:]
        sens_ovlp_plot_rec = sens_ovlp_th[to_cut:]
        prec_ovlp_plot_rec = prec_ovlp_th[to_cut:]

        y_plot = np.interp(x_plot, fah_ovlp_plot_rec[::-1], sens_ovlp_plot_rec[::-1])
        sens_ovlp_plot.append(y_plot)
        y_plot = np.interp(x_plot, sens_ovlp_plot_rec[::-1], prec_ovlp_plot_rec[::-1])
        prec_ovlp_plot.append(y_plot)

    score_05 = [x[25] for x in score]
    sens_05 = [x[25] for x in sens_ovlp]
    fah_05 = [x[25] for x in fah_epoch]

    print('\n' + '='*60)
    print('EVALUATION RESULTS (at threshold = 0.5)')
    print('='*60)
    print(f'Evaluation Score: {np.nanmean(score_05):.2f}')
    print(f'Sensitivity (any-overlap): {np.nanmean(sens_05):.4f}')
    print(f'False Alarm Rate (per hour): {np.nanmean(fah_05):.2f}')
    print('='*60 + '\n')
 
# Auto-pick the lowest-FA threshold that achieves at least 50% mean sensitivity
    sens_arr = np.array(sens_ovlp, dtype=float)
    fae_arr = np.array(fah_epoch, dtype=float)
    sens_mean = np.nanmean(sens_arr, axis=0)
    fae_mean = np.nanmean(fae_arr, axis=0)
    candidates = np.where(sens_mean >= 0.50)[0]
    if candidates.size > 0:
        best_idx = candidates[np.argmin(fae_mean[candidates])]
    else:
        best_idx = int(np.nanargmax(sens_mean))
    best_th = thresholds[best_idx]
    print('='*60)
    print(f'Auto-selected threshold th={best_th:.2f}')
    print(f'Mean Sensitivity (any-overlap): {sens_mean[best_idx]:.4f}')
    print(f'Mean False Alarm Rate (per hour): {fae_mean[best_idx]:.2f}')
    print('='*60 + '\n')


    with h5py.File(result_file, 'w') as f:
        f.create_dataset('sens_ovlp', data=sens_ovlp)
        f.create_dataset('prec_ovlp', data=prec_ovlp)
        f.create_dataset('fah_ovlp', data=fah_ovlp)
        f.create_dataset('f1_ovlp', data=f1_ovlp)
        f.create_dataset('sens_ovlp_plot', data=sens_ovlp_plot)
        f.create_dataset('prec_ovlp_plot', data=prec_ovlp_plot)
        f.create_dataset('x_plot', data=x_plot)
        f.create_dataset('sens_epoch', data=sens_epoch)
        f.create_dataset('spec_epoch', data=spec_epoch)
        f.create_dataset('prec_epoch', data=prec_epoch)
        f.create_dataset('fah_epoch', data=fah_epoch)
        f.create_dataset('f1_epoch', data=f1_epoch)
        f.create_dataset('score', data=score)

