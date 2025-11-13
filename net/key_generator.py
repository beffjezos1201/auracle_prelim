import numpy as np
from tqdm import tqdm
from classes.annotation import Annotation
from classes.data import Data


def _max_overlap_fraction(win_start, win_end, events):
    """Compute maximum overlap fraction of window with list of events."""
    if not events:
        return 0.0
    frame_len = win_end - win_start
    max_frac = 0.0
    for ev_start, ev_end in events:
        overlap = max(0.0, min(win_end, ev_end) - max(win_start, ev_start))
        if overlap > 0:
            frac = overlap / frame_len
            if frac > max_frac:
                max_frac = frac
    return max_frac


def generate_data_keys_sequential(config, recs_list, verbose=True):
    """Unified overlap-based sequential segment key generator.

    For each recording it slides windows of length `config.frame` every `config.stride`
    seconds across the whole recording duration. Each window is labeled 1 if the
    maximum fractional overlap with any seizure event >= config.boundary (default 0.5), else 0.

    Returns list/array of segment keys: [recording_index, start_sec, end_sec, label].
    """
    boundary = getattr(config, 'boundary', 0.5)
    segments = []

    for idx, rec in tqdm(enumerate(recs_list), disable=not verbose):
        ann = Annotation.loadAnnotation(config.data_path, rec)
        rec_duration = float(getattr(ann, 'rec_duration', 0.0))
        if rec_duration <= 0:
            # Fallback: derive duration from EDF movement data
            rec_data = Data.loadData(config.data_path, rec, modalities=['mov'])
            if rec_data and 'mov' in rec_data:
                # Use sampling frequency from file if available else config.fs
                fs_mov = rec_data.get('fs', config.fs)
                rec_duration = rec_data['mov'].shape[1] / fs_mov
                if verbose:
                    print(f"Derived duration from EDF for {rec}: {rec_duration:.2f}s")
            else:
                if verbose:
                    print(f"No data to derive duration for {rec}; skipping.")
                continue
        # rec_duration now ensured >0
        frame = float(config.frame)
        stride = float(config.stride)

        if rec_duration < frame:
            if verbose:
                print(f"Recording too short (< frame): {rec} duration={rec_duration:.2f}s")
            continue

        starts = np.arange(0.0, rec_duration - frame + 1e-6, stride)
        ends = starts + frame

        if ann.events:
            labels = [1 if _max_overlap_fraction(s, e, ann.events) >= boundary else 0 for s, e in zip(starts, ends)]
        else:
            labels = [0] * len(starts)

        seg_arr = np.column_stack(([idx] * len(starts), starts, ends, labels))
        segments.extend(seg_arr)

    return segments
