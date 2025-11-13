import numpy as np

def predict_simple_variance(generator, accel_var_threshold=0.5, gyro_var_threshold=None):
    """Predict seizure segment scores using variance of sensor magnitudes.

    Returns continuous scores in [0,1] by per-recording min–max normalization.
    No fixed thresholding is applied here.
    """
    var_all = []
    y_true_all = []
    for i in range(len(generator)):
        x, y = generator[i]  # x: [B, T, CH], y: [B, 2]
        ch = x.shape[2]

        # Accel magnitude variance
        accel_axes = min(3, ch)
        accel_mag = np.linalg.norm(x[:, :, :accel_axes], axis=2)  # [B, T]
        accel_var = np.var(accel_mag, axis=1)                     # [B]

        # Optional gyro: take max to be more sensitive if present
        if ch >= 6:
            gyro_mag = np.linalg.norm(x[:, :, 3:6], axis=2)
            gyro_var = np.var(gyro_mag, axis=1)
            var_batch = np.maximum(accel_var, gyro_var)
        else:
            var_batch = accel_var

        var_all.append(var_batch.astype(np.float32))
        y_true_all.append(y[:, 1].astype(np.uint8))

    var_all = np.concatenate(var_all, axis=0) if var_all else np.array([], dtype=np.float32)
    # Per-recording min–max scaling to [0,1]
    if var_all.size and float(var_all.max()) > float(var_all.min()):
        y_pred = (var_all - var_all.min()) / (var_all.max() - var_all.min())
    else:
        y_pred = np.zeros_like(var_all, dtype=np.float32)

    y_true = np.concatenate(y_true_all, axis=0) if y_true_all else np.array([], dtype=np.uint8)
    return y_pred.astype(np.float32), y_true