import os
import numpy as np
from numpy import mod
import pyedflib
import warnings

class Data:
    @staticmethod
    def loadData(data_path, rec, modalities=['mov']):
        """
        Load movement/accelerometer data from EDF files.

        Args:
            data_path: root data directory (e.g. 'S:\\Auracle\\Seizel\\ds005873')
            rec: [subject_id, recording_id]
            modalities: ['mov'] by default

        Returns:
            dict {'mov': np.ndarray with shape (n_channels, n_samples)} or None if not found
        """
        sub, run = rec
        edf_path = os.path.join(data_path, sub, 'ses-01', 'mov', f"{sub}_ses-01_task-szMonitoring_{run}_mov.edf")

        if not os.path.exists(edf_path):
            warnings.warn(f"EDF file not found for {sub} run {run}")
            return None

        try:
            with pyedflib.EdfReader(edf_path) as edf:
                n_channels = edf.signals_in_file
                channels = edf.getSignalLabels()
                fs = edf.getSampleFrequencies()
                
                # Read all channels
                data = []
                for i in range(n_channels):
                    data.append(edf.readSignal(i))
                
                mov_data = np.array(data)
                
                return {
                    'mov': mov_data,
                    'channels': channels,
                    'fs': fs[0] if len(set(fs)) == 1 else fs
                }

        except Exception as e:
            warnings.warn(f"Error reading {edf_path}: {e}")
            return None