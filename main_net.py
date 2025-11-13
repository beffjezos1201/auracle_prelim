import os
from net import main_func
from net.DL_config import Config


###########################################
## Initialize standard config parameters ##
###########################################

## Configuration for the generator and models:
config = Config()
config.data_path = r'S:\Auracle\Seizel\ds005873'             # path to data
config.save_dir = 'net/save_dir'                                # save directory of intermediate and output files
if not os.path.exists(config.save_dir):
  os.mkdir(config.save_dir)
config.fs = 25                                                 # Sampling frequency of the data after post-processing
config.CH = 3                                                   # Nr of accelerometer axes
config.cross_validation = 'fixed'                               # validation type
config.batch_size = 128                                         # batch size
config.frame = 4                                                # window size of input segments in seconds
config.stride = 2                                               # stride between segments (of background EEG) in seconds
config.stride_s = 0.5                                           # stride between segments (of seizure EEG) in seconds
config.boundary = 0.35                                           # relaxed overlap boundary for window labeling
config.factor = 8                                               # balancing factor between nr of segments in each class

## (Removed deep network hyper-parameters not needed for SimpleRule)

###########################################
###########################################

##### INPUT CONFIGS:
config.model = 'SimpleRule'                                  # model architecture 
config.dataset = 'SZ2'                                          # patients to use (check 'datasets' folder)
config.sample_type = 'subsample'                                # sampling method (subsample = remove background EEG segments)
config.add_to_name = 'simple-var'                                      # str to add to the end of the experiment's config name 
config.accel_var_threshold = 0.5                                       # threshold for accel variance rule
config.gyro_var_threshold = None                                       # set (e.g., 0.5) if gyro channels available (channels >=6)

###########################################
###########################################

load_generators = False  # Unused in SimpleRule
save_generators = False  # Unused in SimpleRule

print('Getting predictions on the test set...')
main_func.predict(config)

print('Getting evaluation metrics...')
main_func.evaluate(config)
