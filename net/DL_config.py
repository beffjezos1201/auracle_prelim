import tensorflow as tf
import pickle
import os

class Config():
    """ Class to create and store an experiment configuration object with the architecture hyper-parameters, input and sampling types.
    
    Args:
        data_path (str): path to data
        model (str): model architecture (you have 3: Chrononet, EEGnet, DeepConvNet)
        dataset (str): patients split  (check 'datasets' folder)

        fs (int): desired sampling frequency of the input data.
        CH (int): number of channels of the input data.
        frame (int): window size of input segments in seconds.
        stride (float): stride between segments (of background EEG) in seconds
        stride_s (float): stride between segments (of seizure EEG) in seconds
        boundary (float): proportion of seizure data in a window to consider the segment in the positive class
        batch_size (int): batch size for training model
        sample_type (str): sampling method (default is subsample, removes background EEG segments to match the number of seizure segments times the balancing factor)
        factor(int): balancing factor between number of segments in each class. The number of background segments is the number of seizure segments times the balancing factor.
        l2 (float): L2 regularization penalty
        lr (float): learning rate
        dropoutRate (float): layer's dropout rate
        nb_epochs (int): number of epochs to train model
        class_weights (dict): weight of each class for computing the loss function
        cross_validation (str): validation type (default is 'fixed' set of patients for training and validation)
        save_dir (str): save directory for intermediate and output files

    """

    def __init__(self, data_path=None, model='ChronoNet', dataset='SZ2', fs=None, CH=None, frame=2, stride=1, batch_size=64, sample_type='subsample', factor=2, save_dir='savedir', accel_var_threshold=0.5, gyro_var_threshold=None):

        self.data_path = data_path
        self.model = model
        self.dataset = dataset
        self.save_dir = save_dir
        self.fs = fs
        self.CH = CH
        self.frame = frame
        self.stride = stride
        # Removed seizure-specific training strides/boundary not needed for SimpleRule
        self.batch_size = batch_size
        self.sample_type = sample_type
        self.factor = factor
        self.savedir = save_dir
        # Simple rule parameters (only used when model=='SimpleRule')
        self.accel_var_threshold = accel_var_threshold  # variance threshold on accel magnitude
        self.gyro_var_threshold = gyro_var_threshold    # optional variance threshold on gyro magnitude (if gyro present)

    def save_config(self, save_path):
        name = self.get_name()
        with open(os.path.join(save_path, name + '.cfg'), 'wb') as output:  # Overwrites any existing file.
            pickle.dump(self.__dict__, output, pickle.HIGHEST_PROTOCOL)


    def load_config(self, config_path, config_name):
        if not os.path.exists(config_path):
            raise ValueError('Directory is empty or does not exist')

        with open(os.path.join(config_path, config_name), 'rb') as input:
            config = pickle.load(input)

        self.__dict__.update(config)

        
    def get_name(self):
        if hasattr(self, 'add_to_name'):
            return '_'.join([self.model, self.sample_type, 'factor' + str(self.factor),  self.add_to_name])
        else:
            return '_'.join([self.model, self.sample_type, 'factor' + str(self.factor)])
