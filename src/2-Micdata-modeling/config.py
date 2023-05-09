class MRI8_NVF4_10mm_0p2mmVVV:
    CSV_FILE_DIR = './data/VF_FVF_DATASETS/CSV/'
    DATA_NAME = 'MRI8_NVF4_10mm_0p2mmVVV'
    ADDITIONAL_PATH = ''
    SAMPLE_SPAN = [160000, 180000]
    FILTER_METHOD = 'bandpass_filtering'
    BANDPASS_PARAMS = {
                    'passband_edge_freq':[100, 350], 
                    'stopband_edge_freq':[20, 800], 
                    'passband_edge_max_loss':1, 
                    'stopband_edge_min_loss':40
                    }
    ESN_PARAMS = {
                'units':500, 
                'SR':0.1, 
                'input_shape':10, 
                'output_shape':10, 
                'W_in_scale':0.9, 
                'W_res_density':0.1, 
                'leak_rate':1.0, 
                'alpha':0.0005, 
                'seed':0
                }

    TRAIN_TEST_DATA_PARAMS = {
                            'n_shift':10,
                            'n_dimension':10,
                            'n_train':5000,
                            'n_predstep':1
                            }
    MODEL_DIR = './results/esn_model/'

class MRI8_NVF4_10mm_0p0mmVVV:
    CSV_FILE_DIR = './data/VF_FVF_DATASETS/CSV/'
    DATA_NAME = 'MRI8_NVF4_10mm_0p0mmVVV'
    ADDITIONAL_PATH = ''
    SAMPLE_SPAN = [160000, 180000]
    FILTER_METHOD = 'bandpass_filtering'
    BANDPASS_PARAMS = {
                    'passband_edge_freq':[100, 350], 
                    'stopband_edge_freq':[20, 800], 
                    'passband_edge_max_loss':1, 
                    'stopband_edge_min_loss':40
                    }
    ESN_PARAMS = {
                'units':500, 
                'SR':0.1, 
                'input_shape':10, 
                'output_shape':10, 
                'W_in_scale':0.5, 
                'W_res_density':0.1, 
                'leak_rate':1.0, 
                'alpha':0.0005, 
                'seed':0
                }

    TRAIN_TEST_DATA_PARAMS = {
                            'n_shift':10,
                            'n_dimension':10,
                            'n_train':5000,
                            'n_predstep':1
                            }
    MODEL_DIR = './results/esn_model/'

class MRI8_NVF4_10mm_0p0mmVVVV:
    CSV_FILE_DIR = './data/VF_FVF_DATASETS/CSV/'
    DATA_NAME = 'MRI8_NVF4_10mm_0p0mmVVVV'
    ADDITIONAL_PATH = ''
    SAMPLE_SPAN = [160000, 180000]
    FILTER_METHOD = 'bandpass_filtering'
    BANDPASS_PARAMS = {
                    'passband_edge_freq':[100, 450], 
                    'stopband_edge_freq':[50, 800], 
                    'passband_edge_max_loss':1, 
                    'stopband_edge_min_loss':20
                    }
    ESN_PARAMS = {
                'units':500, 
                'SR':1.1, 
                'input_shape':10, 
                'output_shape':10, 
                'W_in_scale':0.3, 
                'W_res_density':0.1, 
                'leak_rate':1.0, 
                'alpha':0.01, 
                'seed':0
                }
    TRAIN_TEST_DATA_PARAMS = {
                            'n_shift':10,
                            'n_dimension':10,
                            'n_train':5000,
                            'n_predstep':1
                            }
    MODEL_DIR = './results/esn_model/'