class MRI8_NVF4_10mm_0p2mmVVV:
    CSV_FILE_DIR = './data/VF_FVF_DATASETS/CSV/'
    VIDEO_FILE_DIR = './data/VF_FVF_DATASETS/VIDEO/'
    DATA_NAME = 'MRI8_NVF4_10mm_0p2mmVVV'
    CSV_SAMPLE_SPAN = (15000, 127000)
    FILTERING_CSV_METHODS = 'bandpass_filtering'
    BANDPASS_FILTERING_PARAMS = {
                                'passband_edge_freq':[90, 200], 
                                'stopband_edge_freq':[20, 450], 
                                'passband_edge_max_loss':1, 
                                'stopband_edge_min_loss':10
                                }

    BIFURCATION_CONVERT_PARAMS = { 
                                'parameter_sample_step': 1000,
                                'standardize': False, 
                                'mean0':True 
                                }
    BIFURCATION_DISCRETIZATION_PARAMS = { 
                                'upside_down': False,
                                'prominence': 0.01 
                                }
    BIFURCATION_DISCRETIZATION_PARAMS = {
                                        'upside_down': True,
                                        'prominence': 0.1 
                                        }