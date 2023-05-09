class MRI8_NVF4_10mm_0p2mmVVV:
    CSV_FILE_DIR = './data/VF_FVF_DATASETS/CSV/'
    VIDEO_FILE_DIR = './data/VF_FVF_DATASETS/VIDEO/'
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
    LINE_SCANNING_PARAMS_VF = {'position':[50, 40],'width': 40}
    LINE_SCANNING_PARAMS_FVF = {'position':[70, 80],'width': 70}


class MRI8_NVF4_10mm_0p0mmVVV:
    CSV_FILE_DIR = './data/VF_FVF_DATASETS/CSV/'
    VIDEO_FILE_DIR = './data/VF_FVF_DATASETS/VIDEO/'
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
    LINE_SCANNING_PARAMS_VF = {'position':[50, 40],'width': 40}
    LINE_SCANNING_PARAMS_FVF = {'position':[70, 80],'width': 70}

class MRI8_NVF4_10mm_0p0mmVVVV:
    CSV_FILE_DIR = './data/VF_FVF_DATASETS/CSV/'
    VIDEO_FILE_DIR = './data/VF_FVF_DATASETS/VIDEO/'
    DATA_NAME = 'MRI8_NVF4_10mm_0p0mmVVVV'
    ADDITIONAL_PATH = ''
    SAMPLE_SPAN = [161000, 180000]
    FILTER_METHOD = 'bandpass_filtering'
    BANDPASS_PARAMS = {
                    'passband_edge_freq':[100, 450], 
                    'stopband_edge_freq':[50, 800], 
                    'passband_edge_max_loss':1, 
                    'stopband_edge_min_loss':20
                    }
    LINE_SCANNING_PARAMS_VF = {'position':[50, 40],'width': 40}
    LINE_SCANNING_PARAMS_FVF = {'position':[70, 80],'width': 70}