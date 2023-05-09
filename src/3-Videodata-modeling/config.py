class MRI8_NVF4_10mm_0p0mmVVVV:
    CSV_FILE_DIR = './data/VF_FVF_DATASETS/CSV/'
    VIDEO_FILE_DIR = './data/VF_FVF_DATASETS/VIDEO/'
    DATA_NAME = 'MRI8_NVF4_10mm_0p0mmVVVV'
    CSV_ADDITIONAL_PATH = ''
    SAMPLE_SPAN = [40000, 140000]
    LINE_SCANNING_PARAMS_VF = {'position':[50, 40],'width': 40}
    LINE_SCANNING_PARAMS_FVF = {'position':[70, 80],'width': 70}
    FILTERING_VIDEO_PARAMS_VF = {'kernel_length':10, 'kernel_size':3}
    FILTERING_VIDEO_PARAMS_FVF = {'kernel_length':10, 'kernel_size':3}
    AUTOENCODER_METHOD = 'MLP'
    AUTOENCODER_PARAMS_VF = { 'hidden_layers_shape': [20],
                            'latent_dim': 1,
                            'latent_regularizer': None,
                            'random_state': 0,
                            'learning_rate': 1e-3,
                            'loss_function': 'mse',
                            'epochs': 300,
                            'batch_size': 1024,
                            'verbose': 0,
                            'callbacks': None }
    AUTOENCODER_PARAMS_FVF = { 'hidden_layers_shape': [35],
                            'latent_dim': 1,
                            'latent_regularizer': None,
                            'random_state': 0,
                            'learning_rate': 1e-3,
                            'loss_function': 'mse',
                            'epochs': 300,
                            'batch_size': 1024,
                            'verbose': 0,
                            'callbacks': None }
    BF_CONVERT_PARAMS = { 'parameter_sample_step': 1000,'standardize': True, 'mean0':False }
    BF_DISCRETIZATION_PARAMS_VF = { 'upside_down': True,'prominence': 1 }
    BF_DISCRETIZATION_PARAMS_FVF = { 'upside_down': False,'prominence': 0.1 }
    SINDY_PARAM_LIM = (55, 90)
    SINDY_TRAIN_N_SAMPLE_VF = 6
    SINDY_TRAIN_PARAMS_LIST_VF = [0, 1, 2, 3, 4, 5,  6,  8, 9, 10, 11,  12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24,    26, 27, 28,     30, 31, 32, 33, 34]
    SINDY_PARAMS_VF = {'order':7, 'threshold':0.1, 'alpha':1e-1, }
    SINDY_FREERUN_PARAMS_VF = {'data0':1.5, 'parameter_start_stop_step':(0.75, 1.001, 0.0001), 'n_run':10, 'n_idling_run':10}
    SINDY_TRAIN_N_SAMPLE_FVF = 15
    SINDY_TRAIN_PARAMS_LIST_FVF = [0,   5,   10, 11, 12, 13, 14,   15, 16, 17,   19,   22, 21, 23, 24, 25,   27, 28, 29,    34]
    SINDY_PARAMS_FVF = {'order':7, 'threshold':0.1, 'alpha':1e-1, }
    SINDY_FREERUN_PARAMS_FVF = {'data0':1.0, 'parameter_start_stop_step':(0.75, 1.001, 0.0001), 'n_run':10, 'n_idling_run':10}
    LV_DIR = './results/lv_of_video/'
    SINDY_MODEL_DIR = './results/sindy_model/'