import numpy as np
import pysindy as ps
import matplotlib.pylab as plt
from matplotlib import gridspec
from matplotlib.colors import Normalize
import matplotlib.ticker as ticker
from lib.functions import *

########## LOAD DATA ##########
def LoadCSV(file_path, data_name_list, sample_span):
    data_list, _ = load_csv_data(file_path, data_name_list, sample_span)
    return data_list

def LoadCSVandVIDEOS(csv_path, data_name_list, sample_span, 
                      videos_path_list, videos_shooting_time_interval=1/10000):
    csv_data_list, _ = load_csv_data(csv_path, data_name_list, sample_span)
    ### generate new csv time data　(to match the time data of csv and video)
    csv_t_data_list, _ = load_csv_data(csv_path, ['t'], [0, None], quiet=True)
    csv_t_data = csv_t_data_list[0]
    csv_t_data_new = (np.arange(0, csv_t_data.shape[0])*(csv_t_data[1]-csv_t_data[0]))[sample_span[0]:sample_span[1]]
    ### LOAD VIDEO (same csv span)
    video_data_list = []
    video_t_data_list = []
    for i in range(len(videos_path_list)):
        video_data, video_t_data = load_video_data(
                                            video_path=videos_path_list[i],
                                            time_span=[csv_t_data_new[0], csv_t_data_new[-1]],
                                            shooting_time_interval=videos_shooting_time_interval
                                            )
        video_data_list.append(video_data)
        video_t_data_list.append(video_t_data+csv_t_data[0])
    return csv_data_list, video_data_list, video_t_data_list


########## PREPROCESSING ##########
class PreProcessing():

    def __init__(self, data, t_data, video_data_list=None, video_t_data_list=None):
        self.raw_data = data
        self.data = data
        self.t_data = t_data
        self.video_data_list = video_data_list
        self.video_t_data_list = video_t_data_list
        if video_data_list==None and video_t_data_list==None:
            self.preprocess_video = False
        else:
            self.preprocess_video = True

    def cut(self, sample_span, new_t=False):
        self.data = self.data[sample_span[0]:sample_span[1]]
        self.t_data = self.t_data[sample_span[0]:sample_span[1]]
        video_data_list_cut = []
        video_t_data_list_cut = []
        if self.preprocess_video:
            for i in range(len(self.video_data_list)):
                idx_start = np.abs(np.asarray(self.video_t_data_list[i])-self.t_data[0]).argmin()
                idx_stop = np.abs(np.asarray(self.video_t_data_list[i])-self.t_data[-1]).argmin()
                video_data_list_cut.append(self.video_data_list[i][idx_start:idx_stop])
                video_t_data_list_cut.append(self.video_t_data_list[i][idx_start:idx_stop])
        if new_t: ### t0=0, t1=0+dt, t2=0+2dt ...
            self.t_data = np.arange(0, self.t_data.shape[0])*(self.t_data[1]-self.t_data[0])
            video_t_data_list_cut = []
            if self.preprocess_video:
                for i in range(len(self.video_data_list)):
                    video_t_data_list_cut.append(np.arange(0, self.video_t_data_list[i].shape[0])*(self.video_t_data_list[i][1]-self.video_t_data_list[i][0]))
        self.video_data_list = video_data_list_cut
        self.video_t_data_list = video_t_data_list_cut

    def filter(self, method='bandpass_filtering', params={'passband_edge_freq':np.array([90, 200]), 'stopband_edge_freq':np.array([20, 450]), 'passband_edge_max_loss':1, 'stopband_edge_min_loss':10}):
        if method=='bandpass_filtering':
            self.data = bandpass_filter(
                                        data=self.data,
                                        t_data=self.t_data,
                                        passband_edge_freq=params['passband_edge_freq'],
                                        stopband_edge_freq=params['stopband_edge_freq'],
                                        passband_edge_max_loss=params['passband_edge_max_loss'],
                                        stopband_edge_min_loss=params['stopband_edge_min_loss'],
                                        )
        else:
            print('There is no such method.')
    
    def filter_video(self, params_list=[{'kernel_length':10, 'kernel_size':3}]):
        video_data_list_filtered = []
        for i in range(len(self.video_data_list)):
            video_data_list_filtered.append(video_filtering(self.video_data_list[i], kernel_length=params_list[i]['kernel_length'], kernel_size=params_list[i]['kernel_size']))
        self.video_data_list = video_data_list_filtered

    def linescanning_video(self, params_list=[{'position':[70, 80], 'width':70}]):
        video_data_list_scanned = []
        for i in range(len(self.video_data_list)):
            video_data_list_scanned.append(line_scanning(self.video_data_list[i], position=params_list[i]['position'], width=params_list[i]['width']))
        self.video_data_list = video_data_list_scanned

    def standardize_video(self):
        video_data_list_standardized = []
        for i in range(len(self.video_data_list)):
            video_data_list_standardized.append(standardize(self.video_data_list[i]))
        self.video_data_list = video_data_list_standardized

class PreProcessing_forESN():

    def __init__(self, data, t_data):
        self.raw_data = data
        self.data = data
        self.t_data = t_data

    def cut(self, span, new_t=False):
        self.data = self.data[span[0]:span[1]]
        self.t_data = self.t_data[span[0]:span[1]]
        if new_t: ### t0=0, t1=0+dt, t2=0+2dt ...
            self.t_data = np.arange(0, self.t_data.shape[0])*(self.t_data[1]-self.t_data[0])

    def filter(self, method='bandpass_filtering', params={'passband_edge_freq':np.array([90, 200]), 'stopband_edge_freq':np.array([20, 450]), 'passband_edge_max_loss':1, 'stopband_edge_min_loss':10}):
        if method=='bandpass_filtering':
            self.data = bandpass_filter(
                                        data=self.data,
                                        t_data=self.t_data,
                                        passband_edge_freq=params['passband_edge_freq'],
                                        stopband_edge_freq=params['stopband_edge_freq'],
                                        passband_edge_max_loss=params['passband_edge_max_loss'],
                                        stopband_edge_min_loss=params['stopband_edge_min_loss'],
                                        )
        else:
            print('There is no such method.')

    def embed(self, n_shift, n_dimension):
        length = len(self.data) - ((n_dimension-1)*n_shift)
        data_embedded = np.zeros((length, n_dimension))
        for i in range(n_dimension):
            data_embedded[:, i] = np.roll(self.data, -i*n_shift)[:-((n_dimension-1)*n_shift)]
        self.data_embedded = data_embedded
        self.t_data_embedded = self.t_data[:data_embedded.shape[0]]
        self.n_shift = n_shift
        self.n_dimension = n_dimension

    def train_test_split(self, n_train, n_predstep):
        train_X = self.data_embedded[:(n_train-n_predstep-(self.data_embedded.shape[1]-1)*self.n_shift)].reshape(-1, 1, self.data_embedded.shape[1])
        train_Y = self.data_embedded[n_predstep:(n_train-(self.data_embedded.shape[1]-1)*self.n_shift)].reshape(-1, 1, self.data_embedded.shape[1])
        test_X = self.data_embedded[n_train:-n_predstep].reshape(-1, 1, self.data_embedded.shape[1])
        test_Y = self.data_embedded[n_train+n_predstep:].reshape(-1, 1, self.data_embedded.shape[1])
        return train_X, train_Y, test_X, test_Y


########## CONVERTING ##########
class  BifurcationConvert():

    def __init__(self, data, parameter_data, t_data=None):
        self.data = data
        self.parameter_data = parameter_data
        self.t_data = t_data
        
    def convert(self, params={'parameter_sample_step':1000, 'standardize':True, 'mean0':True}):
        ### for parameter data
        sample_step_paramdata = params['parameter_sample_step']
        start_points_paramdata = (np.arange(0, self.parameter_data.shape[0], sample_step_paramdata)).astype(int)
        ### for data
        sample_step_data = int(sample_step_paramdata*(self.data.shape[0]/self.parameter_data.shape[0]))
        start_points_data = (start_points_paramdata*(self.data.shape[0]/self.parameter_data.shape[0])).astype(int)
        ### convert
        parameter_data_list = []
        convert_data_list = []
        convert_t_data_list = []
        for i in range(len(start_points_paramdata)):
            parameter_data_list.append(np.average(self.parameter_data[start_points_paramdata[i]:start_points_paramdata[i]+sample_step_paramdata]))
            convert_data_list.append(self.data[start_points_data[i]:start_points_data[i]+sample_step_data])
            if self.t_data.all()!=None:
                convert_t_data_list.append(self.t_data[start_points_data[i]:start_points_data[i]+sample_step_data])
        self.bifurcation_data_list = convert_data_list
        self.bifurcation_parameter_list = parameter_data_list
        if self.t_data.all()!=None:
            self.bifurcation_t_data_list = convert_t_data_list
        if params['standardize']:
            convert_data_list_standardized = []
            for i in range(len(convert_data_list)):
                convert_data_list_standardized.append(standardize(convert_data_list[i]))
            self.bifurcation_data_list = convert_data_list_standardized
        if params['mean0']:
            convert_data_list_mean0 = []
            for i in range(len(convert_data_list)):
                convert_data_list_mean0.append(mean0(convert_data_list[i]))
            self.bifurcation_data_list = convert_data_list_mean0
            
    def discretization(self,  params={'upside_down':True, 'prominence':2.}):
        if self.bifurcation_data_list[0].ndim!=1:
            print('Warning: Data is not 1D. Convert to 1D and continue to run.')

        if params['upside_down']:
            upside_down = -1
        else:
            upside_down = 1

        peaks_index_list = []
        peaks_list = []
        bifurcation_data_list_upside_down = []
        for i in range(len(self.bifurcation_data_list)):
            if self.bifurcation_data_list[0].ndim!=1:
                peaks, peaks_index = fing_peaks_index(self.bifurcation_data_list[i].rehape(-1)*upside_down, params['prominence'])
            else:
                peaks, peaks_index = fing_peaks_index(self.bifurcation_data_list[i]*upside_down, params['prominence'])
            peaks_index_list.append(peaks_index)
            peaks_list.append(peaks)
            bifurcation_data_list_upside_down.append(self.bifurcation_data_list[i]*upside_down)
        self.bifurcation_data_list_discreted = peaks_list
        self.bifurcation_data_list_discreted_index = peaks_index_list
        self.bifurcation_data_list = bifurcation_data_list_upside_down

class TimeDelayEmbedding:

    def __init__(self, data, lags_num, bins=16):
        self.data = data
        self.lags_num = lags_num
        self.bins = bins
        self.data_discrete = self.discretization()

    # Discretization of continuous data
    def discretization(self):
        hist, bins = np.histogram(self.data, bins=self.bins, density=True)
        bins_indices = np.digitize(self.data, bins)
        data_discrete = self.data[bins_indices]
        return data_discrete

    ### Mutual Information
    def mutual_information(self):
        mutual_info = np.zeros(self.lags_num)
        for i in range(0, self.lags_num):
            unlagged = self.data_discrete[:-i]
            lagged = np.roll(self.data_discrete, -i)[:-i]
            mutual_info[i] = mutual_info_score(unlagged, lagged)
        ### find time delay
        mutual_info_min = None
        for i in range(0, self.lags_num):
            if mutual_info_min is None and i > 1 and mutual_info[i - 1] < mutual_info[i]:
                mutual_info_min = i - 1 # first　minimum time-delay
        if mutual_info_min is None:
            mutual_info_min = 0 # cannot find time-delay
        return  mutual_info, mutual_info_min
    
    ### Generate time delay coordinates
    def timedelay_embedding(data, dim, n_shift):
        ### the length of time delay data
        td_data_len = len(data) - ((dim - 1)*n_shift)
        ### generate time delay coordinates
        td_data = np.zeros((td_data_len, dim))
        for i in range(dim):
            td_data[:, i] = np.roll(data, -i * n_shift)[:-((dim - 1)*n_shift)]
        return td_data
    

########## MACHINE LEARNING ##########
class EchoStateNetwork():

    def __init__(self, units=256, SR=0.99, input_shape=1, output_shape=1, W_in_scale=1.0, W_res_density=0.1, leak_rate=1.0, alpha=1.0e-4, seed=0):
        self.units = units
        self.SR = SR
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.W_in_scale = W_in_scale
        self.W_res_density = W_res_density
        self.leak_rate = leak_rate
        self.alpha = alpha
        set_seed(seed)
        self.W_in = np.random.uniform(size=(self.input_shape, self.units), low=-self.W_in_scale, high=self.W_in_scale)
        self.W_res = W_res_create(shape=(self.units, self.units), SR=self.SR, density=self.W_res_density)
        self.bias = np.random.uniform(size=(1, self.units), low=-0.1, high=0.1)
        self.x_n = np.random.uniform(size=(1, self.units))
        self.W_out =np.random.uniform(size=(self.units, self.output_shape))

    def fit(self, in_layer_data, out_layer_data):
        x = in_layer_data
        y = out_layer_data
        Xt_X, Xt_Y = 0.0, 0.0
        for i in tqdm.tqdm(range(x.shape[0]), desc="Learning", leave=False):
            In = np.matmul([x[i,:]], self.W_in)
            Res = np.matmul(self.x_n, self.W_res)
            self.x_n = ((1.0 - self.leak_rate) * self.x_n + self.leak_rate * np.tanh(In + Res)).reshape(1, self.units)
            y_n = (y[i,:]).reshape(1, self.output_shape)
            Xt_X += np.matmul(np.transpose(self.x_n), self.x_n)
            Xt_Y += np.matmul(np.transpose(self.x_n), y_n)
        Xt_X_aI = Xt_X + self.alpha * np.eye(int(self.units))
        self.W_out = np.matmul(np.linalg.inv(Xt_X_aI), Xt_Y)
        self.opt_x_n = self.x_n

    def predict(self, in_layer_data, return_reservoir=False):
        x = in_layer_data
        ans, reservoir = [], []
        for i in tqdm.tqdm(range(x.shape[0]), desc="Predicting (One Step)", leave=False):
            In = np.matmul([x[i,:]], self.W_in)
            Res = np.matmul(self.x_n, self.W_res)
            self.x_n = ((1.0 - self.leak_rate) * self.x_n + self.leak_rate * np.tanh(In + Res)).reshape(1, self.units)
            pred = np.matmul(self.x_n, self.W_out)
            ans.append(pred.reshape(-1).tolist())
            reservoir.append(self.x_n)
        self.reservoir_predict = np.array(reservoir)  
        self.predict_ans = np.array(ans)
        if return_reservoir: 
            return self.predict_ans, self.reservoir_predict    
        else: 
            return self.predict_ans

    def freerun(self, in_layer_data0, pred_range=100, return_reservoir=False):
        self.freerun_length = pred_range
        x = in_layer_data0
        ans, reservoir = [], []
        for _ in tqdm.tqdm(range(pred_range), desc="Predicting (Freerun)", leave=False):
            In = np.matmul([x], self.W_in)
            Res = np.matmul(self.x_n, self.W_res)
            self.x_n = ((1.0 - self.leak_rate) * self.x_n + self.leak_rate * np.tanh(In + Res)).reshape(1, self.units)
            pred = np.matmul(self.x_n, self.W_out)
            ans.append(pred.reshape(-1).tolist())
            reservoir.append(self.x_n)
            x = pred
        self.reservoir_freerun = np.array(reservoir)  
        self.freerun_ans = np.array(ans)
        if return_reservoir: 
            return self.freerun_ans, self.reservoir_freerun       
        else: 
            return self.freerun_ans
    
    def computing_lyapunov_exponent(self, dt):
        W_dict = {"W_in": self.W_in, "W_res": self.W_res, "W_out": self.W_out}
        lyapunov_exponents, lyapunov_dim = lyapunov_exponent(self.reservoir_freerun, W_dict, self.freerun_length, dt)
        self.lyapunov_exponents = lyapunov_exponents
        self.lyapunov_dim = lyapunov_dim
        return self.lyapunov_exponents, self.lyapunov_dim

class AutoEncoder():

    def __init__(self, data, method='PCA', 
                 params_pca={'n_principal_components':3},
                 params_mlp={'hidden_layers_shape':[20], 'latent_dim':1, 'latent_regularizer':None, 'random_state':0,
                             'learning_rate':1e-3, 'loss_function':'mse',
                             'epochs':300, 'batch_size':1024, 'verbose':0, 'callbacks':None}):
        self.method = method
        self.data = data

        if self.method=='PCA':
            self.params = params_pca
            self.ae_model = principal_component_analysis(self.data)

        if self.method=='MLP':
            self.params = params_mlp
            self.ae_model = MLPAutoencoder(
                                        input_layer_shape = (self.data.shape[1], 1), 
                                        hidden_layers_shape = self.params['hidden_layers_shape'], 
                                        latent_dim = self.params['latent_dim'],
                                        latent_regularizer = self.params['latent_regularizer'],
                                        random_state = self.params['random_state']
                                        )
            
    def fit(self, data):

        if self.method=='PCA':
            self.ae_model.fit(data)

        if self.method=='MLP':
            self.ae_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=self.params['learning_rate']), loss=self.params['loss_function'])
            self.train_history = self.ae_model.fit(
                                                x=tf.convert_to_tensor(data),                         
                                                y=tf.convert_to_tensor(data),
                                                epochs=self.params['epochs'],
                                                batch_size=self.params['batch_size'],
                                                verbose=self.params['verbose'],
                                                callbacks=self.params['callbacks'],
                                                )

    def encode(self, data, filtering=True, params={'window_length':51, 'polyorder':3}):

        if self.method=='PCA':
            self.latent_vector = self.ae_model.transform(data, self.params['n_principal_components'])
        
        if self.method=='MLP':
            self.latent_vector =  self.ae_model.encoder.predict(data)

        if filtering:
            self.latent_vector_filtered = savgol_filtering(self.latent_vector, window_length=params['window_length'], polyorder=params['polyorder'])
            return self.latent_vector_filtered
        else:
            return self.latent_vector
        
    def decode(self, data):

        if self.method=='PCA':
            return self.ae_model.transform_inverse(data)
        
        if self.method=='MLP':
            return self.ae_model.decoder.predict(data)

class SINDy():

    def __init__(self, feature_names=["x", "mu"]):
        self.feature_names = feature_names

    def setup_traindata(self, data, parameter_data, n_sample=None):
        if n_sample==None:
            n_sample_list = []
            for i in range(len(data)):
                n_sample_list.append(data[i].shape[0])
            n_sample = np.min(np.array(n_sample_list))

        train_X = [np.zeros((n_sample, 2)) for i in range(len(parameter_data))] 
        for i in range(len(parameter_data)):
            for j in range(n_sample):
                train_X[i][j] = [data[i][j], parameter_data[i]]
        self.train_X = train_X

    def fit(self, train_X, params={
                                'order':7, 
                                'threshold':0.01, 
                                'alpha':0.05, 
                                }):
        self.model = ps.SINDy(
                            feature_names=self.feature_names, 
                            optimizer=ps.STLSQ(threshold=params['threshold'], alpha=params['alpha']), 
                            feature_library=ps.PolynomialLibrary(degree=params['order']), 
                            discrete_time=True
                            )
        self.model.fit(x=train_X, multiple_trajectories=True)
        self.params = params

    def print_model(self):
        print('[ SINDy params ]')
        print('order     | '+str(self.params['order']))
        print('threshold | '+str(self.params['threshold']))
        print('alpha     | '+str(self.params['alpha']))
        print('------------------------------')
        ### equations
        print('[ SINDy model ]')
        self.model.print()

    def freerun(self, data0, parameter_start_stop_step=(0, 1, 0.0001), 
                n_idling_run=10, n_run=10, stop_run_difference=0.01):
        ### setup test data
        test_parameter_data = np.arange(parameter_start_stop_step[0], 
                                        parameter_start_stop_step[1]+parameter_start_stop_step[2], 
                                        parameter_start_stop_step[2])
        test_X = [np.zeros((1, 2)) for i in range(test_parameter_data.shape[0])] 
        for i in range(test_parameter_data.shape[0]):
                test_X[i] = [[data0, test_parameter_data[i]]]
        self.test_X = test_X
        ### freerun
        freerun_test_X = np.zeros((test_parameter_data.shape[0] * n_run, 2))
        x_idx = 0
        for i in tqdm.tqdm(range(test_parameter_data.shape[0]), desc="Predicting (Freerun)", leave=False):
            xss = self.model.simulate(test_X[i][0], n_idling_run)[-1]
            stop_condition = lambda x: np.abs(x[0] - xss[0]) < stop_run_difference
            x = self.model.simulate(xss, n_run, stop_condition=stop_condition)
            x_idx_new = x_idx + x.shape[0]
            freerun_test_X[x_idx:x_idx_new] = x
            x_idx = x_idx_new
        self.freerun_test_X = freerun_test_X[:x_idx]



