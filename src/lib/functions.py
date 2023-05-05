import os
import random
import numpy as np
import pandas as pd
import cv2
from scipy.signal import buttord, butter, filtfilt, savgol_filter, find_peaks, stft
from scipy.ndimage import convolve
from scipy.stats import zscore
from scipy.linalg import qr
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import normalized_mutual_info_score
import tensorflow as tf
import tqdm


########## UTILS ##########
def set_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


########## LOADING DATA ##########
def load_csv_data(csv_path, data_name_list, sample_span, quiet=False):
    if quiet==False:
        print('Loading csv data')
        print('file path | '+csv_path)
        print('data list | '+", ".join(data_name_list))
    elif quiet==True:
        pass
    data_df = pd.read_csv(csv_path)
    data_list = []
    for i in range(len(data_name_list)):
        data_list.append(data_df[[data_name_list[i]]].values[sample_span[0]:sample_span[1], 0])
    index = np.arange(sample_span[0], sample_span[1])
    return data_list, index

def load_video_data(video_path, time_span, shooting_time_interval, to_GRAY=True,):
    print('Loading video data')
    print('file path | '+video_path)
    ### VideoCapture (get object)
    cap = cv2.VideoCapture(video_path)
    ### get video property
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    ### generate video time data
    video_t = np.arange(0, frame_count+1)*shooting_time_interval
    ### decide start load point & end load point
    start_frame = np.argmin(abs(video_t-time_span[0]))
    stop_frame = np.argmin(abs(video_t-time_span[1]))
    t_data = video_t[start_frame:stop_frame] 
    ### load video
    frames = []
    for i in tqdm.tqdm(range(stop_frame), desc="Loading", leave=False):
        ret, frame = cap.read()
        if ret: # read successed
            if i > int(start_frame-1):
                ### RGB 3ch --> GRAY 1ch
                if to_GRAY:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                ### save frame
                frames.append(frame)
        else : # read failed
            break
    cap.release()
    ### convert datatype: list init8 --> numpy float64
    video_data = np.array(frames).astype(np.float64)
    return video_data, t_data


########## PREPROCESSING (GENERAL) ##########
def standardize(data):
    standardscaler = StandardScaler()
    if data.ndim==1:
        data = data.reshape(-1, 1)
        standardscaler.fit(data)
        return (standardscaler.transform(data)).squeeze()
    else:
        standardscaler.fit(data)
        return standardscaler.transform(data)
    
def mean0(data):
    return data - np.mean(data, axis=0)

def savgol_filtering(data, window_length, polyorder):
    return savgol_filter(data, window_length=window_length, polyorder=polyorder, axis=0)


########## PREPROCESSING (MIC) ##########
def bandpass_filter(data, t_data, passband_edge_freq, stopband_edge_freq, passband_edge_max_loss, stopband_edge_min_loss,):
    dt = t_data[1] - t_data[0]
    sampling_rate = 1. / dt
    niquist_freq = sampling_rate / 2.
    passband_edge_freq_normalize = passband_edge_freq / niquist_freq
    stopband_edge_freq_normalize = stopband_edge_freq / niquist_freq
    butterworth_order, butterworth_natural_freq = buttord(
                                                        wp=passband_edge_freq_normalize, 
                                                        ws=stopband_edge_freq_normalize,
                                                        gpass=passband_edge_max_loss,
                                                        gstop=stopband_edge_min_loss
                                                        )
    numerator_filterfunc, denominator_filterfunc = butter(
                                                        N=butterworth_order,
                                                        Wn=butterworth_natural_freq,
                                                        btype='band'
                                                        )
    data_filtered = filtfilt(
                            b=numerator_filterfunc,
                            a=denominator_filterfunc,
                            x=data
                            )
    return data_filtered
    

########## PREPROCESSING (VIDEO) ##########
def video_filtering(video_data, kernel_length, kernel_size):
    k = np.ones((kernel_length, kernel_size, kernel_size))/float(kernel_size*kernel_size*kernel_length)
    video_data_filtered = convolve(video_data, k)
    return video_data_filtered

def line_scanning(video_data, position, width):
    kymograph_data = video_data[:, position[1], (position[0]-int(width/2)):(position[0]+int(width/2))]
    return kymograph_data

def gamma_correction(data, gamma):
    return 255.*(data/225.)**(1/gamma)


########## CONVERTING (BIFURCATION DIAGRAM) ##########
def fing_peaks_index(data, prominence):
    peaks_index, _ = find_peaks(data, prominence=prominence)
    peaks = data[peaks_index]
    return peaks, peaks_index


########## TIME-SERIES ANALYSIS ##########
def fft(data, t_data):
    dt = t_data[1] - t_data[0]
    f = np.fft.fft(data)
    freq = np.fft.fftfreq(data.shape[0], dt)
    amp = np.abs(f/(data.shape[0]/2))
    plt_lim = int(data.shape[0]/2)
    return freq[1:plt_lim], amp[1:plt_lim]

def short_time_fourier_transform(data, dt, nperseg):
    return stft(x=data, fs=1/dt, nperseg=nperseg)

def mutual_info_score(unlagged, lagged):
    return normalized_mutual_info_score(unlagged, lagged)


########## ECHO-STATE NETWORK ##########
def W_res_create(shape, SR=0.99, density=0.1, seed=0):
    set_seed(seed)
    W1 = np.random.uniform(size=shape, low=-1.0, high=1.0)
    W2 = np.random.uniform(size=shape, low=0.0, high=1.0)
    W2 = (W2 > (1.0 - density)).astype(np.float)
    W = W1 * W2
    value, _ = np.linalg.eigh(W)
    sr = max(np.abs(value))
    return W * SR / sr

def lyapunov_exponent(reservoir, W_dict, length, dt):
        ones = np.ones((W_dict["W_res"].shape[0], W_dict["W_res"].shape[1]))
        lyapunov_exponent = np.zeros((reservoir.shape[2]))
        Q = np.eye((W_dict["W_res"].shape[0]))
        W = W_dict["W_res"] + np.matmul(W_dict["W_out"], W_dict["W_in"])
        for i in tqdm.tqdm(range(length), desc="Computing Lyapunov Exponent", leave=False):
            S = reservoir[i,:,:].reshape(-1,1) ** 2
            J = (ones - S) * W.T
            Q, R = qr(np.matmul(J, Q))
            lmd = np.log(np.abs(np.diag(R)))
            lyapunov_exponent += lmd
        lyapunov_exponent = lyapunov_exponent / (length*dt)
        l = 0
        for i in range(length):
            l += lyapunov_exponent[i]
            if l < 0: break
        dim = i + (sum(lyapunov_exponent[:i]) / abs(lyapunov_exponent[i]))
        if dim <= 1.0: dim = 1.0
        return lyapunov_exponent, dim


########## DIMENSIONALITY REDUCTION ##########
class principal_component_analysis():

    def __init__(self, data):
        self.data = data
        self.pca = PCA()

    def fit(self, data):
        self.pca.fit(data)
        self.variance_ratios = list(np.cumsum(self.pca.explained_variance_ratio_))

    def transform(self, data, n_pc):
        self.n_principal_components = n_pc
        self.principal_components = self.pca.transform(data)
        return self.principal_components[:, :self.n_principal_components]
    
    def transform_inverse(self, data):
        return self.pca.inverse_transform(data)
    
class MLPAutoencoder(tf.keras.Model):
    def __init__(
                self, 
                input_layer_shape, # (time_window, n_features)
                hidden_layers_shape, # networl_shape ([node_num, node_num, ...])
                latent_dim, # n_latent
                latent_regularizer=None,
                rnn_opts=dict(), 
                activation_func=tf.keras.layers.ELU(alpha=1.0),
                random_state=None,
                **kwargs
                ):
        ### tf.keras.Model
        super().__init__()
        ### MLPAutoencoder
        self.input_layer_shape = input_layer_shape
        self.hidden_layers_shape = hidden_layers_shape
        self.latent_dim = latent_dim
        self.latent_regularizer = latent_regularizer
        ### Initialize state
        tf.random.set_seed(random_state)
        np.random.seed(random_state)
        random.seed(random_state)
        os.environ["PYTHONHASHSEED"] = str(random_state)
        ### Encoder
        self.encoder = tf.keras.Sequential()
        self.encoder.add(tf.keras.layers.InputLayer(input_shape=input_layer_shape))
        self.encoder.add(tf.keras.layers.Flatten())
        self.encoder.add(tf.keras.layers.GaussianNoise(0.5, input_shape=(input_layer_shape[0],))) # smooths the output
        for hidden_size in hidden_layers_shape:
            self.encoder.add(tf.keras.layers.Dense(hidden_size, **rnn_opts))
            self.encoder.add(tf.keras.layers.BatchNormalization())
            self.encoder.add(tf.keras.layers.Activation(activation_func))
        self.encoder.add(tf.keras.layers.Dense(latent_dim, input_shape=(input_layer_shape[0],), **rnn_opts))
        self.encoder.add(tf.keras.layers.BatchNormalization())
        self.encoder.add(
            tf.keras.layers.Reshape(
                (latent_dim,),  
                activity_regularizer=latent_regularizer
            )
        )
        ## Decoder
        self.decoder = tf.keras.Sequential()
        self.decoder.add(tf.keras.layers.Flatten())
        self.decoder.add(tf.keras.layers.GaussianNoise(0.5, input_shape=(latent_dim,)))
        for hidden_size in hidden_layers_shape[::-1]:
            self.decoder.add(tf.keras.layers.Dense(hidden_size,  **rnn_opts))
            self.decoder.add(tf.keras.layers.BatchNormalization())
            self.decoder.add(tf.keras.layers.Activation(activation_func))
        self.decoder.add(tf.keras.layers.Dense(input_layer_shape[0]*input_layer_shape[1], **rnn_opts))
        self.decoder.add(tf.keras.layers.BatchNormalization())
        #self.decoder.add(tf.keras.layers.Activation(activation_func))
        self.decoder.add(tf.keras.layers.Reshape((input_layer_shape)))
        
    def call(self, inputs, training=False):
        outputs = self.decoder(self.encoder(inputs))
        return outputs
    