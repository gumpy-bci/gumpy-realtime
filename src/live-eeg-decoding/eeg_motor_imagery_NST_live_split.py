# Mirjam Hemberger
# Engineering Practice
# 12.03.2018
# class used for liveEEG based on EEG-motor-imagery-NST.ipynb


import sys, os, os.path
sys.path.append('C:\\Users\\mirja\\Anaconda3\\Lib\\site-packages\\gumpy-master\\gumpy')

import numpy as np
import gumpy
from nst_eeg_live import NST_EEG_LIVE
from nst_eeg_test import NST_EEG_TEST
from nst_eeg_test_split import NST_EEG_TEST_SPLIT



class liveEEG_split():
    def __init__(self, cwd, filename_notlive, flag=0):
        # cwd and filename_notlive specify the location of one .mat file where the recorded notlive data has been stored
        # print(filename_notlive, '\n')
        # load notlive data from path that has been specified
        self.data_notlive = NST_EEG_TEST_SPLIT(cwd, filename_notlive)
        self.data_notlive.load()
        self.data_notlive.print_stats()  

        if 1:
            self.data_notlive.raw_data[:, 0] -=  self.data_notlive.raw_data[:, 2]
            self.data_notlive.raw_data[:, 1] -=  self.data_notlive.raw_data[:, 2]
            self.data_notlive.raw_data[:, 0].shape

        self.labels_notlive=self.data_notlive.labels

        # butter-bandpass filtered version of notlive data
        self.data_notlive_filtbp = gumpy.signal.butter_bandpass(self.data_notlive, lo=2, hi=60)

        # frequency to be removed from the signal
        self.notch_f0 = 50
        # quality factor
        self.notch_Q = 50.0
        # get the cutoff frequency
        # self.notch_w0 = self.notch_f0/(self.data_notlive.sampling_freq/2) 
        # apply the notch filter
        self.data_notlive_filtno = gumpy.signal.notch(self.data_notlive_filtbp, cutoff=self.notch_f0, Q=self.notch_Q, fs=self.data_notlive.sampling_freq)

        # normalize the data first
        #self.data_notlive_filtno = gumpy.signal.normalize(self.data_notlive_filtno, 'mean_std')

        self.alpha_bands = np.array(self.alpha_subBP_features(self.data_notlive_filtno))
        self.beta_bands = np.array(self.beta_subBP_features(self.data_notlive_filtno))

        # Feature extraction using sub-bands
        # Method 1: logarithmic sub-band power

        # use 2 seconds of data for building the model (for only 1 second delete self.features3 und self.features4 and adapt self.features_notlive)
        self.w1 = [0,128]
        self.w2 = [128,256]
        self.w3 = [256,384]
        self.w4 = [384,512]

        self.features1 = self.log_subBP_feature_extraction(
            self.alpha_bands, self.beta_bands, 
            self.data_notlive.trials, self.data_notlive.sampling_freq,
            self.w1)

        self.features2 = self.log_subBP_feature_extraction(
            self.alpha_bands, self.beta_bands, 
            self.data_notlive.trials, self.data_notlive.sampling_freq,
            self.w2)       

        self.features3 = self.log_subBP_feature_extraction(
            self.alpha_bands, self.beta_bands, 
            self.data_notlive.trials, self.data_notlive.sampling_freq,
            self.w3) 

        self.features4 = self.log_subBP_feature_extraction(
            self.alpha_bands, self.beta_bands, 
            self.data_notlive.trials, self.data_notlive.sampling_freq,
            self.w4)                                   

        # concatenate the features and normalize the data
        self.features_notlive = np.concatenate((self.features1.T, self.features2.T, self.features3.T, self.features4.T)).T
        self.features_notlive -= np.mean(self.features_notlive)
        self.features_notlive = gumpy.signal.normalize(self.features_notlive, 'min_max')

        # Method 2: DWT is not used!
        # We'll work with the data that was postprocessed using a butter bandpass
        if False:          
            self.w = [0, 512]
            # extract the features
            self.trials = self.data_notlive.trials
            self.fs = self.data_notlive.sampling_freq
            self.features1= np.array(self.dwt_features(self.data_notlive_filtno, self.trials, 5, self.fs, self.w, 3, "db4"))
            self.features2= np.array(self.dwt_features(self.data_notlive_filtno, self.trials, 5, self.fs, self.w, 4, "db4"))
            # concat the features and normalize
            self.features_notlive = np.concatenate((self.features1.T, self.features2.T)).T
            self.features_notlive -= np.mean(self.features_notlive)
            self.features_notlive = gumpy.signal.normalize(self.features_notlive, 'min_max')

        self.test_size = 0.2
        self.split_features = np.array(gumpy.split.normal(self.features_notlive, self.labels_notlive, self.test_size))
        # the functions return a list with the data according to the following example
        self.features_notlive_train = self.split_features[0]
        self.features_notlive_test = self.split_features[1]
        self.labels_notlive_train = self.split_features[2]
        self.labels_notlive_test = self.split_features[3] 

        self.pos_fit = False
        #print(self.features_notlive)
        #print(self.labels_notlive)



    def fit(self):
        # builds a model with data from all trials_notlive
        # Sequential Feature Selection Algorithm
        out_realtime = gumpy.features.sequential_feature_selector(self.features_notlive_train, self.labels_notlive_train, 'QuadraticLDA', (6, 20), 10, 'SFFS')
        print('\n\nAverage score:', out_realtime[1]*100)
        self.sfs_object = out_realtime[3]
        self.estimator_object = self.sfs_object.est_
        ### fit the estimator object with the selected features and test
        self.estimator_object.fit(self.sfs_object.transform(self.features_notlive_train), self.labels_notlive_train)
        self.labels_pred_notlive = self.estimator_object.predict(self.sfs_object.transform(self.features_notlive_train))
        self.acc_notlive = 100 * np.sum(abs(self.labels_pred_notlive-self.labels_notlive_train)<1) \
                / np.shape(self.labels_pred_notlive)
        print('\nAccuracy of notlive fit:', self.acc_notlive[0], '\n')
        self.pos_fit = True



    def classify_live(self, data_live=0):
        # classifies every trial live
 
        ### predict label of live trial and check whether it is correct
        # ==1 because self.labels_live are 1,2,3 (same as in matlab files) but classifier predicts 0,1,2 for left, right, both
        if self.pos_fit:
            labels_pred_live = self.estimator_object.predict(self.sfs_object.transform(self.features_notlive_test))
            pred_true = (self.labels_notlive_test - labels_pred_live) == 1

        else:
            print("No fit was performed yet. Please train the model first.\n")
            sys.exit()

        pred_valid = True

        return labels_pred_live, pred_true, pred_valid




    # Alpha and Beta sub-bands
    def alpha_subBP_features(self, data):
        # filter data in sub-bands by specification of low- and high-cut frequencies
        alpha1 = gumpy.signal.butter_bandpass(data, 8.5, 11.5, order=6)
        alpha2 = gumpy.signal.butter_bandpass(data, 9.0, 12.5, order=6)
        alpha3 = gumpy.signal.butter_bandpass(data, 9.5, 11.5, order=6)
        alpha4 = gumpy.signal.butter_bandpass(data, 8.0, 10.5, order=6)
        # return a list of sub-bands
        return [alpha1, alpha2, alpha3, alpha4]

    def beta_subBP_features(self, data):
        beta1 = gumpy.signal.butter_bandpass(data, 14.0, 30.0, order=6)
        beta2 = gumpy.signal.butter_bandpass(data, 16.0, 17.0, order=6)
        beta3 = gumpy.signal.butter_bandpass(data, 17.0, 18.0, order=6)
        beta4 = gumpy.signal.butter_bandpass(data, 18.0, 19.0, order=6)
        return [beta1, beta2, beta3, beta4]


    # Feature extraction using sub-bands (The following examples show how the sub-bands can be used to extract features.)
    # Method 1: logarithmic sub-band power
    def powermean(self, data, trial, fs, w):
        return  np.power(data[trial+fs*5+w[0]: trial+fs*5+w[1],0],2).mean(), \
                np.power(data[trial+fs*5+w[0]: trial+fs*5+w[1],1],2).mean(), \
                np.power(data[trial+fs*5+w[0]: trial+fs*5+w[1],2],2).mean()

    def log_subBP_feature_extraction(self, alpha, beta, trials, fs, w):
        # number of features combined for all trials
        n_features = 15
        # initialize the feature matrix
        X = np.zeros((len(trials), n_features))
        
        # Extract features
        for t, trial in enumerate(trials):
            power_c31, power_c41, power_cz1 = self.powermean(alpha[0], trial, fs, w)
            power_c32, power_c42, power_cz2 = self.powermean(alpha[1], trial, fs, w)
            power_c33, power_c43, power_cz3 = self.powermean(alpha[2], trial, fs, w)
            power_c34, power_c44, power_cz4 = self.powermean(alpha[3], trial, fs, w)
            power_c31_b, power_c41_b, power_cz1_b = self.powermean(beta[0], trial, fs, w)
            
            X[t, :] = np.array(
                [np.log(power_c31), np.log(power_c41), np.log(power_cz1),
                np.log(power_c32), np.log(power_c42), np.log(power_cz2),
                np.log(power_c33), np.log(power_c43), np.log(power_cz3), 
                np.log(power_c34), np.log(power_c44), np.log(power_cz4),
                np.log(power_c31_b), np.log(power_c41_b), np.log(power_cz1_b)])
        return X

    # Method 2: DWT
    def dwt_features(self, data, trials, level, sampling_freq, w, n, wavelet): 
        import pywt
        
        # number of features per trial
        n_features = 9 
        # allocate memory to store the features
        X = np.zeros((len(trials), n_features))

        # Extract Features
        for t, trial in enumerate(trials):
            signals = data[trial + sampling_freq*5 + (w[0]) : trial + sampling_freq*5 + (w[1])]
            coeffs_c3 = pywt.wavedec(data = signals[:,0], wavelet=wavelet, level=level)
            coeffs_c4 = pywt.wavedec(data = signals[:,1], wavelet=wavelet, level=level)
            coeffs_cz = pywt.wavedec(data = signals[:,2], wavelet=wavelet, level=level)

            X[t, :] = np.array([
                np.std(coeffs_c3[n]), np.mean(coeffs_c3[n]**2),  
                np.std(coeffs_c4[n]), np.mean(coeffs_c4[n]**2),
                np.std(coeffs_cz[n]), np.mean(coeffs_cz[n]**2), 
                np.mean(coeffs_c3[n]),
                np.mean(coeffs_c4[n]), 
                np.mean(coeffs_cz[n])])
            
        return X

  
