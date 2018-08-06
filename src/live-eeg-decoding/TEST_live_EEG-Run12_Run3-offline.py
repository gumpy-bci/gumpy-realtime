# Mirjam Hemberger 
# 14.03.2018
# this script is used to test the functions in EEG-motor-imaginery-NST_live.py and nst_eeg_live without having to call them from record_data or run_session
# Training of the model with two Run.mat files, testing with one Run.mat file


#import matplotlib.pyplot as plt 

import sys, os, os.path
sys.path.append('C:\\Users\\mirja\\Anaconda3\\Lib\\site-packages\\gumpy-master\\gumpy')

import numpy as np
import gumpy

from eeg_motor_imagery_NST_live import liveEEG
from gumpy.data.nst_eeg_live import NST_EEG_LIVE

if __name__ == '__main__':
    save_stdout = sys.stdout
    fh = open('Results_LiveClassfication.txt', 'w')
    sys.stdout = fh
    subjects = {'s_all'}
    print('\nTraining with Run1.mat, Run2.mat, Run4.mat, Run5.mat and Run6.mat, testing with Run3.mat')
    print('Classifier: Quadratic LDA\n')
    
    for subject in subjects:
        print('\n\n\nData identification:', subject, '\n')
        data_base_dir = 'C:\\Users\\mirja\\Documents\\TUM\\IP\\NST'
        base_dir = os.path.join(data_base_dir, subject)
        
        #isn't used for training with Run1.mat and Run2.mat
        file_name = 'Run1.mat'
        
        file_name2 = 'Run3.mat'

        #used for testing
        flag = 1

        myclass = liveEEG(base_dir,file_name, flag)
        myclass.fit()

        myclass2 = liveEEG(base_dir, file_name2)

        count_pred_true = 0
        count_pred_false = 0

        for i in range(0,myclass2.data_notlive.trials.shape[0]):
            fs = myclass2.data_notlive.sampling_freq
            label = myclass2.data_notlive.labels[i]
            trial = myclass2.data_notlive.trials[i]
            if i < (myclass2.data_notlive.trials.shape[0] - 1):
                next_trial = myclass2.data_notlive.trials[i+1]
                X = myclass2.data_notlive.raw_data[trial:next_trial]
            else:
                X = myclass2.data_notlive.raw_data[trial:]

            trial = 0
            nst_eeg_live = NST_EEG_LIVE(base_dir, file_name2)
            nst_eeg_live.load_from_mat(label,trial,X,fs)

            current_classifier, pred_true, pred_valid = myclass.classify_live(nst_eeg_live)

            if not pred_valid:
                continue

            #print('Classification result: ',current_classifier[0],'\n')
            if pred_true:
                #print('This is true!\n')
                count_pred_true = count_pred_true + 1
            else:
                count_pred_false += 1
                #print('This is false!\n')

        print('Count of true predictions:', count_pred_true)
        print('Count of false predictions:', count_pred_false)
        print('Percentage of true predictions:', 100*count_pred_true/(count_pred_false+count_pred_true), '\n\n')

    sys.stdout = save_stdout
    fh.close()
