# Mirjam Hemberger 
# 14.03.2018
# this script is used to test the functions in EEG-motor-imaginery-NST_live.py and nst_eeg_live without having to call them from record_data or run_session
# Training of the model and testing with a certain ration, e.g. 80% data for training, 20% for testing


#import matplotlib.pyplot as plt 

import sys, os, os.path
sys.path.append('C:\\Users\\mirja\\Anaconda3\\Lib\\site-packages\\gumpy-master\\gumpy')

import numpy as np
import gumpy

from eeg_motor_imagery_NST_live_split import liveEEG_split
from gumpy.data.nst_eeg_live import NST_EEG_LIVE

if __name__ == '__main__':
    save_stdout = sys.stdout
    fh = open('Results_LiveClassfication.txt', 'w')
    sys.stdout = fh
    subjects = {'s1','s4', 's20'}
    print('\nTraining-testing with 50-50 percent')
    print('Classifier: Quadratic LDA\n')
    
    for subject in subjects:
        print('\n\n\nData identification:', subject, '\n')
        data_base_dir = 'C:\\Users\\mirja\\Documents\\TUM\\IP\\NST'
        base_dir = os.path.join(data_base_dir, subject)
        
        #isn't used for training and testing with data split
        file_name = 'Run1.mat'
        file_name2 = 'Run3.mat'

        myclass = liveEEG_split(base_dir,file_name)

        myclass.fit()

        count_pred_true = 0
        count_pred_false = 0

        current_classifier, pred_true, pred_valid = myclass.classify_live()

        #print('Classification Result:', current_classifier)
        #print('pred_true words:', pred_true)

        pred_true = np.int_(pred_true)
        print('pred_true numbers:', pred_true)
        print('true numbers:', pred_valid)
        count_pred_true = sum(pred_true)
        count_pred_false = len(pred_true) - count_pred_true

        print('Count of true predictions:', count_pred_true)
        print('Count of false predictions:', count_pred_false)
        print('Percentage of true predictions:', 100*count_pred_true/(count_pred_false+count_pred_true), '\n\n')

    sys.stdout = save_stdout
    fh.close()
