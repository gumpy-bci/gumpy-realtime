import os, sys

if os.name == "nt":
    # DIRTY workaround from stackoverflow
    # when using scipy, a keyboard interrupt will kill python
    # so nothing after catching the keyboard interrupt will
    # be executed

    import imp
    import ctypes
    import _thread
    import win32api

    #basepath = imp.find_module('numpy')[1]
    #ctypes.CDLL(os.path.join(basepath, 'core', 'libmmd.dll'))
    #ctypes.CDLL(os.path.join(basepath, 'core', 'libifcoremd.dll'))
    ctypes.CDLL(os.path.join('C:\\Users\\mirja\\Anaconda3\\pkgs\\icc_rt-2017.0.4-h97af966_0\\Library\\bin', 'libmmd.dll'))
    ctypes.CDLL(os.path.join('C:\\Users\\mirja\\Anaconda3\\pkgs\\icc_rt-2017.0.4-h97af966_0\\Library\\bin', 'libifcoremd.dll'))
    

    def handler(dwCtrlType, hook_sigint=_thread.interrupt_main):
        if dwCtrlType == 0:
            hook_sigint()
            return 1
        return 0

    win32api.SetConsoleCtrlHandler(handler, 1)


import threading           # NOQA
import scipy.io as sio     # NOQA
import pylsl               # NOQA
from utils import time_str # NOQA
import time                # NOQA
import numpy as np

from gumpy.data.nst_eeg_live import NST_EEG_LIVE


class NoRecordingDataError(Exception):
    def __init__(self):
        self.value = "Received no data while recording"

    def __str__(self):
        return repr(self.value)


def record(channel_data=[], time_stamps=[]):
    streams = pylsl.resolve_stream('type', 'EEG')
    inlet   = pylsl.stream_inlet(streams[0])

    while True:
        try:
            sample, time_stamp = inlet.pull_sample()
            time_stamp += inlet.time_correction()

            time_stamps.append(time_stamp)
            channel_data.append(sample)

            # first col of one row of the record_data matrix is time_stamp,
            # the following cols are the sampled channels
        except KeyboardInterrupt:
            complete_samples = min(len(time_stamps), len(channel_data))
            sio.savemat("recording_" + time_str() + ".mat", {
                "time_stamps"  : time_stamps[:complete_samples],
                "channel_data" : channel_data[:complete_samples]
            })
            break


class RecordData():
    def __init__(self, Fs, age, gender="male", with_feedback=False,
                 record_func=record):
        # timepoints when the subject starts imagination
        self.trial = []

        self.X = []

        self.trial_time_stamps = []
        self.time_stamps       = []

        # 0 negative_feedback
        # 1 positive feedback
        self.feedbacks = []

        # containts the lables of the trials:
        # 1: left
        # 2: right
        # 3: both hands
        self.Y = []

        # sampling frequncy
        self.Fs = Fs

        self.trial_offset = 4

        self.gender   = gender
        self.age      = age
        self.add_info = "with feedback" if with_feedback else "with no feedback"

        recording_thread = threading.Thread(
            target=record_func,
            args=(self.X, self.time_stamps),
        )
        recording_thread.daemon = True
        self.recording_thread   = recording_thread

    def __iter__(self):
        yield 'trial'            , self.trial
        yield 'age'              , self.age
        yield 'X'                , self.X
        yield 'time_stamps'      , self.time_stamps
        yield 'trial_time_stamps', self.trial_time_stamps
        yield 'Y'                , self.Y
        yield 'Fs'               , self.Fs
        yield 'gender'           , self.gender
        yield 'add_info'         , self.add_info
        yield 'feedbacks'        , self.feedbacks

    def add_trial(self, label):
        self.trial_time_stamps.append(pylsl.local_clock())
        self.Y.append(label)
        self.trial.append(len(self.X)-1)

    def add_feedback(self, feedback):
        self.feedbacks.append(feedback)

    def start_recording(self):
        self.recording_thread.start()
        time.sleep(2)
        if len(self.X) == 0:
            raise NoRecordingDataError()

    ### this function is not required anymore, because self.trial is updated in add_trial()
    ### kept for historical reasons
    def set_trial_start_indexes(self):
        if len(self.trial) > 0:
            self.trial = []

        i = 0
        for trial_time_stamp in self.trial_time_stamps:
            for j in range(i, len(self.time_stamps)):
                time_stamp = self.time_stamps[j]
                if trial_time_stamp <= time_stamp:
                    self.trial.append(j - 1)
                    i = j
                    break

    def stop_recording_and_dump(self, file_name="session_" + time_str() + ".mat"):
        #self.set_trial_start_indexes() #solved by collecting index in add_trial()
        sio.savemat(file_name, dict(self))

        return file_name


    ### used for live processing step 2: dump all data generated so far, another filename than stop_recording_and_dump
    def stop_recording_and_dump_live(self, file_name="session_live_" + time_str() + ".mat"):
        #self.set_trial_start_indexes() #solved by collecting index in add_trial()
        sio.savemat(file_name, dict(self))

        return file_name

    def get_last_trial(self, filename_live):
        # generate a NST_EEG_LIVE object and save data of last trial into it
        last_label = self.Y[-1:]
        ### subtract one trial offset, because add trial is allways called when the moto imagery starts and not in the beginning of each trial 
        last_trial = self.trial[-1:][0] - self.Fs*self.trial_offset
        X = np.array(self.X[slice(last_trial[0],None,None)])
        #print(X.shape)
        last_trial = 0 #hand over 0 as index to dataset object, because the new index in the slice of X that will be handed over is 0
        cwd = os.getcwd()
        self.nst_eeg_live = NST_EEG_LIVE(cwd, filename_live)
        self.nst_eeg_live.load_from_mat(last_label, last_trial, X, self.Fs)
        return self.nst_eeg_live



if __name__ == '__main__':
    pass #record()
