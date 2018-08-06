# Mirjam Hemberger
# Engineering Practice
# 13.03.2018
# modification of run_session.py to allow live processing and classification of EEG
# phase 1: collect data (certain amount of trials, defined by trials_notlive (-l))
# phase 2: analyse and classify collected data and build model
# phase 3: uses model to classify every trial live

import pygame
import re
import sys
import time
import random
import argparse
import screeninfo
import os
from record_data_liveEEG import RecordData
from eeg_motor_imagery_NST_live import liveEEG


on_windows = os.name == 'nt'

if on_windows:
    import winsound

class MyApp_liveEEG():
    def __init__(self, trial_count, trials_notlive, Fs=256, age=25, gender='male', with_feedback=False):
        self.trial_count = trial_count
        self.trials_notlive = trials_notlive
        
        self.screen_width, self.screen_height = self.get_screen_width_and_height()

        self.root_dir  = os.path.join(os.path.dirname(__file__), "..")
        self.image_dir = os.path.join(self.root_dir, "images")
        self.sound_dir = os.path.join(self.root_dir, "sounds")

        self.black   = (0,   0, 0)
        self.green   = (0, 255, 0)
        self.radius  = 100
        self.mid_pos = (self.screen_width // 2, self.screen_height // 2)

        pygame.init()
        pygame.mixer.init()
        pygame.mixer.music.load(os.path.join(self.sound_dir, "beep.mp3"))

        self.screen = pygame.display.set_mode((self.screen_width, self.screen_height), pygame.FULLSCREEN)
        self.screen.fill(self.black)

        self.red_arrow       = pygame.image.load(os.path.join(self.image_dir, "red_arrow.png"))
        self.red_arrow_left  = pygame.transform.rotate(self.red_arrow, 270)
        self.red_arrow_right = pygame.transform.rotate(self.red_arrow, 90)

        self.red_arrow_width, self.red_arrow_height = self.red_arrow_left.get_size()
        self.red_arrow_right_pos = (self.screen_width - self.red_arrow_width, (self.screen_height - self.red_arrow_height) // 2)
        self.red_arrow_left_pos  = (0, (self.screen_height - self.red_arrow_height) // 2)

        self.happy_smiley = pygame.image.load(os.path.join(self.image_dir, "happy_smiley.png"))
        self.sad_smiley   = pygame.image.load(os.path.join(self.image_dir, "sad_smiley.png"))

        self.smiley_width, self.smiley_height = self.happy_smiley.get_size()
        self.smiley_mid_pos = ((self.screen_width - self.smiley_width) // 2, (self.screen_height - self.smiley_height) // 2)


        self.filename_notlive = ""
        self.cwd = os.getcwd()
        self.current_classifier = []

        ### check inputs, whether the size is acceptable
        if self.trials_notlive > self.trial_count:
            raise ValueError("'trials_notlive' cannot be larger than 'trials'")
        
        if self.trial_count % 3:
            raise ValueError("'trials' must be devisable by 3")

        if self.trials_notlive % 3:
            raise ValueError("'trials_notlive' must be devisable by 3")

        ### Generate position lists for both live and notlive trials
        self.trial_count_for_each_cue_pos = self.trial_count // 3
        self.trial_count_for_each_cue_pos_notlive = self.trials_notlive // 3

        self.cue_pos_choices_notlive = {
            "left"  : self.trial_count_for_each_cue_pos_notlive,
            "right" : self.trial_count_for_each_cue_pos_notlive,
            "both"  : self.trial_count_for_each_cue_pos_notlive
        }

        self.cue_pos_choices_live = {
            "left"  : (self.trial_count_for_each_cue_pos-self.trial_count_for_each_cue_pos_notlive),
            "right" : (self.trial_count_for_each_cue_pos-self.trial_count_for_each_cue_pos_notlive),
            "both"  : (self.trial_count_for_each_cue_pos-self.trial_count_for_each_cue_pos_notlive)
        }

        #print(cue_pos_choices)
        print("\nNot live trials:")
        print(self.cue_pos_choices_notlive)
        print("\nLive trials:")
        print(self.cue_pos_choices_live, '\n')

        ### Initialise states
        if self.trial_count_for_each_cue_pos > 0:
            self.state = 1
        else:
            self.state = 0

        ### initialise the RecordData thread
        self.record_data_liveEEG = RecordData(Fs, age, gender, with_feedback)
        self.record_data_liveEEG.start_recording()

    def run_session(self):  
        print("Not live trials:")
        print(self.cue_pos_choices_notlive)
        for trial in range(self.trial_count+1):
            self.run_trial()

        pygame.quit()
        sys.exit()

    def get_screen_width_and_height(self):
        monitor_info = screeninfo.get_monitors()[0]
        if not monitor_info:
            sys.exit("couldn't find monitor")
        m = re.match("monitor\((\d+)x(\d+)\+\d+\+\d+\)", str(monitor_info))

        self.screen_width, self.screen_height = int(m.group(1)), int(m.group(2))
        return self.screen_width, self.screen_height

    def play_beep(self):
        pygame.mixer.music.play()

    # this decorator is defined to protect the classification function to run only once
    # otherwise the sequential feature selector causes problems, because it uses
    # the parallel processing toolbox
    def run_once(f):
        def wrapper(*args, **kwargs):
            if not wrapper.has_run:
                wrapper.has_run = True
                return f(*args, **kwargs)
        wrapper.has_run = False
        return wrapper

    @run_once
    def classify_notlive(self):
        ### does the classification (called in state 2)
        if __name__ == '__main__':
            print('\nClassification is starting. This might take a while.\n')
            #self.cue_pos_choices_notlive.append('end')
            self.cue_pos_choices_notlive = ['end']
            self.filename_notlive = self.record_data_liveEEG.stop_recording_and_dump() 
            print(self.cwd, ' ', self.filename_notlive, '\n')
            self.liveEEG = liveEEG(self.cwd, self.filename_notlive)
            self.liveEEG.fit()
            print('Classification completed. Back in run_session.\n')

    def show_motor_imagery(self, cue_pos_choices, with_feedback=False):
        ### experimental paradigm, user interface which is shown to the participant
        self.screen.fill(self.black)
        pygame.display.update()
        time.sleep(3)

        pygame.draw.circle(self.screen, self.green, self.mid_pos, self.radius)
        pygame.display.update()
        time.sleep(1)
        # ensure that each cue pos will be equally chosen
        cue_pos = random.choice(list(cue_pos_choices.keys()))
        cue_pos_choices[cue_pos] -= 1
        if cue_pos_choices[cue_pos] == 0:
            del cue_pos_choices[cue_pos]

        if cue_pos == "left":
            self.screen.blit(self.red_arrow_left, self.red_arrow_left_pos)
            self.record_data_liveEEG.add_trial(1)
        elif cue_pos == "right":
            self.screen.blit(self.red_arrow_right, self.red_arrow_right_pos)
            self.record_data_liveEEG.add_trial(2)
        elif cue_pos == "both":
            self.screen.blit(self.red_arrow_right, self.red_arrow_right_pos)
            self.screen.blit(self.red_arrow_left, self.red_arrow_left_pos)
            self.record_data_liveEEG.add_trial(3)
        pygame.display.update()
        time.sleep(0.5)

        if on_windows:
            winsound.Beep(2500, 500)
            time.sleep(3)
        else:
            self.play_beep()
            time.sleep(3.5)

        self.screen.fill(self.black)
        pygame.display.update()

        if on_windows:
            winsound.Beep(2500, 500)
            time.sleep(1.5)
        else:
            self.play_beep()
            time.sleep(2)

        if with_feedback:
            one_or_zero = random.choice([0, 1])
            smiley = [self.sad_smiley, self.happy_smiley][one_or_zero]
            self.record_data_liveEEG.add_feedback(one_or_zero)
            self.screen.blit(smiley, self.smiley_mid_pos)
            pygame.display.update()
            time.sleep(3)

        return cue_pos_choices

    def run_trial(self, with_feedback=False):
        ### this is the state-machine calling the different functions for each state
        if self.state == 1:
            ### state 1: record data nonlive
            self.cue_pos_choices_notlive = self.show_motor_imagery(self.cue_pos_choices_notlive)
            print(self.cue_pos_choices_notlive)
            
            #change state from 1 to 2 or stop recording and exit
            if len(self.cue_pos_choices_notlive) == 0 and (self.trial_count_for_each_cue_pos-self.trial_count_for_each_cue_pos_notlive) > 0:
                self.state = 2
            elif len(self.cue_pos_choices_notlive) == 0 and (self.trial_count_for_each_cue_pos-self.trial_count_for_each_cue_pos_notlive) == 0:   
                self.record_data_liveEEG.stop_recording_and_dump()
                pygame.quit()
                sys.exit()

        elif self.state == 2:
            ### state 2: do classification of notlive data and change to state 3
            self.classify_notlive()
            self.state = 3

            # start the first live trial, because state 3 starts with classification of previous trial
            print("Live trials:")
            print(self.cue_pos_choices_live)
            self.cue_pos_choices_live = self.show_motor_imagery(self.cue_pos_choices_live)
            print(self.cue_pos_choices_live)

        elif self.state == 3:
            ### state 3: do live recording and classification
            # first classify the data from previous trial and then start new trial
            filename_live = self.record_data_liveEEG.stop_recording_and_dump_live() 
            self.current_classifier, pred_true, pred_valid = self.liveEEG.classify_live(self.record_data_liveEEG.get_last_trial(filename_live))
            # display result of classification
            print('\nClassification result: ', self.current_classifier,'\n')
            if pred_true:
                print('---This is true!---\n')
            else:
                print('---This is false!---\n')
            
            # now start next trial (only if there is one trial left. otherwise stop recording and exit)
            if  len(self.cue_pos_choices_live) == 0:
                self.record_data_liveEEG.stop_recording_and_dump_live()
                pygame.quit()
                sys.exit()
            else: 
                self.cue_pos_choices_live = self.show_motor_imagery(self.cue_pos_choices_live)     
                print(self.cue_pos_choices_live)
        
        else:
            ### other
            pass
            # e.g. self.state is not defined
            
            




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="eeg experiment with pygame visualisation")
    parser.add_argument("-t", "--trials"       , help="number of trials"        , default=72   , type=int)
    parser.add_argument("-f", "--Fs"           , help="sampling frequency"      , required=True, type=int)
    parser.add_argument("-a", "--age"          , help="age of the subject"      , required=True, type=int)
    parser.add_argument("-g", "--gender"       , help="gender of the subject"   , required=True)
    parser.add_argument("-l", "--trials_notlive",help="number of trials before live", default=66,type=int)
    parser.add_argument("-w", "--with_feedback", help="with additional feedback", type=bool)
    
    args = vars(parser.parse_args())

    app = MyApp_liveEEG(args["trials"], args["trials_notlive"], args["Fs"], args["age"], gender=args["gender"], with_feedback=args["with_feedback"])
    
    app.run_session()
