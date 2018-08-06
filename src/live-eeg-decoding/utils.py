import time

def time_str():
    return time.strftime("%H_%M_%d_%m_%Y", time.gmtime())
