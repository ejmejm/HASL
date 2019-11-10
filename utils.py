OBS_DIM = 64 # Must be a multiple of 2^4 with the current model
OBS_DEPTH = 3

### Logging ###

def init_logger(lp):
    global log_path
    log_path = lp

    f = open(log_path, 'w+')
    f.close()

def log(string):
    print(string)

    with open(log_path, 'a') as f:
        f.write(string + '\n')