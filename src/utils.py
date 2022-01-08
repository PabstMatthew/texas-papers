import sys
import os
import pickle as pkl

# Controls whether dbg prints do anything.
DEBUG = True

# Color constants.
CYAN = '\033[96m'
BLUE = '\033[94m'
RED = '\033[92m'
BOLD = '\033[1m'
END = '\033[0m'

'''
    Helpful printing and utility functions.
'''

# Start an operation.
def dbg_start(msg):
    dbg(msg+' ...', end=' ')

# End the last operation.
def dbg_end():
    dbg('Done!', label=False)

# Print a debug message, if DEBUG is enabled.
def dbg(msg, end='\n', label=True):
    if DEBUG:
        if label:
            print(BLUE+BOLD+'[DBG]'+END, end=' ')
        print(msg, end=end)
        sys.stdout.flush()

# Prints info.
def info(msg):
    print(CYAN+BOLD+'[INFO] '+END+msg)

# Prints an error and exits.
def err(msg):
    print(RED+BOLD+'[ERR] '+END+msg)
    assert False

# Prints a warning.
def warn(msg):
    print(RED+BOLD+'[WARN] '+msg+END)

'''
    Functions to allow caching intermediate and final results.
'''

# Get the absolute path to the repository, in case the script was called from a different directory.
TOP_LEVEL_DIR = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
CACHE_PATH = os.path.join(TOP_LEVEL_DIR,'.cache')
RESOURCE_PATH = os.path.join(TOP_LEVEL_DIR,'resources')

'''
    Writes some data to the cache.
        scope: a unique string identifier for this type of data.
        name: a string identifying this particular data.
        data: some object to be cached.
        flush: if True, existing cache data will be overwritten.
'''
def cache_write(scope, name, data, flush=False):
    # Make sure the cache exists.
    if not os.path.exists(CACHE_PATH):
        os.mkdir(CACHE_PATH)
    # Form the path to this cached file.
    fpath = os.path.join(CACHE_PATH, scope+'-'+name)
    # If flushing, make sure the old file is gone.
    if flush and os.path.exists(fpath):
        os.remove(fpath)
    # Cache the file.
    if not os.path.exists(fpath):
        with open(fpath, 'wb') as f:
            pkl.dump(data, f)

'''
    Reads some data from the cache.
        scope: a unique string identifier for this type of data.
        name: a string identifying this particular data.
        returns: the cached data if it exists, otherwise None.
'''
def cache_read(scope, name):
    fpath = os.path.join(CACHE_PATH, scope+'-'+name)
    if os.path.exists(fpath):
        with open(fpath, 'rb') as f:
            return pkl.load(f)
    else:
        return None

'''
    Writes some data to the resources directory.
        scope: a unique string identifier for this type of data.
        name: a string identifying this particular data.
        data: a string to be saved.
'''
def resource_write(scope, name, data):
    if not isinstance(data, str):
        err('resource_write is reserved for string data!')
    # Form the path to this resource.
    fpath = os.path.join(RESOURCE_PATH, scope+'-'+name+'.txt')
    # Cache the file.
    with open(fpath, 'w') as f:
        f.write(data)

'''
    Reads some data from the resources directory.
        scope: a unique string identifier for this type of data.
        name: a string identifying this particular data.
        returns: the data if it exists, otherwise None.
'''
def resource_read(scope, name):
    fpath = os.path.join(RESOURCE_PATH, scope+'-'+name+'.txt')
    if os.path.exists(fpath):
        with open(fpath, 'r') as f:
            return f.read()
    else:
        return None

