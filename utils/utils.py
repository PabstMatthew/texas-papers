import sys
import os
import pickle as pkl

DEBUG = True

BLUE = '\033[94m'
RED = '\033[92m'
BOLD = '\033[1m'
END = '\033[0m'

def dbg_start(msg):
    dbg(msg+' ...', end=' ')

def dbg_end():
    dbg('Done!', label=False)

def dbg(msg, end='\n', label=True):
    if DEBUG:
        if label:
            print(BLUE+BOLD+'[DBG]'+END, end=' ')
        print(msg, end=end)
        sys.stdout.flush()

def err(msg):
    print(RED+BOLD+'[ERR] '+END+msg)
    assert False

def warn(msg):
    print(RED+BOLD+'[WARN] '+msg+END)

CACHE_PATH = os.path.expanduser('~/.cache/texas-papers')

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

def cache_read(scope, name):
    fpath = os.path.join(CACHE_PATH, scope+'-'+name)
    if os.path.exists(fpath):
        with open(fpath, 'rb') as f:
            return pkl.load(f)
    else:
        return None

