import sys

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
    print(RED+BOLD+'[WARN] '+END)

