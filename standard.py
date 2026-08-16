from time import time, sleep, perf_counter

def timer(func):
    def wrapper(*args, **kwargs):
        start = perf_counter()
        result = func(*args, **kwargs)
        end = perf_counter()
        print(f"{func.__name__} took {end - start:.4f} seconds")
        return result
    return wrapper

def cross2D(a,b):
    return a[:,0]*b[:,1] - a[:,1]*b[:,0]
def cross2Db(a,b):
    return a[:,:,0]*b[:,:,1] - a[:,:,1]*b[:,:,0]


def sign(x):
    if x>0: return 1
    elif x<0: return -1
    else: return 0
