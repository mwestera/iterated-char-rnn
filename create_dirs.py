import os
import config_utils
import sys

"""
To be run prior to training a batch of models sampled from the same config file.
It takes the config file as argument, and creates & prints the appropriate directory name, 
based on those settings from the config that are to be randomly sampled. 
"""

if len(sys.argv) <= 1:
    print("Create dirs says nope.")
    quit()

def make_dir_keep_existing(path, sep=''):
    """
    Creates a directory for the given path. If the directory already exists, it first changes the path name by apending a number.
    Source: https://stackoverflow.com/questions/13852700/python-create-file-but-if-name-exists-add-number
    """
    import itertools
    import tempfile
    def name_sequence():
        count = itertools.count()
        yield ''
        next(count)
        while True:
            yield '{s}{n:d}'.format(s = sep, n = next(count))
    orig = tempfile._name_sequence 
    with tempfile._once_lock:
        tempfile._name_sequence = name_sequence() # temporarily overwrite tempfile's method
        path = os.path.normpath(os.path.join('models', path))
        parent, dirname = os.path.split(path)
        dirname = tempfile.mkdtemp(dir = parent, prefix = dirname)
        tempfile._name_sequence = orig
    return dirname

fixed_params_string = config_utils.fixed_params_to_string(sys.argv[1])
subdir = os.path.basename(make_dir_keep_existing(fixed_params_string, sep="--"))

print(subdir)