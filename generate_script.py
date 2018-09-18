import sys
"""
To run, e.g.:
        python generate_script.py 20 config.ini
Prints script for 20 runs with random hyperparameters according to config.ini
"""

if __name__ == '__main__':

    # Read command line args
    if len(sys.argv) == 1:
        print('Argument missing: requested number of runs.')
        quit()
    num_runs = int(sys.argv[1])

    # Defaults
    num_parallel_runs = 1
    config = 'config.ini'

    # See if they're overridden
    if len(sys.argv) >= 3:
        try:
            num_parallel_runs = int(sys.argv[2])
        except ValueError:
            config = sys.argv[2]
    if len(sys.argv) >= 4:
        try:
            num_parallel_runs = int(sys.argv[3])
        except ValueError:
            config = sys.argv[3]

    # Print commands for making directory
    # TODO @Carina add usr_id to directory name (currently neither in command line args nor in config).
    print("subdir=$(python create_dirs.py {0})".format(config))
    print("cp {0} models/$subdir/.".format(config))

    # And print num_runs python calls for training models:
    for run in range(1, num_runs+1):

        # provide config file, request random, subdir, number of run, and set verbosity low.
        arg_str = "-c {0} -r -d $subdir -n {1} -v 1".format(config, run)

        # TODO Do parallel runs even work? Perhaps avoid placing an '&' if this is the last run to be added?
        parallel = {0: ""}.get(run % num_parallel_runs, "&")
        arg_str += ' '+parallel

        print("python main.py", arg_str)