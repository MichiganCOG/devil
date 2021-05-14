import os
from subprocess import check_call


def merge_dicts(a, b):
    ret = dict(**a)
    for k, v in b.items():
        ret[k] = v

    return ret


def sbatch(script_path, sbatch_params, cwd=None, env=None, print_cmd=False):
    # Shallow-copy given sbatch params and add specified environment variables
    sbatch_params_with_export = dict(**sbatch_params)
    if 'export' not in sbatch_params_with_export:
        # If not specified, don't export environment variables; if specified, only export specified ones
        sbatch_params_with_export['export'] = 'NONE' if env is None else ','.join([
            '{}={}'.format(k, v) for k, v in env.items()
        ])

    # Add the current path (sbatch acts strangely without this)
    default_env = {'PATH': os.environ['PATH']}

    # Run sbatch with provided sbatch arguments and environment variables
    cmd = ['sbatch'] + [f'--{k}={v}' if v != '' else f'--{k}' for k, v in sbatch_params_with_export.items()] + [script_path]
    if print_cmd:
        print(' '.join(cmd))
    check_call(cmd, cwd=cwd, env=default_env)
