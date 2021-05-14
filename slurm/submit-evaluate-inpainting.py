import argparse
import os
from subprocess import check_call

from util import sbatch

SCRIPT_DIR = os.path.abspath(os.path.dirname(__file__))
PROJ_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, '..'))


def convert_extra_eval_script_arg(arg):
    """Convert the given string to an argument flag, if applicable, by converting :/:: to -/--.

    :param arg: The string to convert to an argument flag if applicable
    :return: The corresponding argument or argument flag
    """
    if arg.startswith(':'):
        return arg.replace(':', '-', 2)
    return arg


def main(model, source_split, mask_split, print_cmd, mode, use_tar, extra_sbatch_params, extra_env_vars,
         extra_eval_script_args):

    extra_eval_script_args = list(map(convert_extra_eval_script_arg, extra_eval_script_args))
    dataset_name = f'{source_split}_{mask_split}'
    tar_suffix = '.tar' if use_tar else ''

    env = {
        'SOURCE_DATASET_PATH': os.path.join(PROJ_DIR, 'datasets', 'devil', f'{source_split}{tar_suffix}'),
        'MASK_DATASET_PATH': os.path.join(PROJ_DIR, 'datasets', 'devil', f'{mask_split}{tar_suffix}'),
        'EVAL_FEATS_PATH': os.path.join(PROJ_DIR, 'eval-data', 'devil', f'{source_split}{tar_suffix}'),
        'PRED_ROOT': os.path.join(PROJ_DIR, 'inpainting-results', 'devil', dataset_name, model),
        'OUTPUT_PATH': os.path.join(PROJ_DIR, 'inpainting-results-quantitative', 'devil', dataset_name, f'{model}.npz'),
        'USER': os.environ['USER'],
        'SCRATCH_ROOT': '/tmp',
        'EXTRA_EVAL_SCRIPT_ARGS': ' '.join(extra_eval_script_args),
    }
    for env_var_name, env_var_value in zip(extra_env_vars[::2], extra_env_vars[1::2]):
        env[env_var_name] = env_var_value

    if mode == 'local':
        check_call(['bash', os.path.join(SCRIPT_DIR, 'evaluate-inpainting.sh')], env=env)
    else:
        job_name = f'evaluate-inpainting-{model}-{dataset_name}'
        sbatch_params = {
            'job-name': job_name,
            'nodes': 1,
            'gres': 'gpu:1',
            'mem': '30G',
            'time': '2:00:00',
            'output': f'logs/%j-%N-{job_name}.out',
            'error': f'logs/%j-%N-{job_name}.out',
            'mail-user': f'{os.environ["USER"]}@umich.edu',
            'mail-type': 'ALL',
        }
        for param_name, param_value in zip(extra_sbatch_params[::2], extra_sbatch_params[1::2]):
            sbatch_params[param_name] = param_value

        sbatch('evaluate-inpainting.sh', sbatch_params, SCRIPT_DIR, env, print_cmd=print_cmd)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('model', type=str)
    parser.add_argument('source_split', type=str)
    parser.add_argument('mask_split', type=str)
    parser.add_argument('--print_cmd', action='store_true')
    parser.add_argument('-m', '--mode', type=str, choices=['slurm', 'local'], default='slurm')
    parser.add_argument('-t', '--use_tar', action='store_true', help='Flag to use .tar archives')
    parser.add_argument('-s', '--extra_sbatch_params', type=str, nargs='+', default=[])
    parser.add_argument('-e', '--extra_env_vars', type=str, nargs='+', default=[])
    parser.add_argument('-a', '--extra_eval_script_args', type=str, nargs='+',
                        default=['--exclude', 'fid', 'pcons_psnr', 'pcons_ssim', 'warp_error'],
                        help=('Additional arguments to pass to evaluate_inpainting.py. Use :/:: instead of -/-- '
                              '(e.g., ::log_path instead of --log_path)'))
    args = parser.parse_args()

    assert len(args.extra_sbatch_params) % 2 == 0, \
        f'Expected an even number of arguments for --extra_sbatch_params, got {len(args.extra_sbatch_params)}'
    assert len(args.extra_env_vars) % 2 == 0, \
        f'Expected an even number of arguments for --extra_env_vars, got {len(args.extra_env_vars)}'

    main(**vars(args))
