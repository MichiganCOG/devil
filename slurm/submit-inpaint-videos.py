import argparse
import os
from subprocess import check_call

from util import merge_dicts, sbatch

SCRIPT_DIR = os.path.abspath(os.path.dirname(__file__))

SBATCH_PARAMS_EXTRA = {
    'cpnet': {
        'cpus-per-task': 8,
        'gres': 'gpu:1',
        'mem': '10G',
        'time': '8:00:00',
    },
    'dfcnet': {
        'cpus-per-task': 8,
        'gres': 'gpu:1',
        'mem': '60G',
        'time': '3-00:00:00',
    },
    'fgvc': {
        'cpus-per-task': 8,
        'gres': 'gpu:1',
        'mem': '10G',
        'time': '3-00:00:00',
    },
    'jointopt': {
        'cpus-per-task': 8,
        'mem': '60G',
        'time': '2-00:00:00',
    },
    'lgtsm': {
        'cpus-per-task': 8,
        'gres': 'gpu:1',
        'mem': '10G',
        'time': '8:00:00',
    },
    'opn': {
        'cpus-per-task': 8,
        'gres': 'gpu:1',
        'mem': '10G',
        'time': '1-12:00:00',
    },
    'sttn': {
        'cpus-per-task': 8,
        'gres': 'gpu:1',
        'mem': '15G',
        'time': '8:00:00',
    },
    'vinet': {
        'cpus-per-task': 8,
        'gres': 'gpu:1',
        'mem': '10G',
        'time': '4:30:00',
    },
}

MODEL_NAME_RUN_PATH_MAP = {
    'cpnet': 'video-inpainting-projects/Copy-and-Paste-Networks-for-Deep-Video-Inpainting/devil-run.sh',
    'dfcnet': 'video-inpainting-projects/Deep-Flow-Guided-Video-Inpainting/devil-run.sh',
    'fgvc': 'video-inpainting-projects/FGVC/devil-run.sh',
    'jointopt': 'video-inpainting-projects/huang-video-completion/devil-run.sh',
    'lgtsm': 'video-inpainting-projects/Free-Form-Video-Inpainting-2/devil-run.sh',
    'opn': 'video-inpainting-projects/onion-peel-network/devil-run.sh',
    'sttn': 'video-inpainting-projects/STTN/devil-run.sh',
    'vinet': 'video-inpainting-projects/VINet/devil-run.sh',
}

MODEL_NAME_IMAGE_SIZE_DIVISOR_MAP = {
    'cpnet': '8',
    'dfcnet': '64',
    'fgvc': '8',
    'jointopt': '1',
    'lgtsm': '8',
    'opn': '8',
    'sttn': '432 240',
    'vinet': '16',
}


def main(model, source_split, mask_split, mode, index_range, extra_sbatch_params, extra_env_vars):
    assert model in SBATCH_PARAMS_EXTRA
    assert model in MODEL_NAME_RUN_PATH_MAP
    assert model in MODEL_NAME_IMAGE_SIZE_DIVISOR_MAP

    dataset_name = f'{source_split}_{mask_split}'

    env = {
        'RUN_PATH': MODEL_NAME_RUN_PATH_MAP[model],
        'IMAGE_SIZE_DIVISOR': MODEL_NAME_IMAGE_SIZE_DIVISOR_MAP[model],
        'FRAMES_DATASET_PATH': os.path.join('datasets', 'devil', f'{source_split}.tar'),
        'MASKS_DATASET_PATH': os.path.join('datasets', 'devil', f'{mask_split}.tar'),
        'INPAINTING_RESULTS_ROOT': os.path.join('inpainting-results', 'devil', dataset_name, model),
        'TEMP_ROOT': '/tmp',
        'INDEX_RANGE': '' if index_range is None else f'{index_range[0]}-{index_range[1]}'
    }
    for env_var_name, env_var_value in zip(extra_env_vars[::2], extra_env_vars[1::2]):
        env[env_var_name] = env_var_value

    if mode == 'local':
        check_call(['bash', os.path.join(SCRIPT_DIR, 'inpaint-videos.sh')], env=env)
    else:
        job_name = f'inpaint-videos-{model}-{dataset_name}'
        if index_range is not None:
            job_name += f'_{index_range[0]}-{index_range[1]}'
        basic_sbatch_params = {
            'job-name': job_name,
            'nodes': 1,
            'output': f'logs/%j-%N-{job_name}.out',
            'error': f'logs/%j-%N-{job_name}.out',
            'mail-user': f'{os.environ["USER"]}@umich.edu',
            'mail-type': 'ALL',
        }
        sbatch_params = merge_dicts(basic_sbatch_params, SBATCH_PARAMS_EXTRA[model])

        for param_name, param_value in zip(extra_sbatch_params[::2], extra_sbatch_params[1::2]):
            sbatch_params[param_name] = param_value

        sbatch('inpaint-videos.sh', sbatch_params, SCRIPT_DIR, env)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('model', type=str)
    parser.add_argument('source_split', type=str)
    parser.add_argument('mask_split', type=str)
    parser.add_argument('-m', '--mode', type=str, choices=['slurm', 'local'], default='slurm')
    parser.add_argument('-i', '--index_range', type=int, nargs=2, default=None)
    parser.add_argument('-s', '--extra_sbatch_params', type=str, nargs='+', default=[])
    parser.add_argument('-e', '--extra_env_vars', type=str, nargs='+', default=[])
    args = parser.parse_args()

    assert len(args.extra_sbatch_params) % 2 == 0, \
        f'Expected an even number of arguments for --extra_sbatch_params, got {len(args.extra_sbatch_params)}'
    assert len(args.extra_env_vars) % 2 == 0, \
        f'Expected an even number of arguments for --extra_env_vars, got {len(args.extra_env_vars)}'

    main(**vars(args))
