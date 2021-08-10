import argparse
import os
from tempfile import NamedTemporaryFile

from ..common_util.global_vars import PROJ_DIR
from ..common_util.misc import extract_tar_to_path, download_url

RESULT_TO_REMOTE_URL_MAP = {
    'flickr-all_fvi-fgd-h': {
        'cpnet': 'https://web.eecs.umich.edu/~szetor/media/DEVIL/inpainting-results/devil/flickr-all_fvi-fgd-h/cpnet.tar.gz',
        'dfcnet': 'https://web.eecs.umich.edu/~szetor/media/DEVIL/inpainting-results/devil/flickr-all_fvi-fgd-h/dfcnet.tar.gz',
        'fgvc': 'https://web.eecs.umich.edu/~szetor/media/DEVIL/inpainting-results/devil/flickr-all_fvi-fgd-h/fgvc.tar.gz',
        'jointopt': 'https://web.eecs.umich.edu/~szetor/media/DEVIL/inpainting-results/devil/flickr-all_fvi-fgd-h/jointopt.tar.gz',
        'opn': 'https://web.eecs.umich.edu/~szetor/media/DEVIL/inpainting-results/devil/flickr-all_fvi-fgd-h/opn.tar.gz',
        'sttn': 'https://web.eecs.umich.edu/~szetor/media/DEVIL/inpainting-results/devil/flickr-all_fvi-fgd-h/sttn.tar.gz',
        'vinet': 'https://web.eecs.umich.edu/~szetor/media/DEVIL/inpainting-results/devil/flickr-all_fvi-fgd-h/vinet.tar.gz',
    },
    'flickr-all_fvi-fgd-l': {
        'cpnet': 'https://web.eecs.umich.edu/~szetor/media/DEVIL/inpainting-results/devil/flickr-all_fvi-fgd-l/cpnet.tar.gz',
        'dfcnet': 'https://web.eecs.umich.edu/~szetor/media/DEVIL/inpainting-results/devil/flickr-all_fvi-fgd-l/dfcnet.tar.gz',
        'fgvc': 'https://web.eecs.umich.edu/~szetor/media/DEVIL/inpainting-results/devil/flickr-all_fvi-fgd-l/fgvc.tar.gz',
        'jointopt': 'https://web.eecs.umich.edu/~szetor/media/DEVIL/inpainting-results/devil/flickr-all_fvi-fgd-l/jointopt.tar.gz',
        'opn': 'https://web.eecs.umich.edu/~szetor/media/DEVIL/inpainting-results/devil/flickr-all_fvi-fgd-l/opn.tar.gz',
        'sttn': 'https://web.eecs.umich.edu/~szetor/media/DEVIL/inpainting-results/devil/flickr-all_fvi-fgd-l/sttn.tar.gz',
        'vinet': 'https://web.eecs.umich.edu/~szetor/media/DEVIL/inpainting-results/devil/flickr-all_fvi-fgd-l/vinet.tar.gz',
    },
    'flickr-all_fvi-fgm-h': {
        'cpnet': 'https://web.eecs.umich.edu/~szetor/media/DEVIL/inpainting-results/devil/flickr-all_fvi-fgm-h/cpnet.tar.gz',
        'dfcnet': 'https://web.eecs.umich.edu/~szetor/media/DEVIL/inpainting-results/devil/flickr-all_fvi-fgm-h/dfcnet.tar.gz',
        'fgvc': 'https://web.eecs.umich.edu/~szetor/media/DEVIL/inpainting-results/devil/flickr-all_fvi-fgm-h/fgvc.tar.gz',
        'jointopt': 'https://web.eecs.umich.edu/~szetor/media/DEVIL/inpainting-results/devil/flickr-all_fvi-fgm-h/jointopt.tar.gz',
        'opn': 'https://web.eecs.umich.edu/~szetor/media/DEVIL/inpainting-results/devil/flickr-all_fvi-fgm-h/opn.tar.gz',
        'sttn': 'https://web.eecs.umich.edu/~szetor/media/DEVIL/inpainting-results/devil/flickr-all_fvi-fgm-h/sttn.tar.gz',
        'vinet': 'https://web.eecs.umich.edu/~szetor/media/DEVIL/inpainting-results/devil/flickr-all_fvi-fgm-h/vinet.tar.gz',
    },
    'flickr-all_fvi-fgm-l': {
        'cpnet': 'https://web.eecs.umich.edu/~szetor/media/DEVIL/inpainting-results/devil/flickr-all_fvi-fgm-l/cpnet.tar.gz',
        'dfcnet': 'https://web.eecs.umich.edu/~szetor/media/DEVIL/inpainting-results/devil/flickr-all_fvi-fgm-l/dfcnet.tar.gz',
        'fgvc': 'https://web.eecs.umich.edu/~szetor/media/DEVIL/inpainting-results/devil/flickr-all_fvi-fgm-l/fgvc.tar.gz',
        'jointopt': 'https://web.eecs.umich.edu/~szetor/media/DEVIL/inpainting-results/devil/flickr-all_fvi-fgm-l/jointopt.tar.gz',
        'opn': 'https://web.eecs.umich.edu/~szetor/media/DEVIL/inpainting-results/devil/flickr-all_fvi-fgm-l/opn.tar.gz',
        'sttn': 'https://web.eecs.umich.edu/~szetor/media/DEVIL/inpainting-results/devil/flickr-all_fvi-fgm-l/sttn.tar.gz',
        'vinet': 'https://web.eecs.umich.edu/~szetor/media/DEVIL/inpainting-results/devil/flickr-all_fvi-fgm-l/vinet.tar.gz',
    },
    'flickr-all_fvi-fgs-h': {
        'cpnet': 'https://web.eecs.umich.edu/~szetor/media/DEVIL/inpainting-results/devil/flickr-all_fvi-fgs-h/cpnet.tar.gz',
        'dfcnet': 'https://web.eecs.umich.edu/~szetor/media/DEVIL/inpainting-results/devil/flickr-all_fvi-fgs-h/dfcnet.tar.gz',
        'fgvc': 'https://web.eecs.umich.edu/~szetor/media/DEVIL/inpainting-results/devil/flickr-all_fvi-fgs-h/fgvc.tar.gz',
        'jointopt': 'https://web.eecs.umich.edu/~szetor/media/DEVIL/inpainting-results/devil/flickr-all_fvi-fgs-h/jointopt.tar.gz',
        'opn': 'https://web.eecs.umich.edu/~szetor/media/DEVIL/inpainting-results/devil/flickr-all_fvi-fgs-h/opn.tar.gz',
        'sttn': 'https://web.eecs.umich.edu/~szetor/media/DEVIL/inpainting-results/devil/flickr-all_fvi-fgs-h/sttn.tar.gz',
        'vinet': 'https://web.eecs.umich.edu/~szetor/media/DEVIL/inpainting-results/devil/flickr-all_fvi-fgs-h/vinet.tar.gz',
    },
    'flickr-all_fvi-fgs-l': {
        'cpnet': 'https://web.eecs.umich.edu/~szetor/media/DEVIL/inpainting-results/devil/flickr-all_fvi-fgs-l/cpnet.tar.gz',
        'dfcnet': 'https://web.eecs.umich.edu/~szetor/media/DEVIL/inpainting-results/devil/flickr-all_fvi-fgs-l/dfcnet.tar.gz',
        'fgvc': 'https://web.eecs.umich.edu/~szetor/media/DEVIL/inpainting-results/devil/flickr-all_fvi-fgs-l/fgvc.tar.gz',
        'jointopt': 'https://web.eecs.umich.edu/~szetor/media/DEVIL/inpainting-results/devil/flickr-all_fvi-fgs-l/jointopt.tar.gz',
        'opn': 'https://web.eecs.umich.edu/~szetor/media/DEVIL/inpainting-results/devil/flickr-all_fvi-fgs-l/opn.tar.gz',
        'sttn': 'https://web.eecs.umich.edu/~szetor/media/DEVIL/inpainting-results/devil/flickr-all_fvi-fgs-l/sttn.tar.gz',
        'vinet': 'https://web.eecs.umich.edu/~szetor/media/DEVIL/inpainting-results/devil/flickr-all_fvi-fgs-l/vinet.tar.gz',
    },
    'flickr-bsm-h_fvi-all': {
        'cpnet': 'https://web.eecs.umich.edu/~szetor/media/DEVIL/inpainting-results/devil/flickr-bsm-h_fvi-all/cpnet.tar.gz',
        'dfcnet': 'https://web.eecs.umich.edu/~szetor/media/DEVIL/inpainting-results/devil/flickr-bsm-h_fvi-all/dfcnet.tar.gz',
        'fgvc': 'https://web.eecs.umich.edu/~szetor/media/DEVIL/inpainting-results/devil/flickr-bsm-h_fvi-all/fgvc.tar.gz',
        'jointopt': 'https://web.eecs.umich.edu/~szetor/media/DEVIL/inpainting-results/devil/flickr-bsm-h_fvi-all/jointopt.tar.gz',
        'opn': 'https://web.eecs.umich.edu/~szetor/media/DEVIL/inpainting-results/devil/flickr-bsm-h_fvi-all/opn.tar.gz',
        'sttn': 'https://web.eecs.umich.edu/~szetor/media/DEVIL/inpainting-results/devil/flickr-bsm-h_fvi-all/sttn.tar.gz',
        'vinet': 'https://web.eecs.umich.edu/~szetor/media/DEVIL/inpainting-results/devil/flickr-bsm-h_fvi-all/vinet.tar.gz',
    },
    'flickr-bsm-l_fvi-all': {
        'cpnet': 'https://web.eecs.umich.edu/~szetor/media/DEVIL/inpainting-results/devil/flickr-bsm-l_fvi-all/cpnet.tar.gz',
        'dfcnet': 'https://web.eecs.umich.edu/~szetor/media/DEVIL/inpainting-results/devil/flickr-bsm-l_fvi-all/dfcnet.tar.gz',
        'fgvc': 'https://web.eecs.umich.edu/~szetor/media/DEVIL/inpainting-results/devil/flickr-bsm-l_fvi-all/fgvc.tar.gz',
        'jointopt': 'https://web.eecs.umich.edu/~szetor/media/DEVIL/inpainting-results/devil/flickr-bsm-l_fvi-all/jointopt.tar.gz',
        'opn': 'https://web.eecs.umich.edu/~szetor/media/DEVIL/inpainting-results/devil/flickr-bsm-l_fvi-all/opn.tar.gz',
        'sttn': 'https://web.eecs.umich.edu/~szetor/media/DEVIL/inpainting-results/devil/flickr-bsm-l_fvi-all/sttn.tar.gz',
        'vinet': 'https://web.eecs.umich.edu/~szetor/media/DEVIL/inpainting-results/devil/flickr-bsm-l_fvi-all/vinet.tar.gz',
    },
    'flickr-cm-h_fvi-all': {
        'cpnet': 'https://web.eecs.umich.edu/~szetor/media/DEVIL/inpainting-results/devil/flickr-cm-h_fvi-all/cpnet.tar.gz',
        'dfcnet': 'https://web.eecs.umich.edu/~szetor/media/DEVIL/inpainting-results/devil/flickr-cm-h_fvi-all/dfcnet.tar.gz',
        'fgvc': 'https://web.eecs.umich.edu/~szetor/media/DEVIL/inpainting-results/devil/flickr-cm-h_fvi-all/fgvc.tar.gz',
        'jointopt': 'https://web.eecs.umich.edu/~szetor/media/DEVIL/inpainting-results/devil/flickr-cm-h_fvi-all/jointopt.tar.gz',
        'opn': 'https://web.eecs.umich.edu/~szetor/media/DEVIL/inpainting-results/devil/flickr-cm-h_fvi-all/opn.tar.gz',
        'sttn': 'https://web.eecs.umich.edu/~szetor/media/DEVIL/inpainting-results/devil/flickr-cm-h_fvi-all/sttn.tar.gz',
        'vinet': 'https://web.eecs.umich.edu/~szetor/media/DEVIL/inpainting-results/devil/flickr-cm-h_fvi-all/vinet.tar.gz',
    },
    'flickr-cm-l_fvi-all': {
        'cpnet': 'https://web.eecs.umich.edu/~szetor/media/DEVIL/inpainting-results/devil/flickr-cm-l_fvi-all/cpnet.tar.gz',
        'dfcnet': 'https://web.eecs.umich.edu/~szetor/media/DEVIL/inpainting-results/devil/flickr-cm-l_fvi-all/dfcnet.tar.gz',
        'fgvc': 'https://web.eecs.umich.edu/~szetor/media/DEVIL/inpainting-results/devil/flickr-cm-l_fvi-all/fgvc.tar.gz',
        'jointopt': 'https://web.eecs.umich.edu/~szetor/media/DEVIL/inpainting-results/devil/flickr-cm-l_fvi-all/jointopt.tar.gz',
        'opn': 'https://web.eecs.umich.edu/~szetor/media/DEVIL/inpainting-results/devil/flickr-cm-l_fvi-all/opn.tar.gz',
        'sttn': 'https://web.eecs.umich.edu/~szetor/media/DEVIL/inpainting-results/devil/flickr-cm-l_fvi-all/sttn.tar.gz',
        'vinet': 'https://web.eecs.umich.edu/~szetor/media/DEVIL/inpainting-results/devil/flickr-cm-l_fvi-all/vinet.tar.gz',
    },
}


def path_info_to_remote_file_url(split, model):
    try:
        url = RESULT_TO_REMOTE_URL_MAP[split][model]
    except KeyError:
        raise ValueError(f'Failed to find remote file name for split {split} and model {model}')
    return url


def main(model, source_split, mask_split):
    comb_split_name = f'{source_split}_{mask_split}'
    try:
        remote_file_url = path_info_to_remote_file_url(comb_split_name, model)
    except ValueError:
        raise RuntimeError('Failed to find a valid URL for the given arguments')

    with NamedTemporaryFile() as temp_file:
        download_url(remote_file_url, temp_file)
        output_dir = os.path.join(PROJ_DIR, 'inpainting-results', 'devil', comb_split_name, model)
        extract_tar_to_path(output_dir, temp_file.name)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('model', type=str)
    parser.add_argument('source_split', type=str)
    parser.add_argument('mask_split', type=str)
    args = parser.parse_args()

    main(**vars(args))
