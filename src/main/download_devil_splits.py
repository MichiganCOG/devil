import argparse
import os

from ..common_util.global_vars import PROJ_DIR
from ..common_util.misc import extract_tar_to_path, download_url, makedirs

SPLIT_TO_REMOTE_URL_MAP = {
    'flickr-all': 'https://web.eecs.umich.edu/~szetor/media/DEVIL/datasets/devil/flickr-all.tar',
    'flickr-bsm-h': 'https://web.eecs.umich.edu/~szetor/media/DEVIL/datasets/devil/flickr-bsm-h.tar',
    'flickr-bsm-l': 'https://web.eecs.umich.edu/~szetor/media/DEVIL/datasets/devil/flickr-bsm-l.tar',
    'flickr-cm-h': 'https://web.eecs.umich.edu/~szetor/media/DEVIL/datasets/devil/flickr-cm-h.tar',
    'flickr-cm-l': 'https://web.eecs.umich.edu/~szetor/media/DEVIL/datasets/devil/flickr-cm-l.tar',
    'fvi-all': 'https://web.eecs.umich.edu/~szetor/media/DEVIL/datasets/devil/fvi-all.tar',
    'fvi-fgd-h': 'https://web.eecs.umich.edu/~szetor/media/DEVIL/datasets/devil/fvi-fgd-h.tar',
    'fvi-fgd-l': 'https://web.eecs.umich.edu/~szetor/media/DEVIL/datasets/devil/fvi-fgd-l.tar',
    'fvi-fgm-h': 'https://web.eecs.umich.edu/~szetor/media/DEVIL/datasets/devil/fvi-fgm-h.tar',
    'fvi-fgm-l': 'https://web.eecs.umich.edu/~szetor/media/DEVIL/datasets/devil/fvi-fgm-l.tar',
    'fvi-fgs-h': 'https://web.eecs.umich.edu/~szetor/media/DEVIL/datasets/devil/fvi-fgs-h.tar',
    'fvi-fgs-l': 'https://web.eecs.umich.edu/~szetor/media/DEVIL/datasets/devil/fvi-fgs-l.tar',
}


def main(splits, extract):
    for split in splits:
        url = SPLIT_TO_REMOTE_URL_MAP[split]
        save_tar_path = os.path.join(PROJ_DIR, 'datasets', 'devil', f'{split}.tar')
        makedirs(os.path.dirname(save_tar_path))
        with open(save_tar_path, 'wb') as f:
            download_url(url, f)

        if extract:
            output_dir = os.path.join(PROJ_DIR, 'datasets', 'devil', split)
            makedirs(output_dir)
            extract_tar_to_path(output_dir, save_tar_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('splits', type=str, nargs='+')
    parser.add_argument('-e', '--extract', action='store_true',
                        help='Whether to extract the downloaded .tar file in addition to saving it')
    args = parser.parse_args()

    main(**vars(args))
