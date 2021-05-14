import argparse
import os

from ..common_util.global_vars import PROJ_DIR
from ..common_util.misc import extract_tar_to_path, download_url, makedirs

URL_FORMAT = 'https://umich.box.com/shared/static/{remote_file_name}'

SPLIT_TO_REMOTE_FILE_NAME_MAP = {
    'flickr-all': '2ixhwq68a6vdq3v2bset8wu03z6izw7x.tar',
    'flickr-bsm-h': '2apilfdbam18aa54rbp2qo4tssz4h044.tar',
    'flickr-bsm-l': 'vqmszha3kn3h7pst27g38rpav6hyk1k2.tar',
    'flickr-cm-h': 'pvjpf74zdyh2c0m62t1nm04gnlkpo0lt.tar',
    'flickr-cm-l': 'cmvrpjrvoz645u8p5x0ldxofzf7gafz6.tar',
    'fvi-all': 'li9i0bfpzjza2ygoinqbyuuc7jovihzl.tar',
    'fvi-fgd-h': 'cblb2llbrxnos7qpf9kzo4yyiuonur64.tar',
    'fvi-fgd-l': 'wmm15os8ys8qt27epfe6ua419dcj4o1c.tar',
    'fvi-fgm-h': 'b432x8h49fq18sjcx2dq0qtyjbwalyez.tar',
    'fvi-fgm-l': '4xtmjgc7o1j4zrlzvtkkme6mgszaup9d.tar',
    'fvi-fgs-h': 'cgqksxyy9ffa8j5qtmj2q4tt2epx27bm.tar',
    'fvi-fgs-l': 't4wd3fv48ogvfx2uqm5tijgflfkdqklv.tar',
}


def main(splits, extract):
    for split in splits:
        remote_file_name = SPLIT_TO_REMOTE_FILE_NAME_MAP[split]
        remote_file_url = URL_FORMAT.format(remote_file_name=remote_file_name)
        save_tar_path = os.path.join(PROJ_DIR, 'datasets', 'devil', f'{split}.tar')
        makedirs(os.path.dirname(save_tar_path))
        with open(save_tar_path, 'wb') as f:
            download_url(remote_file_url, f)

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
