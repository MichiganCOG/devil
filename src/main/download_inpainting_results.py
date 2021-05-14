import argparse
import os
from tempfile import NamedTemporaryFile

from ..common_util.global_vars import PROJ_DIR
from ..common_util.misc import extract_tar_to_path, download_url

URL_FORMAT = 'https://umich.box.com/shared/static/{remote_file_name}'

RESULT_TO_REMOTE_FILE_NAME_MAP = {
    'flickr-all_fvi-fgd-h': {
        'cpnet': 'wrgzs6wl1dhmiaiu85ifza96y4dlgb94.gz',
        'dfcnet': '67tor1xqp4thyzms4jryz58543xxwteg.gz',
        'fgvc': 'janlkrrh5f9d6o3k4bfrug0je55mje1z.gz',
        'jointopt': 'mprquyjqtgw8jxtxgcdtps9syf97zdhe.gz',
        'opn': '2evqtrtldr48nq01xfvzu12qsqdux1od.gz',
        'sttn': 'g8xo76ro87hvi2t41hp9t8os36yjwc3i.gz',
        'vinet': '6264n7fassb55tf48ofdfkpun676vfi5.gz',
    },
    'flickr-all_fvi-fgd-l': {
        'cpnet': 'sob11uc7y0nt5uiqa37fss4io1pkynqu.gz',
        'dfcnet': 'caewi3gx6puzoeazeze7j7ldubfqnlz8.gz',
        'fgvc': 'rt19aihgw7x10pzh0u72p78z2np92hp8.gz',
        'jointopt': 'xegg43btw4rl59czixii8w4y6y46mrf1.gz',
        'opn': 'rp4s7d930kvkqsxkgk7w4265pv3ua8er.gz',
        'sttn': 'rm8vd7bzt9s7qwum6xkz03sz0axojur0.gz',
        'vinet': 'gzcful4okii4msa3xdzb4d4mvlbig8bv.gz',
    },
    'flickr-all_fvi-fgm-h': {
        'cpnet': 'xhp66a4ig041kkumromev5grwjzho9m2.gz',
        'dfcnet': 'az9ekplzvzxt2oob5er2umyecl8pgymp.gz',
        'fgvc': 'nvby1me8ldtriflz9q4gv8dy2rkzcfiz.gz',
        'jointopt': 'tsbszv4rttz47bdanxkbpvg3nkdixbcb.gz',
        'opn': 'tbels0smzopg5oxifq6bsh4umjwlc68m.gz',
        'sttn': 'xe6v6wofojot8le52hr8njxtav6jaces.gz',
        'vinet': 'oe52cdzt38d97hz5qr0n4mk6peqbnojb.gz',
    },
    'flickr-all_fvi-fgm-l': {
        'cpnet': 'p2usx99lgcbx0juzn5seligf7s54nvdw.gz',
        'dfcnet': 'aunhd3uijr9l8cao3ypj4myjqobcj4wa.gz',
        'fgvc': '3xho76ybzp88yslctp8paioo7oe0wdgt.gz',
        'jointopt': '6b7haikjn6wpmo4ak0yqr7li4o3iezoo.gz',
        'opn': 'urtbupfalnolcsfcas7jmu2263fidhxa.gz',
        'sttn': '4nx699628hx138apyja3h4kwvqysvfet.gz',
        'vinet': 'iyp77x02mgl4ss6rnlo9zkdvcg6e8iy9.gz',
    },
    'flickr-all_fvi-fgs-h': {
        'cpnet': 'psc4u4z2kfqiphp6qyql89zg4nyvxo1r.gz',
        'dfcnet': 'g5ygf0cv4hh13dl5ds7sn0w0oa80lzdw.gz',
        'fgvc': '3ncyub4v1kowcevdzv0hb9fd48vu7f84.gz',
        'jointopt': 'o3837yloth7tmif3mzzdqezsvpxzvahj.gz',
        'opn': 'ps30o8ivgfsvrokuayie1nfmhr2tzemj.gz',
        'sttn': '09kzy1bwowlrm1rjbis1axv5o1spu5zt.gz',
        'vinet': 'j76vtx3r725750k3taeu98w27jk87fao.gz',
    },
    'flickr-all_fvi-fgs-l': {
        'cpnet': 'pzq8tbmn20nlvrot9fwupnrwl5iz1gqp.gz',
        'dfcnet': 'id6g4fkuav13t7gupfx39bhv7s4q50pa.gz',
        'fgvc': 'fftwrk613ibhkpt5420fhcksguj33jv5.gz',
        'jointopt': 'myj3h6d89a81wo3bi8c3wogcvdwvcygw.gz',
        'opn': '9wf3ysjbktipe6p8w5r2l36adp1ijvd6.gz',
        'sttn': 'g8fsmemb4s4d8820thfpsqbhg7o02465.gz',
        'vinet': 'zla1vpp7n3toq6dqhnaq9f8424k19cff.gz',
    },
    'flickr-bsm-h_fvi-all': {
        'cpnet': 'drqgjarp4r1yu42so2ymjvy7hj9eav57.gz',
        'dfcnet': 'lzpimjt3m9yv7c5go9gv2q0bd9rofdpm.gz',
        'fgvc': 'ltnhd817l70lro19k2cq1inu1lkyl37e.gz',
        'jointopt': 'mvozo2ls65c6jf2948p4s6s53lim91i4.gz',
        'opn': 's4oyzbtu7pb7wpdpph60s74zepox9pg0.gz',
        'sttn': 'lb3niys4gvqlyrv567lttxscz93dqi8j.gz',
        'vinet': 'd70ceojp3r2lx8ey4h6v2ytw8yublqic.gz',
    },
    'flickr-bsm-l_fvi-all': {
        'cpnet': '6w9hndg88dynh7pfrpykslgzp8nv7nmi.gz',
        'dfcnet': '05dps08cknh4y0xlkddamntdz63jvret.gz',
        'fgvc': 'bjjjux4jjjd965qmff3vy678jn1skoox.gz',
        'jointopt': 'qqia5ov509u2cijggl7tt8wpaupkcqs8.gz',
        'opn': '7pmrpsby2byokx4fy5jez8w6ce8njzth.gz',
        'sttn': 'h9j6diqnpdxcphupjq3lekcy8wwzu603.gz',
        'vinet': 'ow2a4irabvdw5i4ko2e3wff1r4tkvz9h.gz',
    },
    'flickr-cm-h_fvi-all': {
        'cpnet': 'spog460y4oxgycqei0gg7gjr7akkqtii.gz',
        'dfcnet': 'nx6foa33rbuzhcjqsu1b00h9e9iyate2.gz',
        'fgvc': 'pvt1lcacwpgilgbb9yq9do8k1vq2d0dq.gz',
        'jointopt': 'sqsqpftfsh5hm3uzfsan0ifrgz781idm.gz',
        'opn': 'hsmy0zh4jvpjupfxo4lwhlkgx3exiipl.gz',
        'sttn': 'g3m8kheb9owqxdud71sjq9bsai1qz685.gz',
        'vinet': '9jh8t4lkbe0tq6ga00eqnje32kpdhah1.gz',
    },
    'flickr-cm-l_fvi-all': {
        'cpnet': 'ihit25fhx2fs5ra2s2f3f4kq9h5m2rw2.gz',
        'dfcnet': '5dcxg1sf3otl2dq9g8a78mb4zjyj8j3w.gz',
        'fgvc': 'byiyi4lye1fnrsyjhljvlup036aw28na.gz',
        'jointopt': 'cewhbsvzeumhptt2sd1ag6vw99dm3s8u.gz',
        'opn': '8l6vbndrqlucisjxznts62q4w78lmpze.gz',
        'sttn': '6m971qaozsp12snm2u177e3rjth81j0m.gz',
        'vinet': '0gsbu49zbzkvqwng20p8kbo28spvpysv.gz',
    },
}


def path_info_to_remote_file_url(split, model):
    try:
        remote_file_name = RESULT_TO_REMOTE_FILE_NAME_MAP[split][model]
    except KeyError:
        raise ValueError(f'Failed to find remote file name for split {split} and model {model}')
    return URL_FORMAT.format(remote_file_name=remote_file_name)


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
