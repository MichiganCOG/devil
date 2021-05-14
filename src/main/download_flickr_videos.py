import argparse
import os

import flickrapi
import yaml

from ..flickr.util import video_metadata_to_url, FlickrVideoDownloader

SCRIPT_DIR = os.path.abspath(os.path.dirname(__file__))
PROJ_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..'))


def main(force_overwrite, num_threads):
    """The target function for this script.

    :param force_overwrite: Whether to overwrite previously-downloaded videos
    :param num_threads: How many concurrent workers to use when downloading videos
    """

    # Create save root
    save_root = os.path.join(PROJ_ROOT, 'datasets', 'flickr-raw')
    if not os.path.isdir(save_root):
        os.makedirs(save_root)

    # Read config
    with open('flickr-config.yml') as f:
        config = yaml.safe_load(f)

    # Get metadata for the desired videos
    all_metadata = []
    flickr = flickrapi.FlickrAPI(config['api_key'], config['api_secret'], format='parsed-json')
    for video_id in config['whitelisted_videos']:
        metadata = flickr.photos.getInfo(photo_id=video_id)['photo']
        all_metadata.append(metadata)

    # Add URLs to work queue
    flickr_video_downloader = FlickrVideoDownloader(save_root, num_threads, force_overwrite)
    for metadata in all_metadata:
        url = video_metadata_to_url(metadata)
        flickr_video_downloader.add_url(url)

    # Wait for all URLs to finish downloading
    flickr_video_downloader.join()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--force-overwrite', action='store_true',
                        help='Flag to overwrite previously-downloaded videos')
    parser.add_argument('-t', '--num_threads', type=int, default=4,
                        help='The number of concurrent threads used to download videos')
    args = parser.parse_args()

    main(**vars(args))
