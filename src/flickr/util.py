import os
import re
import shutil
from queue import Queue
from threading import Thread, RLock
from urllib.error import HTTPError
from urllib.request import urlopen
from warnings import warn

import parse

LOG_LOCK = RLock()
NUM_RESULTS_PER_PAGE = 25
VIDEO_URL_FORMAT = 'http://www.flickr.com/photos/{user_id}/{video_id}/play/1080p/{secret}'


def log(type, message, **kwargs):
    """Write a message to the console in a thread-safe manner.

    :param type: What kind of message to write (choices: 'print', 'warn')
    :param message: The message to write
    :param kwargs: Other arguments to pass to the actual logging function
    """

    with LOG_LOCK:
        if type == 'print':
            print(message, **kwargs)
        elif type == 'warn':
            warn(message, **kwargs)


def run_download_flickr_video(queue, save_root, force_overwrite):
    """The target function for the URL downloader threads.

    :param queue: The job queue containing URLs to download
    :param save_root: The root directory under which to save videos
    :param force_overwrite: Whether to overwrite previously-downloaded videos
    """

    while True:
        url = queue.get()
        try:
            save_path = download_flickr_video(url, save_root, force_overwrite)
            log('print', f'Saved video to {save_path}')
        except HTTPError as e:
            if e.code == 404:
                log('warn', f'HTTP error 404 returned for URL {url}')
        except FileExistsError as e:
            log('warn', f'File already exists for URL {url}, skipping')

        queue.task_done()


def video_metadata_to_url(video_meta):
    """Converts Flickr video metadata to a downloadable URL.

    The URL may lead to a 404 if a high-resolution copy of the video is not available, or if the given metadata does not
    correspond to a video. Downstream processing is required to handle such cases.

    :param video_meta: The metadata for a given Flickr item
    :return: A valid URL to the given Flickr video
    """

    video_id = video_meta['id']
    user_id = video_meta['owner']['nsid']
    secret = video_meta['secret']
    url = VIDEO_URL_FORMAT.format(user_id=user_id, video_id=video_id, secret=secret)

    return url


def download_flickr_video(url, save_root, force_overwrite):
    """Downloads a Flickr video if the URL exists.

    :param url: The URL to try to download
    :param save_root: The folder to save the video to
    :param force_overwrite: Whether to overwrite previously-downloaded videos
    :return: The path that the video was saved to
    """

    # (Try to) open the URL
    response = urlopen(url)
    # Extract the file extension from the resolved URL
    m = re.match(r'(.*)\?s=.*', response.url)
    _, ext = os.path.splitext(m.group(1))
    # Build the path to save the video to
    video_meta = parse.parse(VIDEO_URL_FORMAT, url)
    user_id = video_meta['user_id']
    video_id = video_meta['video_id']
    save_path = os.path.join(save_root, f'{user_id}-{video_id}{ext}')
    # Save the video
    if os.path.isfile(save_path) and not force_overwrite:
        raise FileExistsError(f'File already exists at {save_path}')
    else:
        with open(save_path, 'wb') as f:
            shutil.copyfileobj(response, f)

    return save_path


class FlickrVideoDownloader(object):

    def __init__(self, save_root, num_threads=4, force_overwrite=False):
        """Constructor

        :param save_root: The directory where all videos will be saved
        :param num_threads: How many worker threads to use
        :param force_overwrite: Whether to overwrite previously-downloaded videos
        """
        self.queue = Queue()
        self.thread_pool = [
            Thread(
                target=run_download_flickr_video,
                daemon=True,
                args=(self.queue, save_root, force_overwrite)
            ) for _ in range(num_threads)
        ]
        for thread in self.thread_pool:
            thread.start()


    def add_url(self, url):
        """Queues a given video URL for download.

        :param url: URL to an MP4 file
        """
        self.queue.put(url)


    def join(self):
        """Closes the queue and waits for all jobs to finish."""
        self.queue.join()
