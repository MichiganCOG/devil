import os
import re
import tarfile
import traceback
from glob import glob
from inspect import isclass
from queue import Queue
from subprocess import check_output, CalledProcessError
from threading import Thread

import numpy as np
import pandas as pd
import requests
from sklearn.metrics import auc
from tqdm import tqdm


def makedirs(path):
    """Constructs a folder at the given path if it does not exist; otherwise, does nothing. NOT THREAD-SAFE."""
    if len(path) > 0 and not os.path.isdir(path):
        os.makedirs(path)


def get_all_frame_paths(videos_root, video_name, suffix):
    """Gives a list of all frame paths for the given video and frame type.

    :param videos_root: Root directory containing all video frame directories
    :param video_name: The name of the video, or "*" to select all videos under video_root
    :param suffix: The suffix for the frame type. Can be "gt", "mask", or "pred"
    :return: list
    """
    if suffix not in ['gt', 'mask', 'pred']:
        raise ValueError(f'Unsupported suffix {suffix}')

    file_path_pattern = os.path.join(videos_root, video_name, f'frame_*_{suffix}.*')
    matched_files = sorted(glob(file_path_pattern))

    return matched_files


def get_video_names_and_frame_counts(video_frame_root_path, max_num_videos):
    # Determine the videos to evaluate
    video_names = sorted(os.listdir(video_frame_root_path))
    if max_num_videos is not None:
        video_names = video_names[:max_num_videos]
    # Count number of frames per video
    video_frame_counts = [None for _ in video_names]
    for v, video_name in enumerate(video_names):
        gt_frame_list = get_all_frame_paths(video_frame_root_path, video_name, 'gt')
        video_frame_counts[v] = len(gt_frame_list)
    return video_names, video_frame_counts


class Unbuffered(object):
    """https://stackoverflow.com/a/107717"""

    def __init__(self, stream):
        self.stream = stream

    def write(self, data):
        self.stream.write(data)
        self.stream.flush()

    def writelines(self, datas):
        self.stream.writelines(datas)
        self.stream.flush()

    def __getattr__(self, attr):
        return getattr(self.stream, attr)


def str_no_class_fmt(data_type):
    """Prints the name of the given class without extra formatting characters. For example, `str_no_class_fmt(int)` will
    yield "int" instead of the usual "<class 'int'>" produced by the default `str()` function.

    :param data_type: A class (e.g., one returned by `type()`)
    :return: str
    """
    if not isclass(data_type):
        raise ValueError(f'Data type {data_type} is not a class')
    m = re.match('<class \'(.*)\'>', str(data_type))
    return m.group(1)


def pr_to_ipr_info(precision, recall):
    """Produces interpolated precision-recall information (interp. precision values and AUC) from precision

    This function returns the interpolated precision, obtained as ipr(r_0) = max(pr(r)) for r > r_0, as well as the
    AUC under the interpolated precision-recall plot. For more information, see Everingham et al.

    Everingham et al. The PASCAL Visual Object Classes (VOC) Challenge. IJCV 2010.
    http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.157.5766&rep=rep1&type=pdf

    :param recall: Descending list of recall values from `precision_recall_curve` (N float NumPy array)
    :param precision: Ascending list of precision values from `precision_recall_curve` (N float NumPy array)
    :return: The interpolated precision values (N float NumPy array) and the associated AUC (float)
    """
    # For each precision value, replace it with the highest precision value among equal or higher recall values
    i_precision = np.array([np.max(precision[:i + 1]) for i in range(len(precision))])
    # Compute AUC (trapezoidal rule) for "interpolated" precision-recall curve
    ipr_auc = auc(recall, i_precision)

    return i_precision, ipr_auc


def data_to_dataframe(data):
    """Generates a DataFrame whose columns contain the given data.

    :param data: Dictionary mapping from column names (str) to float data (NumPy array, list, or float)
    :return: DataFrame
    """
    assert isinstance(data, dict), 'data is not a dict'
    for v in data.values():
        assert isinstance(v, np.ndarray) or isinstance(v, list) or isinstance(v, float), \
            f'Item type {type(v)} is not supported'

    # Get maximum length of inputs
    max_num_rows = 1
    for v in data.values():
        if type(v) in [np.ndarray, list]:
            max_num_rows = max(max_num_rows, len(v))

    # Initialize empty data
    data_table_np = np.nan * np.ones((max_num_rows, len(data)))
    # Populate each column with the corresponding data values
    for i, (k, v) in enumerate(data.items()):
        if type(v) in [np.ndarray, list]:
            data_table_np[:len(v), i] = v
        else:
            data_table_np[0, i] = v
    # Convert the NumPy data table to a PANDAS data frame
    data_table_df = pd.DataFrame(data_table_np, columns=data.keys())

    return data_table_df


def parse_datetime(datetime_str):
    """Converts the given datetime string to a timestamp using the `date` shell utility.

    :param datetime_str: The string to convert. Must be understood by the `date` shell utility.
    :return: The timestamp corresponding to the given datetime string
    """
    cmd = f'date -d \'{datetime_str}\' \'+%s\''
    try:
        cmd_out = check_output(cmd, shell=True)
    except CalledProcessError:
        raise ValueError(f'Failed to parse datetime {datetime_str}')
    timestamp = int(cmd_out.strip().decode())

    return timestamp


class ThreadPool(object):
    @staticmethod
    def run_job(queue, work_fn):
        while True:
            args, kwargs = queue.get()
            try:
                work_fn(*args, **kwargs)
            except Exception:
                traceback.print_stack()
                return
            queue.task_done()


    def __init__(self, work_fn, num_workers, maxsize=0):
        self.queue = Queue(maxsize)
        self.threads = [Thread(target=self.run_job, daemon=True,
                               args=(self.queue, work_fn)) for _ in range(num_workers)]
        for thread in self.threads:
            thread.start()


    def add_job(self, *args, **kwargs):
        self.queue.put((args, kwargs))


    def join(self):
        """Closes the queue and waits for all jobs to finish."""
        self.queue.join()


def extract_tar_to_path(dest, *args, **kwargs):
    """Extracts the contents of the given file to the given destination.

    :param dest: The folder where the contents should be extracted
    :param args: Arguments to pass to tarfile.open
    :param kwargs: Keyword arguments to pass to tarfile.open
    """
    old_cwd = os.getcwd()

    makedirs(dest)
    os.chdir(dest)

    try:
        with tarfile.open(*args, **kwargs) as f:
            f.extractall()
    except Exception:
        raise RuntimeError(f'Failed to extract specified file to {dest}')
    finally:
        os.chdir(old_cwd)


def download_url(url, dest_file_obj, n_chunk=1):
    """Download a remote file given its URL while showing a progress bar.

    Adapted from https://stackoverflow.com/a/37573701

    :param url: The URL to download the file from
    :param dest_file_obj: File handler to write to (e.g., from open())
    :param n_chunk: How many 1024-byte chunks to retrieve per response iteration
    """
    block_size = n_chunk * 1024  # in bytes
    response = requests.get(url, stream=True)
    file_size = int(response.headers.get('content-length', 0))  # in bytes

    progress_bar = tqdm(total=file_size, unit='iB', unit_scale=True)
    for data in response.iter_content(block_size):
        progress_bar.update(len(data))
        dest_file_obj.write(data)
    progress_bar.close()

    # Write any buffered data so subsequent code can operate on the complete downloaded file
    dest_file_obj.flush()


def equalize_list_lengths(*lists):
    """Returns trimmed list copies such that they all have the same length (i.e., that of the shortest list).

    :param lists: The lists to trim
    :return: zip object
    """
    tuples = [tup for tup in zip(*lists)]
    return zip(*tuples)
