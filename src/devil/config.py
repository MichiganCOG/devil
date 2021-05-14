import json
from types import SimpleNamespace


# These objects defines the available configuration parameters, their expected types, and their default values. They
# should be trees of nested dictionaries whose leaves are 2-tuples containing the data type and the default value.
_NORMALIZED_IMAGE_NORM_SCORER_SCHEMA = {
    'degree': (int, 1)
}

_FEATURE_MATCHER_COMPUTER_ARGS = {
    'feature_type': (str, 'surf'),
    'max_num_features': (int, 750),
}

_AFFINE_TRANSFORM_COMPUTER_SCHEMA = {
    'feature_matcher_computer_args': _FEATURE_MATCHER_COMPUTER_ARGS,
    'ransac_threshold': (float, 5.0)
}

_AFFINE_SCORE_SCORER_SCHEMA = {
    'affine_transform_computer_args': _AFFINE_TRANSFORM_COMPUTER_SCHEMA,
    'normalized_image_norm_scorer_args': _NORMALIZED_IMAGE_NORM_SCORER_SCHEMA
}

_AFFINE_FLOW_SCORER_SCHEMA = {
    'affine_transform_computer_args': _AFFINE_TRANSFORM_COMPUTER_SCHEMA
}

_LOW_CAMERA_MOTION_SEGMENTS_COMPUTER_SCHEMA = {
    'threshold': (float, 375.0),
    'min_length': (int, 12),
    'affine_score_scorer_args': _AFFINE_SCORE_SCORER_SCHEMA
}

_CONFIG_SCHEMA = {
    'ScoreDavisVideos': {
        # Path to the DAVIS dataset (should contain "Annotations" and "JPEGImages" folders)
        'davis_root_path': (str, '/z/dat/DAVIS'),
        # Path to the list of names of DAVIS videos to use
        'video_list_path': (str, None),
        # The maximum number of videos to process
        'max_num_videos': (int, None),
        # Path to the output file
        'output_path': (str, 'scores/scores.tsv'),
        # Scores to output
        'output_scores': (list, ['camera_motion_score',
                                 'fg_motion_score',
                                 'bg_scene_motion_score',
                                 'fg_displacement_score',
                                 'fg_size_score']),

        'CameraMotionScorer': {
            'scorer_class_name': (str, 'CameraMotionFrameDisplacementScorer'),
            'scorer_args': {
                'num_sampled_frames': (int, 10),
                'percentile': (float, 96.0),
                'affine_transform_computer_args': _AFFINE_TRANSFORM_COMPUTER_SCHEMA
            }
        },
        'ForegroundMotionScorer': {
            'scorer_class_name': (str, 'ForegroundMotionIoUScorer'),
            'scorer_args': {
                'num_sampled_frames': (int, 10),
                'percentile': (float, 90.0),
                'mask_percentage_threshold': (float, 3e-5),
                'affine_transform_computer_args': _AFFINE_TRANSFORM_COMPUTER_SCHEMA
            }
        },
        'BackgroundSceneMotionScorer': {
            'scorer_class_name': (str, 'BackgroundSceneMotionFlowScorer'),
            'scorer_args': {
                'align_threshold': (float, 375.0),
                'min_segment_length': (int, 12),
                'affine_flow_scorer_args': _AFFINE_FLOW_SCORER_SCHEMA,
                'low_camera_motion_segments_computer_args': _LOW_CAMERA_MOTION_SEGMENTS_COMPUTER_SCHEMA
            }
        },
    },
    'ScoreFlickrClips': {
        # Path to the Flickr clips dataset
        'dataset_root_path': (str, 'datasets/flickr-raw-clips'),
        # Path to the list of names of videos to use
        'video_list_path': (str, None),
        # The maximum number of videos to process
        'max_num_videos': (int, None),
        # Path to the output file
        'output_path': (str, 'scores/flickr-raw-clips/scores.tsv'),
        # The resolution at which to process videos
        'proc_resolution': (list, [854, 480]),
        # Scores to output
        'output_scores': (list, [
            'camera_motion_score',
            'fg_size_score',
            'cut_score',
            'bg_scene_motion_score',
        ]),
        'CameraMotionScorer': {
            'scorer_class_name': (str, 'CameraMotionFrameDisplacementScorer'),
            'scorer_args': {
                'num_sampled_frames': (int, 10),
                'percentile': (float, 96.0),
                'affine_transform_computer_args': _AFFINE_TRANSFORM_COMPUTER_SCHEMA
            }
        },
        'BackgroundSceneMotionScorer': {
            'scorer_class_name': (str, 'BackgroundSceneMotionFlowScorer'),
            'scorer_args': {
                'align_threshold': (float, 375.0),
                'min_segment_length': (int, 12),
                'affine_flow_scorer_args': _AFFINE_FLOW_SCORER_SCHEMA,
                'low_camera_motion_segments_computer_args': _LOW_CAMERA_MOTION_SEGMENTS_COMPUTER_SCHEMA
            },
        },
    },
    'NormalizedImageNormScorer': _NORMALIZED_IMAGE_NORM_SCORER_SCHEMA,
    'AffineTransformComputer': {
        'feature_matcher_computer_args': {
            'feature_type': (str, 'orb'),
            'max_num_features': (int, 750),
        },
        'ransac_threshold': (float, 5.0)
    },
    'AffineFlowScorer': _AFFINE_FLOW_SCORER_SCHEMA,
    'LowCameraMotionSegmentsComputer': _LOW_CAMERA_MOTION_SEGMENTS_COMPUTER_SCHEMA,
    'AffineScoreScorer': {
        'affine_transform_computer_args': {
            'feature_matcher_computer_args': {
                'feature_type': (str, 'orb'),
                'max_num_features': (int, 750),
            },
            'ransac_threshold': (float, 5.0)
        },
        'normalized_image_norm_scorer_args': {
            'degree': (int, 1)
        }
    }
}


def _verify_schema(schema):
    """Checks that the given schema is a tree consisting of non-leaf dictionaries and leaf tuples.

    This function checks that the input is a tree where non-leaf nodes are represented as dictionaries, and leaf nodes
    are represented as 2-tuples. An AssertionError is thrown if this is not the case.

    :param schema: A nested dictionary (dict)
    """
    assert type(schema) in [dict, tuple], f'Expected a dict or a tuple but got {type(schema)}'
    if isinstance(schema, tuple):
        assert len(schema) == 2, f'Expected a tuple with length 2 but got length {len(schema)}'
        if schema[1] is not None:
            assert isinstance(schema[1], schema[0]), f'{str(schema[1])} does not have expected type {str(schema)}'
    elif isinstance(schema, dict):
        for sub_schema in schema.values():
            _verify_schema(sub_schema)


def dict_to_namespace(d):
    """Converts a dictionary object into a SimpleNamespace object.

    :param d: The dictionary to convert (dict)
    :return: SimpleNamespace
    """
    # Convert the dictionary to a JSON string
    json_str = json.dumps(d)
    # Convert the JSON string, but generate namespaces instead of dictionaries
    namespace = json.loads(json_str, object_hook=lambda x: SimpleNamespace(**x))

    return namespace


def namespace_to_dict(n):
    """Converts a SimpleNamespace object into a dict.

    :param n: The namespace object to convert (SimpleNamespace)
    :return: dict
    """
    assert isinstance(n, SimpleNamespace)
    return _namespace_to_dict_util(n)


def _namespace_to_dict_util(n):
    """Recursive function called by namespace_to_dict."""
    if not isinstance(n, SimpleNamespace):
        return n

    ret = {}
    for k, v in vars(n).items():
        ret[k] = _namespace_to_dict_util(v)

    return ret


def overwrite_dict(dict_base, dict_new, base_path=None):
    """Replace elements in a nested base dictionary with corresponding elements from a new dictionary.

    WARNING: This function mutates the base dictionary.

    :param dict_base: The base dictionary whose entries should be replaced (dict)
    :param dict_new: The new dictionary containing entries for replacement (dict)
    :param base_path: A namespace-style path of keys to the dictionary from the root (str)
    """
    assert isinstance(dict_new, dict)
    for k in dict_new:
        # Add the current key to the path
        k_path = str(k) if base_path is None else f'{base_path}.{str(k)}'
        # Make sure that the key in the new dictionary matches one from the base dictionary
        assert k in dict_base, f'Could not find path {k_path} in the base dictionary'
        # Check that the types match between the base dictionary entry and the new one
        if dict_base[k] is not None:
            assert isinstance(type(dict_base[k]), type(dict_new[k])), \
                'The types at {} in the base dictionary do not match (expected {}, got {})'.format(
                    k_path, str(type(dict_base[k])), str(type(dict_new[k])))
        # Recursively replace dictionary entries
        if isinstance(dict_base[k], dict):
            overwrite_dict(dict_base[k], dict_new[k], k_path)
        else:
            # Simply copy over leaf entries
            dict_base[k] = dict_new[k]


def get_default_config():
    """Returns a copy of the default configuration as a SimpleNamespace."""
    return _config_schema_to_namespace(_CONFIG_SCHEMA)


def overwrite_config(base_config, config_pairs_str_list, prefix_key=None):
    """Replaces values in the current configuration as specified by a list of string arguments.

    WARNING: This function mutates the base configuration.

    :param base_config: The configuration to modify (SimpleNamespace)
    :param config_pairs_str_list: A list of string pairs that correspond to the namespace path of the key to replace,
                                  and the value to use during replacement (list of str)
    :param prefix_key: The key that indexes into a deeper layer of the global configuration schema (str)
    """
    # Check that the configuration pair list can actually be divided into pairs
    assert len(config_pairs_str_list) % 2 == 0, 'Odd number of arguments specified (must be even)'

    flat_config_schema_info = {}
    for namespace_path, data_type, default_value in get_config_schema_info(prefix_key=prefix_key):
        flat_config_schema_info[namespace_path] = (data_type, default_value)

    # Process each key-value pair in the list
    for i in range(0, len(config_pairs_str_list), 2):
        config_key = config_pairs_str_list[i]
        config_value_str = config_pairs_str_list[i + 1]

        # Check that the key is inside the config
        assert config_key in flat_config_schema_info, f'The string "{config_key}" is not a valid configuration key'

        # Try to parse the config value into a simple object type
        try:
            config_value =  eval(config_value_str, {'__builtins__': None})
        except TypeError:
            raise ValueError('Failed to parse "{}" (for key "{}") into a simple object type. Try surrounding this '
                             'expression in quotes (escape quotes by \\ if it was entered via the command line).'
                             .format(config_value_str, config_key))
        # Check that the parsed value type matches the expected type
        expected_type = flat_config_schema_info[config_key][0]
        assert isinstance(config_value, expected_type), \
            'The parse of "{}" did not match the expected type {}'.format(config_value_str, expected_type)

        # Assign the config value to the key
        config_key_parts = config_key.split('.')
        cur_subconfig = base_config
        for part in config_key_parts[:-1]:
            cur_subconfig = getattr(cur_subconfig, part)
        setattr(cur_subconfig, config_key_parts[-1], config_value)


def get_nested_dict_entry_from_namespace_path(d, namespace_path):
    """Obtain the entry from a nested dictionary that corresponds to the path given in namespace format.

    A namespace path specifies the keys of the nested dictionary as a dot-separated string (e.g., "key1.key2.key3").

    :param d: The nested dictionary to traverse (dict)
    :param namespace_path: A dot-separated string containing the keys used to traverse the dictionary (str)
    :return: object
    """
    # Try to split off the namespace path into the first key and the rest of the keys
    split_namespace_path = namespace_path.split('.', 1)
    if len(split_namespace_path) == 1:
        # Only one key for a non-nested dict; return the result
        return d[split_namespace_path[0]]
    else:
        cur_key, path_remainder = split_namespace_path
        return get_nested_dict_entry_from_namespace_path(d[cur_key], path_remainder)


def _config_schema_to_namespace(schema):
    """Translates the given schema dict into a nested SimpleNamespace object with default parameters."""
    if isinstance(schema, tuple):
        return schema[1]
    else:
        ret = SimpleNamespace()
        for k, v in schema.items():
            setattr(ret, k, _config_schema_to_namespace(v))
        return ret


def get_config_schema_info(prefix_key=None):
    info = []
    _get_config_setting_info_util(_CONFIG_SCHEMA[prefix_key] if prefix_key else _CONFIG_SCHEMA, '', info)
    # Sort results by namespace path
    processed_paths = sorted(info, key=lambda x: x[0])

    return processed_paths


def _get_config_setting_info_util(cur_item, cur_item_path, cur_namespaces):
    if isinstance(cur_item, tuple):
        # Remove leading period from item path; add path, data type, and default value
        cur_namespaces.append((cur_item_path[1:], *cur_item))
    else:
        for k, v in cur_item.items():
            _get_config_setting_info_util(v, f'{cur_item_path}.{k}', cur_namespaces)


# Initialize this module
_verify_schema(_CONFIG_SCHEMA)
