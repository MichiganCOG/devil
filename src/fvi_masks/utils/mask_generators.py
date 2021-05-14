import numpy as np
from PIL import Image, ImageDraw


def get_video_masks_by_moving_random_stroke(**kwargs):
    return [x for x in get_video_masks_by_moving_random_stroke_iterator(**kwargs)]


def get_video_masks_by_moving_random_stroke_iterator(
        video_len, imageWidth=320, imageHeight=180, nStroke=5, nVertexBound=(10, 30), headSpeedBound=(0, 15),
        headAccelerationBound=(0, 15), maxHeadAccelerationAngle=0.5, brushWidthBound=(5, 20), borderGap=None,
        initBorderGap=None, nAccelPointRatio=0.5, maxPointAccel=10, initSpeedBound=(0, 5), centroidAccelDist=None,
        centroidAccelSlope=None, globalAccelRatio=0.0, globalAccelBound=None, rng=np.random.RandomState()
):
    '''
    Get video masks by random strokes which move randomly between each
    frame, including the whole stroke and its control points

    Parameters
    ----------
        video_len: Number of frames in the video
        imageWidth: Image width
        imageHeight: Image height
        nStroke: Number of drawn lines
        nVertexBound: Lower/upper bound of number of control points for each line
        headSpeedBound: Min and max head speed when creating control points
        headAccelerationBound: Min and max acceleration applying on the current head point
            (a head point and its velocity decides the next point)
        maxHeadAccelerationAngle: The maximum magnitude of the head acceleration vector
        brushWidthBound (min, max): Bound of width for each stroke
        borderGap: The minimum gap between image border and drawn lines (width, height)
        initBorderGap: The minimum gap between image and border and drawn lines in the first frame (width, height)
        nAccelPointRatio: The ratio of control points to accelerate per frame
        maxPointAccel: The maximum magnitude of acceleration applied to control points
        initSpeedBound: The min and max initial speed of any random stroke
        centroidAccelDist: The distance from the stroke's centroid at which control points will accelerate toward the
            center
        centroidAccelSlope: The strength of the control point's acceleration toward the center
        globalAccelRatio: How often to apply a global acceleration to each point group
        globalAccelBound: The magnitude limits for global acceleration
        rng: A np.random.RandomState used to generate random values

    Examples
    ----------
        object_like_setting = {
            "nVertexBound": [5, 30],
            "headSpeedBound": (0, 15),
            "headAccelerationBound": (-5, 5),
            "maxHeadAccelerationAngle": 1.5,
            "brushWidthBound": (20, 50),
            "nAccelPointRatio": 0.5,
            "maxPointAccel": 10,
            "borderGap": None,
            "initSpeedBound": (-5, 5)
        }
        rand_curve_setting = {
            "nVertexBound": [10, 30],
            "headSpeedBound": (0, 20),
            "headAccelerationBound": (-7.5, 7.5),
            "maxHeadAccelerationAngle": 0.5,
            "brushWidthBound": (3, 10),
            "nAccelPointRatio": 0.5,
            "maxPointAccel": 3,
            "borderGap": None,
            "initSpeedBound": (-3, 3)
        }
        get_video_masks_by_moving_random_stroke(video_len=5, nStroke=3, **object_like_setting)
    '''
    assert(video_len >= 1)

    # Initialize a set of control points to draw the first mask
    mask = Image.new(mode='1', size=(imageWidth, imageHeight), color=1)
    control_points_set = []
    for i in range(nStroke):
        brushWidth = rng.random_integers(brushWidthBound[0], brushWidthBound[1])
        Xs, Ys, velocities = get_random_stroke_control_points(
            imageWidth=imageWidth, imageHeight=imageHeight,
            nVertexBound=nVertexBound, headSpeedBound=headSpeedBound,
            headAccelerationBound=headAccelerationBound, maxHeadAccelerationAngle=maxHeadAccelerationAngle,
            initBorderGap=initBorderGap, initSpeedBound=initSpeedBound, rng=rng
        )
        control_points_set.append((Xs, Ys, velocities, brushWidth))
        draw_mask_by_control_points(mask, Xs, Ys, brushWidth, fill=0)
    yield mask

    # Generate the following masks by randomly move strokes and their control points
    for i in range(video_len - 1):
        mask = Image.new(mode='1', size=(imageWidth, imageHeight), color=1)
        for j in range(len(control_points_set)):
            Xs, Ys, velocities, brushWidth = control_points_set[j]
            new_Xs, new_Ys, new_velocities = random_move_control_points(Xs, Ys, velocities, nAccelPointRatio,
                                                                        maxPointAccel, imageWidth, imageHeight,
                                                                        borderGap, centroidAccelDist,
                                                                        centroidAccelSlope, globalAccelRatio,
                                                                        globalAccelBound, rng)
            control_points_set[j] = (new_Xs, new_Ys, new_velocities, brushWidth)
        for Xs, Ys, velocity, brushWidth in control_points_set:
            draw_mask_by_control_points(mask, Xs, Ys, brushWidth, fill=0)
        yield mask


def update_position_constrained(cur_value, delta, lower_bound, upper_bound):
    next_value = cur_value + delta
    flip_velocity = False

    while (lower_bound is not None and next_value < lower_bound) \
            or (upper_bound is not None and next_value > upper_bound):
        flip_velocity = not flip_velocity
        if lower_bound is not None and next_value < lower_bound:
            next_value = lower_bound + (lower_bound - next_value)
        else:
            next_value = upper_bound - (next_value - upper_bound)

    return next_value, flip_velocity


def flip_angle_y(angle):
    return -angle


def flip_angle_x(angle):
    return np.pi - angle


def get_random_vector(magnitude_bound, angle_bound, dist, rng):
    if dist == 'uniform':
        magnitude = rng.uniform(*magnitude_bound)
        angle = rng.uniform(*angle_bound)
    elif dist == 'gaussian':
        magnitude = rng.normal(
            (magnitude_bound[0] + magnitude_bound[1]) / 2,
            (magnitude_bound[1] - magnitude_bound[0]) / 2
        )
        angle = rng.normal(
            (angle_bound[0] + angle_bound[1]) / 2,
            (angle_bound[1] - angle_bound[0]) / 2
        )
    else:
        raise NotImplementedError(f'Distribution type {dist} is not supported.')

    return magnitude, angle


def add_polar_vectors(v1, v2):
    mag1, angle1 = v1
    mag2, angle2 = v2

    v1_x = mag1 * np.cos(angle1)
    v1_y = mag1 * np.sin(angle1)
    v2_x = mag2 * np.cos(angle2)
    v2_y = mag2 * np.sin(angle2)

    v3_x = v1_x + v2_x
    v3_y = v1_y + v2_y

    mag3 = np.sqrt(v3_x**2 + v3_y**2)
    angle3 = (np.arctan2(v3_y, v3_x) + 2 * np.pi) % (2 * np.pi)

    return mag3, angle3


def random_move_control_points(Xs, Ys, lineVelocities, nAccelPointRatio, maxPointAccel, imageWidth, imageHeight,
                               borderGap, centroidAccelDist, centroidAccelSlope, globalAccelRatio, globalAccelBound,
                               rng):
    new_Xs = Xs.copy()
    new_Ys = Ys.copy()
    new_velocities = lineVelocities.copy()

    for i, (X, Y, velocity) in enumerate(zip(Xs, Ys, lineVelocities)):
        speed, angle = velocity
        new_X, flip_velocity_X = update_position_constrained(
            X, int(speed * np.cos(angle)),
            lower_bound=None if borderGap is None else borderGap[0],
            upper_bound=None if borderGap is None else imageWidth - borderGap[0]
        )
        new_Y, flip_velocity_Y = update_position_constrained(
            Y, int(speed * np.sin(angle)),
            lower_bound=None if borderGap is None else borderGap[1],
            upper_bound=None if borderGap is None else imageHeight - borderGap[1]
        )
        if flip_velocity_X:
            angle = flip_angle_x(angle)
        if flip_velocity_Y:
            angle = flip_angle_y(angle)

        if rng.uniform() < nAccelPointRatio:
            acceleration = get_random_vector((0, maxPointAccel), (-np.pi, np.pi), 'uniform', rng)
            speed, angle = add_polar_vectors((speed, angle), acceleration)

        new_Xs[i] = new_X
        new_Ys[i] = new_Y
        new_velocities[i] = [speed, angle]

    # Randomly accelerate the entire blob by the same force
    if rng.uniform() < globalAccelRatio:
        acceleration = get_random_vector(globalAccelBound, (-np.pi, np.pi), 'uniform', rng)
        for i in range(len(new_velocities)):
            new_velocities[i] = add_polar_vectors(new_velocities[i], acceleration)

    # If the points are too far from the centroid, apply forces to bring them to the center
    if centroidAccelDist is not None and centroidAccelSlope is not None:
        for i in range(len(new_velocities)):
            c_dist_x = new_Xs[i] - new_Xs.mean()
            c_dist_y = new_Ys[i] - new_Ys.mean()
            c_dist = np.sqrt(c_dist_x ** 2 + c_dist_y ** 2)
            angle = flip_angle_y(flip_angle_x(np.arctan2(c_dist_y, c_dist_x)))
            c_accel_mag = max(0, centroidAccelSlope * (c_dist - centroidAccelDist))
            new_velocities[i] = add_polar_vectors(new_velocities[i], (c_accel_mag, angle))

    return new_Xs, new_Ys, new_velocities


def get_random_stroke_control_points(imageWidth, imageHeight, nVertexBound, headSpeedBound, headAccelerationBound,
                                     maxHeadAccelerationAngle, initBorderGap, initSpeedBound, rng):
    '''
    Implementation the free-form training masks generating algorithm
    proposed by JIAHUI YU et al. in "Free-Form Image Inpainting with Gated Convolution"
    '''
    X = rng.randint(imageWidth)
    Y = rng.randint(imageHeight)
    if initBorderGap is not None:
        X = np.clip(X, initBorderGap[0], imageWidth - initBorderGap[0])
        Y = np.clip(Y, initBorderGap[1], imageHeight - initBorderGap[1])
    Xs = [X]
    Ys = [Y]

    numVertex = rng.random_integers(nVertexBound[0], nVertexBound[1])

    speed = rng.uniform(*headSpeedBound)
    angle = rng.uniform(0, 2 * np.pi)

    for i in range(numVertex-1):
        # Add acceleration relative to the current head velocity
        a_mag, a_angle = get_random_vector(headAccelerationBound, (0, maxHeadAccelerationAngle), 'uniform', rng)
        speed, angle = add_polar_vectors((speed, angle), (a_mag, a_angle + angle))
        speed = np.clip(speed, *headSpeedBound)

        X, flip_velocity_X = update_position_constrained(
            X, speed * np.cos(angle),
            lower_bound=None if initBorderGap is None else initBorderGap[0],
            upper_bound=None if initBorderGap is None else imageWidth - initBorderGap[0]
        )
        Y, flip_velocity_Y = update_position_constrained(
            Y, speed * np.sin(angle),
            lower_bound=None if initBorderGap is None else initBorderGap[1],
            upper_bound=None if initBorderGap is None else imageHeight - initBorderGap[1]
        )
        if flip_velocity_X:
            angle = flip_angle_x(angle)
        if flip_velocity_Y:
            angle = flip_angle_y(angle)

        Xs.append(X)
        Ys.append(Y)

    velocity = get_random_vector(initSpeedBound, (-np.pi, np.pi), 'uniform', rng=rng)
    velocities = [velocity for _ in range(numVertex)]

    return np.array(Xs), np.array(Ys), np.array(velocities)


def draw_mask_by_control_points(mask, Xs, Ys, brushWidth, fill=255):
    radius = brushWidth // 2 - 1
    for i in range(1, len(Xs)):
        draw = ImageDraw.Draw(mask)
        startX, startY = Xs[i - 1], Ys[i - 1]
        nextX, nextY = Xs[i], Ys[i]
        draw.line((startX, startY) + (nextX, nextY), fill=fill, width=brushWidth)
    for x, y in zip(Xs, Ys):
        draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=fill)
    return mask


# # modified from https://github.com/naoto0804/pytorch-inpainting-with-partial-conv/blob/master/generate_data.py
# def get_random_walk_mask(imageWidth=320, imageHeight=180, length=None):
#     action_list = [[0, 1], [0, -1], [1, 0], [-1, 0]]
#     canvas = np.zeros((imageHeight, imageWidth)).astype("i")
#     if length is None:
#         length = imageWidth * imageHeight
#     x = random.randint(0, imageHeight - 1)
#     y = random.randint(0, imageWidth - 1)
#     x_list = []
#     y_list = []
#     for i in range(length):
#         r = random.randint(0, len(action_list) - 1)
#         x = np.clip(x + action_list[r][0], a_min=0, a_max=imageHeight - 1)
#         y = np.clip(y + action_list[r][1], a_min=0, a_max=imageWidth - 1)
#         x_list.append(x)
#         y_list.append(y)
#     canvas[np.array(x_list), np.array(y_list)] = 1
#     return Image.fromarray(canvas * 255).convert('1')


def get_masked_ratio(mask):
    """
    Calculate the masked ratio.
    mask: Expected a binary PIL image, where 0 and 1 represent
          masked(invalid) and valid pixel values.
    """
    hist = mask.histogram()
    return hist[0] / np.prod(mask.size)
