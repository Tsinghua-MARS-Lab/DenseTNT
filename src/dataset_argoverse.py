import copy
import math
import multiprocessing
import os
import pickle
import random
import zlib
from collections import defaultdict
from multiprocessing import Process
from random import choice
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

import utils_cython
import utils
from utils import get_name, get_file_name_int, get_angle, logging, rotate, round_value, get_pad_vector, get_dis, get_subdivide_polygons
from utils import get_points_remove_repeated, get_one_subdivide_polygon, get_dis_point_2_polygons, larger, equal, assert_
from utils import get_neighbour_points, get_subdivide_points, get_unit_vector, get_dis_point_2_points

TIMESTAMP = 0
TRACK_ID = 1
OBJECT_TYPE = 2
X = 3
Y = 4
CITY_NAME = 5

type2index = {}
type2index["OTHERS"] = 0
type2index["AGENT"] = 1
type2index["AV"] = 2

max_vector_num = 0

VECTOR_PRE_X = 0
VECTOR_PRE_Y = 1
VECTOR_X = 2
VECTOR_Y = 3


def get_sub_map(args: utils.Args, x, y, city_name, vectors=[], polyline_spans=[], mapping=None):
    """
    Calculate lanes which are close to (x, y) on map.

    Only take lanes which are no more than args.max_distance away from (x, y).

    """

    if args.not_use_api:
        pass
    else:
        from argoverse.map_representation.map_api import ArgoverseMap
        assert isinstance(am, ArgoverseMap)
        # Add more lane attributes, such as 'has_traffic_control', 'is_intersection' etc.
        if 'semantic_lane' in args.other_params:
            lane_ids = am.get_lane_ids_in_xy_bbox(x, y, city_name, query_search_range_manhattan=args.max_distance)
            local_lane_centerlines = [am.get_lane_segment_centerline(lane_id, city_name) for lane_id in lane_ids]
            polygons = local_lane_centerlines

            if args.visualize:
                angle = mapping['angle']
                vis_lanes = [am.get_lane_segment_polygon(lane_id, city_name)[:, :2] for lane_id in lane_ids]
                t = []
                for each in vis_lanes:
                    for point in each:
                        point[0], point[1] = rotate(point[0] - x, point[1] - y, angle)
                    num = len(each) // 2
                    t.append(each[:num].copy())
                    t.append(each[num:num * 2].copy())
                vis_lanes = t
                mapping['vis_lanes'] = vis_lanes
        else:
            polygons = am.find_local_lane_centerlines(x, y, city_name,
                                                      query_search_range_manhattan=args.max_distance)
        polygons = [polygon[:, :2].copy() for polygon in polygons]
        angle = mapping['angle']
        for index_polygon, polygon in enumerate(polygons):
            for i, point in enumerate(polygon):
                point[0], point[1] = rotate(point[0] - x, point[1] - y, angle)
                if 'scale' in mapping:
                    assert 'enhance_rep_4' in args.other_params
                    scale = mapping['scale']
                    point[0] *= scale
                    point[1] *= scale

        def dis_2(point):
            return point[0] * point[0] + point[1] * point[1]

        def get_dis(point_a, point_b):
            return np.sqrt((point_a[0] - point_b[0]) ** 2 + (point_a[1] - point_b[1]) ** 2)

        def get_dis_for_points(point, polygon):
            dis = np.min(np.square(polygon[:, 0] - point[0]) + np.square(polygon[:, 1] - point[1]))
            return np.sqrt(dis)

        def ok_dis_between_points(points, points_, limit):
            dis = np.inf
            for point in points:
                dis = np.fmin(dis, get_dis_for_points(point, points_))
                if dis < limit:
                    return True
            return False

        def get_hash(point):
            return round((point[0] + 500) * 100) * 1000000 + round((point[1] + 500) * 100)

        lane_idx_2_polygon_idx = {}
        for polygon_idx, lane_idx in enumerate(lane_ids):
            lane_idx_2_polygon_idx[lane_idx] = polygon_idx

        # There is a lane scoring module (see Section 3.2) in the paper in order to reduce the number of goal candidates.
        # In this implementation, we use goal scoring instead of lane scoring, because we observed that it performs slightly better than lane scoring.
        # Here we only sample sparse goals, and dense goal sampling is performed after goal scoring (see decoder).
        if 'goals_2D' in args.other_params:
            points = []
            visit = {}
            point_idx_2_unit_vector = []

            mapping['polygons'] = polygons

            for index_polygon, polygon in enumerate(polygons):
                for i, point in enumerate(polygon):
                    hash = get_hash(point)
                    if hash not in visit:
                        visit[hash] = True
                        points.append(point)

                # Subdivide lanes to get more fine-grained 2D goals.
                if 'subdivide' in args.other_params:
                    subdivide_points = get_subdivide_points(polygon)
                    points.extend(subdivide_points)
                    subdivide_points = get_subdivide_points(polygon, include_self=True)

            mapping['goals_2D'] = np.array(points)

        for index_polygon, polygon in enumerate(polygons):
            assert_(2 <= len(polygon) <= 10, info=len(polygon))
            # assert len(polygon) % 2 == 1

            # if args.visualize:
            #     traj = np.zeros((len(polygon), 2))
            #     for i, point in enumerate(polygon):
            #         traj[i, 0], traj[i, 1] = point[0], point[1]
            #     mapping['trajs'].append(traj)

            start = len(vectors)
            if 'semantic_lane' in args.other_params:
                assert len(lane_ids) == len(polygons)
                lane_id = lane_ids[index_polygon]
                lane_segment = am.city_lane_centerlines_dict[city_name][lane_id]
            assert_(len(polygon) >= 2)
            for i, point in enumerate(polygon):
                if i > 0:
                    vector = [0] * args.hidden_size
                    vector[-1 - VECTOR_PRE_X], vector[-1 - VECTOR_PRE_Y] = point_pre[0], point_pre[1]
                    vector[-1 - VECTOR_X], vector[-1 - VECTOR_Y] = point[0], point[1]
                    vector[-5] = 1
                    vector[-6] = i

                    vector[-7] = len(polyline_spans)

                    if 'semantic_lane' in args.other_params:
                        vector[-8] = 1 if lane_segment.has_traffic_control else -1
                        vector[-9] = 1 if lane_segment.turn_direction == 'RIGHT' else \
                            -1 if lane_segment.turn_direction == 'LEFT' else 0
                        vector[-10] = 1 if lane_segment.is_intersection else -1
                    point_pre_pre = (2 * point_pre[0] - point[0], 2 * point_pre[1] - point[1])
                    if i >= 2:
                        point_pre_pre = polygon[i - 2]
                    vector[-17] = point_pre_pre[0]
                    vector[-18] = point_pre_pre[1]

                    vectors.append(vector)
                point_pre = point

            end = len(vectors)
            if start < end:
                polyline_spans.append([start, end])

    return (vectors, polyline_spans)


def preprocess_map(map_dict):
    """
    Preprocess map to calculate potential polylines.
    """

    for city_name in map_dict:
        ways = map_dict[city_name]['way']
        nodes = map_dict[city_name]['node']
        polylines = []
        polylines_dict = {}
        for way in ways:
            polyline = []
            points = way['nd']
            points = [nodes[int(point['@ref'])] for point in points]
            point_pre = None
            for i, point in enumerate(points):
                if i > 0:
                    vector = [float(point_pre['@x']), float(point_pre['@y']), float(point['@x']), float(point['@y'])]
                    polyline.append(vector)
                point_pre = point

            if len(polyline) > 0:
                index_x = round_value(float(point_pre['@x']))
                index_y = round_value(float(point_pre['@y']))
                if index_x not in polylines_dict:
                    polylines_dict[index_x] = []
                polylines_dict[index_x].append(polyline)
                polylines.append(polyline)

        map_dict[city_name]['polylines'] = polylines
        map_dict[city_name]['polylines_dict'] = polylines_dict


def preprocess(args, id2info, mapping):
    """
    This function calculates matrix based on information from get_instance.
    """
    polyline_spans = []
    keys = list(id2info.keys())
    assert 'AV' in keys
    assert 'AGENT' in keys
    keys.remove('AV')
    keys.remove('AGENT')
    keys = ['AGENT', 'AV'] + keys
    vectors = []
    two_seconds = mapping['two_seconds']
    mapping['trajs'] = []
    mapping['agents'] = []
    for id in keys:
        polyline = {}

        info = id2info[id]
        start = len(vectors)
        if args.no_agents:
            if id != 'AV' and id != 'AGENT':
                break

        agent = []
        for i, line in enumerate(info):
            if larger(line[TIMESTAMP], two_seconds):
                break
            agent.append((line[X], line[Y]))

        if args.visualize:
            traj = np.zeros([args.hidden_size])
            for i, line in enumerate(info):
                if larger(line[TIMESTAMP], two_seconds):
                    traj = traj[:i * 2].copy()
                    break
                traj[i * 2], traj[i * 2 + 1] = line[X], line[Y]
                if i == len(info) - 1:
                    traj = traj[:(i + 1) * 2].copy()
            traj = traj.reshape((-1, 2))
            mapping['trajs'].append(traj)

        for i, line in enumerate(info):
            if larger(line[TIMESTAMP], two_seconds):
                break
            x, y = line[X], line[Y]
            if i > 0:
                # print(x-line_pre[X], y-line_pre[Y])
                vector = [line_pre[X], line_pre[Y], x, y, line[TIMESTAMP], line[OBJECT_TYPE] == 'AV',
                          line[OBJECT_TYPE] == 'AGENT', line[OBJECT_TYPE] == 'OTHERS', len(polyline_spans), i]
                vectors.append(get_pad_vector(vector))
            line_pre = line

        end = len(vectors)
        if end - start == 0:
            assert id != 'AV' and id != 'AGENT'
        else:
            mapping['agents'].append(np.array(agent))

            polyline_spans.append([start, end])

    assert_(len(mapping['agents']) == len(polyline_spans))

    assert len(vectors) <= max_vector_num

    t = len(vectors)
    mapping['map_start_polyline_idx'] = len(polyline_spans)
    if args.use_map:
        vectors, polyline_spans = get_sub_map(args, mapping['cent_x'], mapping['cent_y'], mapping['city_name'],
                                              vectors=vectors,
                                              polyline_spans=polyline_spans, mapping=mapping)

    # logging('len(vectors)', t, len(vectors), prob=0.01)

    matrix = np.array(vectors)
    # matrix = np.array(vectors, dtype=float)
    # del vectors

    # matrix = torch.zeros([len(vectors), args.hidden_size])
    # for i, vector in enumerate(vectors):
    #     for j, each in enumerate(vector):
    #         matrix[i][j].fill_(each)

    labels = []
    info = id2info['AGENT']
    info = info[mapping['agent_pred_index']:]
    if not args.do_test:
        if 'set_predict' in args.other_params:
            pass
        else:
            assert len(info) == 30
    for line in info:
        labels.append(line[X])
        labels.append(line[Y])

    if 'set_predict' in args.other_params:
        if 'test' in args.data_dir[0]:
            labels = [0.0 for _ in range(60)]

    if 'goals_2D' in args.other_params:
        point_label = np.array(labels[-2:])
        mapping['goals_2D_labels'] = np.argmin(get_dis(mapping['goals_2D'], point_label))

        if 'lane_scoring' in args.other_params:
            stage_one_label = 0
            polygons = mapping['polygons']
            min_dis = 10000.0
            for i, polygon in enumerate(polygons):
                temp = np.min(get_dis(polygon, point_label))
                if temp < min_dis:
                    min_dis = temp
                    stage_one_label = i

            mapping['stage_one_label'] = stage_one_label

    mapping.update(dict(
        matrix=matrix,
        labels=np.array(labels).reshape([30, 2]),
        polyline_spans=[slice(each[0], each[1]) for each in polyline_spans],
        labels_is_valid=np.ones(args.future_frame_num, dtype=np.int64),
        eval_time=30,
    ))

    return mapping


def get_mark_type_to_int():
    from av2.map.lane_segment import LaneMarkType
    mark_types = [LaneMarkType.DASH_SOLID_YELLOW,
                  LaneMarkType.DASH_SOLID_WHITE,
                  LaneMarkType.DASHED_WHITE,
                  LaneMarkType.DASHED_YELLOW,
                  LaneMarkType.DOUBLE_SOLID_YELLOW,
                  LaneMarkType.DOUBLE_SOLID_WHITE,
                  LaneMarkType.DOUBLE_DASH_YELLOW,
                  LaneMarkType.DOUBLE_DASH_WHITE,
                  LaneMarkType.SOLID_YELLOW,
                  LaneMarkType.SOLID_WHITE,
                  LaneMarkType.SOLID_DASH_WHITE,
                  LaneMarkType.SOLID_DASH_YELLOW,
                  LaneMarkType.SOLID_BLUE,
                  LaneMarkType.NONE,
                  LaneMarkType.UNKNOWN]
    mark_type_to_int = defaultdict(int)
    for i, each in enumerate(mark_types):
        mark_type_to_int[each] = i + 1
    return mark_type_to_int


def argoverse2_load_scenario(instance_dir):
    from av2.datasets.motion_forecasting import scenario_serialization
    file_path = sorted(Path(instance_dir).glob("*.parquet"))
    assert len(file_path) == 1
    file_path = file_path[0]
    return scenario_serialization.load_argoverse_scenario_parquet(file_path)


def argoverse2_load_map(instance_dir):
    log_map_dirpath = Path(instance_dir)
    from av2.map.map_api import ArgoverseStaticMap
    vector_data_fnames = sorted(log_map_dirpath.glob("log_map_archive_*.json"))
    if not len(vector_data_fnames) == 1:
        raise RuntimeError(f"JSON file containing vector map data is missing (searched in {log_map_dirpath})")
    vector_data_fname = vector_data_fnames[0]
    return ArgoverseStaticMap.from_json(vector_data_fname)


def argoverse2_get_instance(args: utils.Args, instance_dir):
    from av2.datasets.motion_forecasting import data_schema
    from av2.datasets.motion_forecasting.data_schema import ObjectType

    scenario = argoverse2_load_scenario(instance_dir)

    object_type_to_int = defaultdict(int)
    object_type_to_int[ObjectType.VEHICLE] = 1
    object_type_to_int[ObjectType.PEDESTRIAN] = 2
    object_type_to_int[ObjectType.MOTORCYCLIST] = 3
    object_type_to_int[ObjectType.CYCLIST] = 4
    object_type_to_int[ObjectType.BUS] = 5

    mapping = {}
    vectors = []
    polyline_spans = []
    agents = []
    polygons = []
    labels = []
    gt_trajectory_global_coordinates = []

    # find focal track
    if True:
        tracks = []
        focal_track = None
        for track in scenario.tracks:
            if track.category == data_schema.TrackCategory.FOCAL_TRACK:
                assert track.track_id == scenario.focal_track_id
                focal_track = track
            else:
                tracks.append(track)
        assert focal_track is not None
        tracks = [focal_track] + tracks

        len(focal_track.object_states)
        len(tracks[-1].object_states)

    # find current coordinates and labels of focal track
    if True:
        cent_x = None
        cent_y = None
        angle = None
        normalizer = None

        assert len(focal_track.object_states) == 110
        for timestep, state in enumerate(focal_track.object_states):
            assert timestep == state.timestep
            # current timestep
            if state.timestep == 50 - 1:
                cent_x = state.position[0]
                cent_y = state.position[1]
                angle = -state.heading + math.radians(90)
                normalizer = utils.Normalizer(cent_x, cent_y, angle)
            elif state.timestep >= 50:
                labels.append(normalizer((state.position[0], state.position[1])))
                gt_trajectory_global_coordinates.append((state.position[0], state.position[1]))

        assert cent_x is not None
        mapping.update(dict(
            cent_x=cent_x,
            cent_y=cent_y,
            angle=angle,
            normalizer=normalizer,
        ))

    for track in tracks:
        assert isinstance(track, data_schema.Track)
        start = len(vectors)

        agent = []
        timestep_to_state = {}
        for state in track.object_states:
            if state.observed:
                assert state.timestep < 50
                timestep_to_state[state.timestep] = state
                agent.append(normalizer([state.position[0], state.position[1]]))

        i = 0
        while i < 50:
            if i in timestep_to_state:
                state = timestep_to_state[i]

                vector = np.zeros(args.hidden_size)
                assert isinstance(state, data_schema.ObjectState)

                vector[0], vector[1] = normalizer((state.position[0], state.position[1]))
                vector[2], vector[3] = rotate(state.velocity[0], state.velocity[1], angle)
                vector[4] = state.heading + angle
                vector[5] = state.timestep

                vector[10 + object_type_to_int[track.object_type]] = 1

                offset = 20
                for j in range(8):
                    if (i + j) in timestep_to_state:
                        t = timestep_to_state[i + j].position
                        vector[offset + j * 3], vector[offset + j * 3 + 1] = normalizer((t[0], t[1]))
                        vector[offset + j * 3 + 2] = 1

                i += 4
                vectors.append(vector[::-1])
            else:
                i += 1

        end = len(vectors)
        if end > start:
            agents.append(np.array(agent))
            polyline_spans.append([start, end])

    map_start_polyline_idx = len(polyline_spans)

    if args.use_map:
        from av2.map.lane_segment import LaneType, LaneMarkType

        lane_type_to_int = defaultdict(int)
        lane_type_to_int[LaneType.VEHICLE] = 1
        lane_type_to_int[LaneType.BIKE] = 2
        lane_type_to_int[LaneType.BUS] = 3

        mark_type_to_int = get_mark_type_to_int()

        argoverse2_map = argoverse2_load_map(instance_dir)
        for lane_segment in argoverse2_map.vector_lane_segments.values():
            start = len(vectors)

            for waypoints in [lane_segment.left_lane_boundary.waypoints, lane_segment.right_lane_boundary.waypoints]:
                polyline = []
                for point in waypoints:
                    polyline.append(normalizer([point.x, point.y]))
                polyline = np.array(polyline)
                polygons.append(polyline)

                for i in range(len(polyline)):
                    vector = np.zeros(args.hidden_size)
                    vector[0] = lane_segment.is_intersection

                    offset = 10
                    for j in range(5):
                        if i + j < len(polyline):
                            vector[offset + j * 2] = polyline[i + j, 0]
                            vector[offset + j * 2 + 1] = polyline[i + j, 1]

                    vectors.append(vector)

                    offset = 30
                    vector[offset + mark_type_to_int[lane_segment.left_mark_type]] = 1

                    offset = 50
                    vector[offset + mark_type_to_int[lane_segment.right_mark_type]] = 1

                    offset = 70
                    vector[offset + lane_type_to_int[lane_segment.lane_type]] = 1

            end = len(vectors)
            if end > start:
                polyline_spans.append([start, end])

        argoverse2_map.get_scenario_lane_segment_ids()

        if 'goals_2D' in args.other_params:
            points = []
            visit = {}

            def get_hash(point):
                return round((point[0] + 500) * 100) * 1000000 + round((point[1] + 500) * 100)

            for index_polygon, polygon in enumerate(polygons):
                for i, point in enumerate(polygon):
                    hash = get_hash(point)
                    if hash not in visit:
                        visit[hash] = True
                        points.append(point)

                # Subdivide lanes to get more fine-grained 2D goals.
                if 'subdivide' in args.other_params:
                    subdivide_points = get_subdivide_points(polygon)
                    points.extend(subdivide_points)

            mapping['goals_2D'] = np.array(points)

        pass

    if 'goals_2D' in args.other_params:
        point_label = np.array(labels[-1])
        mapping['goals_2D_labels'] = np.argmin(get_dis(mapping['goals_2D'], point_label))

        if 'lane_scoring' in args.other_params:
            stage_one_label = 0
            min_dis = 10000.0
            for i, polygon in enumerate(polygons):
                temp = np.min(get_dis(polygon, point_label))
                if temp < min_dis:
                    min_dis = temp
                    # A lane consists of two left polyline and right polyline
                    stage_one_label = i // 2

            mapping['stage_one_label'] = stage_one_label

    # print(len(polyline_spans), len(vectors), map_start_polyline_idx, polyline_spans[map_start_polyline_idx])

    mapping.update(dict(
        matrix=np.array(vectors),
        labels=np.array(labels).reshape([args.future_frame_num, 2]),
        gt_trajectory_global_coordinates=np.array(gt_trajectory_global_coordinates),
        polyline_spans=[slice(each[0], each[1]) for each in polyline_spans],
        labels_is_valid=np.ones(args.future_frame_num, dtype=np.int64),
        eval_time=60,

        agents=agents,
        map_start_polyline_idx=map_start_polyline_idx,
        polygons=polygons,
        file_name=os.path.split(instance_dir)[-1],
        trajs=agents,
        vis_lanes=polygons,
    ))

    return mapping


def argoverse_get_instance(lines, file_name, args):
    """
    Extract polylines from one example file content.
    """

    global max_vector_num
    vector_num = 0
    id2info = {}
    mapping = {}
    mapping['file_name'] = file_name

    for i, line in enumerate(lines):

        line = line.strip().split(',')
        if i == 0:
            mapping['start_time'] = float(line[TIMESTAMP])
            mapping['city_name'] = line[CITY_NAME]

        line[TIMESTAMP] = float(line[TIMESTAMP]) - mapping['start_time']
        line[X] = float(line[X])
        line[Y] = float(line[Y])
        id = line[TRACK_ID]

        if line[OBJECT_TYPE] == 'AV' or line[OBJECT_TYPE] == 'AGENT':
            line[TRACK_ID] = line[OBJECT_TYPE]

        if line[TRACK_ID] in id2info:
            id2info[line[TRACK_ID]].append(line)
            vector_num += 1
        else:
            id2info[line[TRACK_ID]] = [line]

        if line[OBJECT_TYPE] == 'AGENT' and len(id2info['AGENT']) == 20:
            assert 'AV' in id2info
            assert 'cent_x' not in mapping
            agent_lines = id2info['AGENT']
            mapping['cent_x'] = agent_lines[-1][X]
            mapping['cent_y'] = agent_lines[-1][Y]
            mapping['agent_pred_index'] = len(agent_lines)
            mapping['two_seconds'] = line[TIMESTAMP]

            # Smooth the direction of agent. Only taking the direction of the last frame is not accurate due to label error.
            if 'direction' in args.other_params:
                span = agent_lines[-6:]
                intervals = [2]
                angles = []
                for interval in intervals:
                    for j in range(len(span)):
                        if j + interval < len(span):
                            der_x, der_y = span[j + interval][X] - span[j][X], span[j + interval][Y] - span[j][Y]
                            angles.append([der_x, der_y])

            der_x, der_y = agent_lines[-1][X] - agent_lines[-2][X], agent_lines[-1][Y] - agent_lines[-2][Y]
    if not args.do_test:
        if 'set_predict' in args.other_params:
            pass
        else:
            assert len(id2info['AGENT']) == 50

    if vector_num > max_vector_num:
        max_vector_num = vector_num

    if 'cent_x' not in mapping:
        return None

    if args.do_eval:
        origin_labels = np.zeros([30, 2])
        for i, line in enumerate(id2info['AGENT'][20:]):
            origin_labels[i][0], origin_labels[i][1] = line[X], line[Y]
        mapping['origin_labels'] = origin_labels

    angle = -get_angle(der_x, der_y) + math.radians(90)

    # Smooth the direction of agent. Only taking the direction of the last frame is not accurate due to label error.
    if 'direction' in args.other_params:
        angles = np.array(angles)
        der_x, der_y = np.mean(angles, axis=0)
        angle = -get_angle(der_x, der_y) + math.radians(90)

    mapping['angle'] = angle
    for id in id2info:
        info = id2info[id]
        for line in info:
            line[X], line[Y] = rotate(line[X] - mapping['cent_x'], line[Y] - mapping['cent_y'], angle)
        if 'scale' in mapping:
            scale = mapping['scale']
            line[X] *= scale
            line[Y] *= scale
    return preprocess(args, id2info, mapping)


class Dataset(torch.utils.data.Dataset):
    def __init__(self, args, batch_size, to_screen=True):
        data_dir = args.data_dir
        self.ex_list = []
        self.args = args

        if args.reuse_temp_file:
            pickle_file = open(os.path.join(args.temp_file_dir, get_name('ex_list')), 'rb')
            self.ex_list = pickle.load(pickle_file)
            # self.ex_list = self.ex_list[len(self.ex_list) // 2:]
            pickle_file.close()
        else:
            if args.argoverse2:
                pass
            else:
                from argoverse.map_representation.map_api import ArgoverseMap
                global am
                am = ArgoverseMap()
            if args.core_num >= 1:
                # TODO
                files = []
                for each_dir in data_dir:
                    root, dirs, cur_files = os.walk(each_dir).__next__()
                    if args.argoverse2:
                        files.extend([os.path.join(each_dir, file) for file in dirs])
                    else:
                        files.extend([os.path.join(each_dir, file) for file in cur_files if
                                      file.endswith("csv") and not file.startswith('.')])
                print(files[:5], files[-5:])

                pbar = tqdm(total=len(files))

                queue = multiprocessing.Queue(args.core_num)
                queue_res = multiprocessing.Queue()

                def calc_ex_list(queue, queue_res, args):
                    res = []
                    dis_list = []
                    while True:
                        file = queue.get()
                        if file is None:
                            break

                        def put_instance_in_queue(instance):
                            if instance is not None:
                                data_compress = zlib.compress(pickle.dumps(instance))
                                res.append(data_compress)
                                queue_res.put(data_compress)
                            else:
                                queue_res.put(None)

                        if args.argoverse2:
                            instance = argoverse2_get_instance(args, file)
                            put_instance_in_queue(instance)
                        else:
                            if file.endswith("csv"):
                                with open(file, "r", encoding='utf-8') as fin:
                                    lines = fin.readlines()[1:]
                                instance = argoverse_get_instance(lines, file, args)
                                put_instance_in_queue(instance)

                processes = [Process(target=calc_ex_list, args=(queue, queue_res, args,)) for _ in range(args.core_num)]
                for each in processes:
                    each.start()
                # res = pool.map_async(calc_ex_list, [queue for i in range(args.core_num)])
                for file in files:
                    assert file is not None
                    queue.put(file)
                    pbar.update(1)

                # necessary because queue is out-of-order
                while not queue.empty():
                    pass

                pbar.close()

                self.ex_list = []

                pbar = tqdm(total=len(files))
                for i in range(len(files)):
                    t = queue_res.get()
                    if t is not None:
                        self.ex_list.append(t)
                    pbar.update(1)
                pbar.close()
                pass

                for i in range(args.core_num):
                    queue.put(None)
                for each in processes:
                    each.join()

            else:
                assert False

            pickle_file = open(os.path.join(args.temp_file_dir, get_name('ex_list')), 'wb')
            pickle.dump(self.ex_list, pickle_file)
            pickle_file.close()
        assert len(self.ex_list) > 0
        if to_screen:
            print("valid data size is", len(self.ex_list))
            logging('max_vector_num', max_vector_num)
        self.batch_size = batch_size

    def __len__(self):
        return len(self.ex_list)

    def __getitem__(self, idx):
        # file = self.ex_list[idx]
        # pickle_file = open(file, 'rb')
        # instance = pickle.load(pickle_file)
        # pickle_file.close()

        data_compress = self.ex_list[idx]
        instance = pickle.loads(zlib.decompress(data_compress))
        return instance


def post_eval(args, file2pred, file2labels, DEs):
    score_file = args.model_recover_path.split('/')[-1]
    for each in args.eval_params:
        each = str(each)
        if len(each) > 15:
            each = 'long'
        score_file += '.' + str(each)
        # if 'minFDE' in args.other_params:
        #     score_file += '.minFDE'
    if args.method_span[0] >= utils.NMS_START:
        score_file += '.NMS'
    else:
        score_file += '.score'

    for method in utils.method2FDEs:
        FDEs = utils.method2FDEs[method]
        miss_rate = np.sum(np.array(FDEs) > 2.0) / len(FDEs)
        if method >= utils.NMS_START:
            method = 'NMS=' + str(utils.NMS_LIST[method - utils.NMS_START])
        utils.logging(
            'method {}, FDE {}, MR {}, other_errors {}'.format(method, np.mean(FDEs), miss_rate, utils.other_errors_to_string()),
            type=score_file, to_screen=True, append_time=True)
    utils.logging('other_errors {}'.format(utils.other_errors_to_string()),
                  type=score_file, to_screen=True, append_time=True)
    if args.argoverse2:
        pass
    else:
        from argoverse.evaluation import eval_forecasting
        metric_results = eval_forecasting.get_displacement_errors_and_miss_rate(file2pred, file2labels, 6, 30, 2.0)
    utils.logging(metric_results, type=score_file, to_screen=True, append_time=True)
    DE = np.concatenate(DEs, axis=0)
    length = DE.shape[1]
    DE_score = [0, 0, 0, 0]
    for i in range(DE.shape[0]):
        DE_score[0] += DE[i].mean()
        for j in range(1, 4):
            index = round(float(length) * j / 3) - 1
            assert index >= 0
            DE_score[j] += DE[i][index]
    for j in range(4):
        score = DE_score[j] / DE.shape[0]
        utils.logging('ADE' if j == 0 else 'DE@1' if j == 1 else 'DE@2' if j == 2 else 'DE@3', score,
                      type=score_file, to_screen=True, append_time=True)

    utils.logging(vars(args), is_json=True,
                  type=score_file, to_screen=True, append_time=True)
