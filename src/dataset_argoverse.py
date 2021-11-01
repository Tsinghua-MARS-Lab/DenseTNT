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

import numpy as np
import torch
from argoverse.map_representation.map_api import ArgoverseMap
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
        assert isinstance(am, ArgoverseMap)
        if 'semantic_lane' in args.other_params:
            lane_ids = am.get_lane_ids_in_xy_bbox(x, y, city_name, query_search_range_manhattan=args.max_distance)
            # lane_centerline = am.city_lane_centerlines_dict[city_name][lane_ids[0]].centerline
            # print(lane_centerline.shape, am.get_ground_height_at_xy(lane_centerline, city_name))
            local_lane_centerlines = [am.get_lane_segment_centerline(lane_id, city_name) for lane_id in lane_ids]
            polygons = local_lane_centerlines
            # z = am.get_ground_height_at_xy(np.array([[x, y]]), city_name)[0]

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

        if args.use_centerline:
            if 'semantic_lane' in args.other_params:
                local_lane_centerlines = [polygon for polygon in polygons]

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

        if 'stage_one' in args.other_params:
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
            global am
            am = ArgoverseMap()
            if args.core_num >= 1:
                # TODO
                files = []
                for each_dir in data_dir:
                    root, dirs, cur_files = os.walk(each_dir).__next__()
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
                        if file.endswith("csv"):
                            with open(file, "r", encoding='utf-8') as fin:
                                lines = fin.readlines()[1:]
                            instance = argoverse_get_instance(lines, file, args)
                            if instance is not None:
                                data_compress = zlib.compress(pickle.dumps(instance))
                                res.append(data_compress)
                                queue_res.put(data_compress)
                            else:
                                queue_res.put(None)

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
    from argoverse.evaluation import eval_forecasting

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
