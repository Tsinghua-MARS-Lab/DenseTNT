# cython -a src/utils_cython.pyx && python src/setup.py build_ext --inplace
# cd src/ && cython -a utils_cython.pyx && python setup.py build_ext --inplace && cd ..
# cython: language_level=3, boundscheck=False, wraparound=False

language_level = 3
import numpy as np
cimport numpy as np
cimport cython
from libc.stdlib cimport rand, RAND_MAX
from libc.time cimport clock, CLOCKS_PER_SEC

cdef extern from "math.h":
    double sin(double x)
    double sqrt(double x)
    double cos(double x)
    double exp(double x)
    double fabs(double x)

cdef:
    float M_PI = 3.14159265358979323846
    int pixel_num_1m = 4

cdef int get_round(float a):
    if a > 0:
        return int(a + 0.5)
    else:
        return -int(fabs(a) + 0.5)

cdef np.float32_t get_dis_point(np.float32_t a, np.float32_t b):
    return sqrt(a * a + b * b)

cdef np.float32_t get_point_for_ratio(np.ndarray[np.float32_t, ndim=1] point, np.ndarray[np.float32_t, ndim=1] end,
                                      np.float32_t ratio, int c):
    return point[c] * (1.0 - ratio) + end[c] * ratio

def _normalize(np.ndarray[np.float32_t, ndim=2] polygon, np.float32_t angle, np.float32_t center_point_y):
    cdef np.float32_t cos_, sin_, min_sqr_dis, temp
    min_sqr_dis = 10000.0
    cdef int i, n
    cos_ = cos(angle)
    sin_ = sin(angle)
    n = polygon.shape[0]
    cdef np.ndarray[np.float32_t, ndim=2] new_points = np.zeros((n, 2), dtype=np.float32)
    for i in range(n):
        new_points[i, 0] = polygon[i, 0] * cos_ - polygon[i, 1] * sin_
        new_points[i, 1] = polygon[i, 0] * sin_ + polygon[i, 1] * cos_
        temp = center_point_y - new_points[i, 1]
        min_sqr_dis = min(min_sqr_dis, new_points[i, 0] * new_points[i, 0] + temp * temp)
    return new_points, min_sqr_dis

def normalize(polygon, cent_x, cent_y, angle, center_point):
    polygon[:, 0] -= cent_x
    polygon[:, 1] -= cent_y
    return _normalize(polygon, angle, center_point[1])

cdef np.float32_t get_sqr_dis_point(np.float32_t a, np.float32_t b):
    return a * a + b * b

def _get_pseudo_label(np.ndarray[np.float32_t, ndim=2] predicts, np.ndarray[np.float32_t, ndim=2] labels,
                      np.ndarray[np.float32_t, ndim=1] self_cost, int is_manhatan, int match_l2):
    cdef np.float32_t a
    cdef int i, n, j, t, k, r
    n = predicts.shape[0]
    k = labels.shape[0]
    assert n >= k
    cdef np.ndarray[np.float32_t, ndim=2] C = np.zeros((n, k), dtype=np.float32)
    cdef np.ndarray[np.float32_t, ndim=2] pseudo_label = np.zeros((n, 2), dtype=np.float32)
    matched = np.zeros(n, dtype=np.int32)

    if match_l2:
        for i in range(n):
            for j in range(k):
                C[i, j] = get_sqr_dis_point(predicts[i, 0] - labels[j, 0], predicts[i, 1] - labels[j, 1]) + self_cost[i]
    elif is_manhatan:
        for i in range(n):
            for j in range(k):
                C[i, j] = fabs(predicts[i, 0] - labels[j, 0]) + fabs(predicts[i, 1] - labels[j, 1]) + self_cost[i]
    else:
        for i in range(n):
            for j in range(k):
                C[i, j] = get_dis_point(predicts[i, 0] - labels[j, 0], predicts[i, 1] - labels[j, 1]) + self_cost[i]

    # remove out of function
    from scipy.optimize import linear_sum_assignment
    r_list, c_list = linear_sum_assignment(C)
    r_list = r_list.astype(np.int32)
    c_list = c_list.astype(np.int32)

    for i in range(k):
        t = c_list[i]
        r = r_list[i]
        matched[r] = 1
        pseudo_label[r, 0] = labels[t, 0]
        pseudo_label[r, 1] = labels[t, 1]

    return pseudo_label, C[r_list, c_list].sum(), matched

def get_pseudo_label(predicts, labels, self_cost, kwargs):
    is_manhatan = kwargs.get('is_manhatan', False)
    match_l2 = kwargs.get('match_l2', False)
    pseudo_label, cost, matched = _get_pseudo_label(predicts, labels, self_cost, is_manhatan, match_l2)
    pseudo_label = pseudo_label[np.nonzero(matched)[0]]
    return pseudo_label, cost, matched

def get_rotate_lane_matrix(lane_matrix, x, y, angle):
    return _get_rotate_lane_matrix(lane_matrix, x, y, angle)

def _get_rotate_lane_matrix(np.ndarray[np.float32_t, ndim=2] lane_matrix, np.float32_t x, np.float32_t y, np.float32_t angle):
    cdef np.float32_t sin_, cos_, dx, dy
    cdef int i, n
    cos_ = cos(angle)
    sin_ = sin(angle)
    n = lane_matrix.shape[0]
    cdef np.ndarray[np.float32_t, ndim=2] res = np.zeros((n, 20), dtype=np.float32)
    for r in range(n):
        for i in range(10):
            dx = lane_matrix[r, i * 2] - x
            dy = lane_matrix[r, i * 2 + 1] - y
            res[r, i * 2] = dx * cos_ - dy * sin_
            res[r, i * 2 + 1] = dx * sin_ + dy * cos_
    return res

cdef np.float32_t get_rand(np.float32_t l, np.float32_t r):
    cdef np.float32_t t = rand()
    return l + t / RAND_MAX * (r - l)

cdef int get_rand_int(int l, int r):
    return l + rand() % (r - l + 1)

cdef np.float32_t get_value(np.ndarray[np.float32_t, ndim=2] goals_2D, np.ndarray[np.float32_t, ndim=1] scores,
                            np.ndarray[np.float32_t, ndim=2] selected_points, int n, objective, int cnt_sample, np.float32_t MRratio,
                            kwargs):
    cdef np.float32_t value = 0.0, cnt_hit, x, y, sum, minFDE, t_float, miss_error, stride, s_x, s_y
    cdef int i, j, k, need, cnt, t_int, objective_int, cnt_len = 0, a, b
    cdef np.ndarray[np.float32_t, ndim=1] point = np.zeros(2, dtype=np.float32)

    for i in range(100):
        if i * i == cnt_sample:
            cnt_len = i
    if cnt_len == 0:
        assert False, 'cnt_sample != square'

    for i in range(n):
        point[0], point[1] = goals_2D[i, 0], goals_2D[i, 1]
        sum = 0.0
        t_int = int(scores[i] * 1000)

        if t_int > 10:
            cnt = cnt_len * 3
        elif t_int > 5:
            cnt = cnt_len * 2
        else:
            cnt = cnt_len

        t_float = cnt
        stride = 1.0 / t_float

        s_x = point[0] - 0.5 + stride / 2.0
        s_y = point[1] - 0.5 + stride / 2.0

        for a in range(cnt):
            for b in range(cnt):
                # x = get_rand(point[0] - 0.5, point[0] + 0.5)
                # y = get_rand(point[1] - 0.5, point[1] + 0.5)
                x = s_x + a * stride
                y = s_y + b * stride
                minFDE = 10000.0
                miss_error = 10.0
                for j in range(6):
                    t_float = get_dis_point(x - selected_points[j, 0], y - selected_points[j, 1])
                    if t_float < minFDE:
                        minFDE = t_float
                if minFDE <= 2.0:
                    miss_error = 0.0
                sum += minFDE * (1.0 - MRratio) + miss_error * MRratio
        sum /= cnt * cnt
        value += scores[i] * sum

    return value

args = None

def _get_optimal_targets(np.ndarray[np.float32_t, ndim=2] goals_2D, np.ndarray[np.float32_t, ndim=1] scores,
                         file_name, objective, int num_step, int cnt_sample, MRratio, float opti_time, kwargs):
    cdef np.float32_t t, threshold, expectation, nxt_expectation, lr, ratio, fire_prob, min_expectation
    cdef int i, j, n, m, t_int, step, op, ok, go
    n = goals_2D.shape[0]
    m = 0
    threshold = 0.001
    cdef np.ndarray[np.float32_t, ndim=2] ans_points = np.zeros((6, 2), dtype=np.float32)
    cdef np.ndarray[np.float32_t, ndim=2] nxt_points = np.zeros((6, 2), dtype=np.float32)
    cdef np.ndarray[np.float32_t, ndim=1] pred_probs = np.zeros(6, dtype=np.float32)
    cdef:
        float best_expectation = 10000.0
        np.ndarray[np.float32_t, ndim=2] best_points = np.zeros((6, 2), dtype=np.float32)

    cdef float start_time = clock()
    if opti_time < 100.0:
        num_step = 1000_000

    for i in range(n):
        if scores[i] >= threshold:
            goals_2D[m, 0] = goals_2D[i, 0]
            goals_2D[m, 1] = goals_2D[i, 1]
            scores[m] = scores[i]
            m += 1
    if m == 0:
        print('warning: m == 0')
        m = n

    n = m

    for j in range(6):
        t_int = get_rand_int(0, n - 1)
        ans_points[j, 0] = goals_2D[t_int, 0]
        ans_points[j, 1] = goals_2D[t_int, 1]

    expectation = get_value(goals_2D, scores, ans_points, n, objective, cnt_sample, MRratio, kwargs)
    for step in range(num_step):
        if (clock() - start_time) / CLOCKS_PER_SEC >= opti_time:
            break

        ratio = step
        ratio = ratio / num_step

        lr = exp(-(ratio * 2))
        for j in range(6):
            nxt_points[j, 0] = ans_points[j, 0]
            nxt_points[j, 1] = ans_points[j, 1]

        op = get_rand_int(0, 0)
        if op == 0:
            while True:
                ok = 0
                for j in range(6):
                    if get_rand(0.0, 1.0) < 0.3:
                        nxt_points[j, 0] += get_rand(-lr, lr)
                        nxt_points[j, 1] += get_rand(-lr, lr)
                        ok = 1
                if ok:
                    break

        nxt_expectation = get_value(goals_2D, scores, nxt_points, n, objective, cnt_sample, MRratio, kwargs)

        go = 0
        if nxt_expectation < expectation:
            go = 1
        else:
            fire_prob = 0.01
            if get_rand(0.0, 1.0) < fire_prob:
                go = 1

        if go:
            expectation = nxt_expectation
            for j in range(6):
                ans_points[j, 0] = nxt_points[j, 0]
                ans_points[j, 1] = nxt_points[j, 1]

        if expectation < best_expectation:
            best_expectation = expectation
            for j in range(6):
                best_points[j, 0], best_points[j, 1] = ans_points[j, 0], ans_points[j, 1]

    min_expectation = 10000.0
    argmin = 0

    for j in range(6):
        for k in range(6):
            nxt_points[k, 0], nxt_points[k, 1] = best_points[j, 0], best_points[j, 1]
        t = get_value(goals_2D, scores, nxt_points, n, 'minFDE', cnt_sample, MRratio, kwargs)
        pred_probs[j] = 1.0 - get_value(goals_2D, scores, nxt_points, n, objective, cnt_sample, MRratio, kwargs)
        if t < min_expectation:
            min_expectation = t
            argmin = j

    return best_expectation, best_points, argmin, pred_probs

def get_optimal_targets(goals_2D, scores, file_name, objective, opti_time, kwargs: dict = None):
    MRratio = kwargs.get('MRratio', 1.0)
    cnt_sample = kwargs.get('cnt_sample', 2)
    num_step = kwargs.get('num_step', 4000)
    expectation, ans_points, argmin, pred_probs = _get_optimal_targets(goals_2D, scores, file_name, objective, num_step, cnt_sample,
                                                                       MRratio, opti_time, kwargs)
    argsort = np.argsort(-pred_probs)
    ans_points = ans_points[argsort]
    pred_probs = pred_probs[argsort]
    # TODO
    # ans_points[0, 0], ans_points[argmin, 0] = ans_points[argmin, 0], ans_points[0, 0]
    # ans_points[0, 1], ans_points[argmin, 1] = ans_points[argmin, 1], ans_points[0, 1]
    return expectation, ans_points, pred_probs

def _get_normalized(np.ndarray[np.float32_t, ndim=3] polygons, np.float32_t x, np.float32_t y, np.float32_t angle):
    cdef:
        np.float32_t cos_, sin_, min_sqr_dis, temp
        int i, n, polygon_idx
    cos_ = cos(angle)
    sin_ = sin(angle)
    n = polygons.shape[1]
    cdef np.ndarray[np.float32_t, ndim=3] new_polygons = np.zeros((polygons.shape[0], n, 2), dtype=np.float32)
    for polygon_idx in range(polygons.shape[0]):
        for i in range(n):
            polygons[polygon_idx, i, 0] -= x
            polygons[polygon_idx, i, 1] -= y
            new_polygons[polygon_idx, i, 0] = polygons[polygon_idx, i, 0] * cos_ - polygons[polygon_idx, i, 1] * sin_
            new_polygons[polygon_idx, i, 1] = polygons[polygon_idx, i, 0] * sin_ + polygons[polygon_idx, i, 1] * cos_
    return new_polygons

def get_normalized(trajectorys, normalizer, reverse=False):
    if trajectorys.dtype is not np.float32:
        trajectorys = trajectorys.astype(np.float32)

    if reverse:
        return _get_normalized(trajectorys, normalizer.origin[0], normalizer.origin[1], -normalizer.yaw)
    return _get_normalized(trajectorys, normalizer.x, normalizer.y, normalizer.yaw)

def get_normalized_points(points: np.ndarray, normalizer, reverse=False):
    if points.dtype is not np.float32:
        points = points.astype(np.float32)

    trajectorys = points[np.newaxis, :]
    if reverse:
        return _get_normalized(trajectorys, normalizer.origin[0], normalizer.origin[1], -normalizer.yaw)[0]
    return _get_normalized(trajectorys, normalizer.x, normalizer.y, normalizer.yaw)[0]

cdef float _set_predict_get_value(np.ndarray[np.float32_t, ndim=2] goals_2D,
                                  np.ndarray[np.float32_t, ndim=1] scores,
                                  np.ndarray[np.float32_t, ndim=2] selected_points,
                                  kwargs):
    cdef:
        int n = goals_2D.shape[0]
        np.ndarray[np.float32_t, ndim=1] point = np.zeros(2, dtype=np.float32)
        float MRratio = 1.0

        float value = 0.0, cnt_hit, x, y, sum, minFDE, t_float, miss_error, stride, s_x, s_y
        int i, j, k, need, cnt, t_int, objective_int, cnt_len = 3, a, b

    if kwargs is not None and 'set_predict-MRratio' in kwargs:
        MRratio = float(kwargs['set_predict-MRratio'])

    for i in range(n):
        point[0], point[1] = goals_2D[i, 0], goals_2D[i, 1]

        if True:
            sum = 0.0
            t_int = int(scores[i] * 1000)

            if t_int > 10:
                cnt = cnt_len * 3
            elif t_int > 5:
                cnt = cnt_len * 2
            else:
                cnt = cnt_len

            t_float = cnt
            stride = 1.0 / t_float

            s_x = point[0] - 0.5 + stride / 2.0
            s_y = point[1] - 0.5 + stride / 2.0

            for a in range(cnt):
                for b in range(cnt):
                    # x = get_rand(point[0] - 0.5, point[0] + 0.5)
                    # y = get_rand(point[1] - 0.5, point[1] + 0.5)
                    x = s_x + a * stride
                    y = s_y + b * stride
                    minFDE = 10000.0
                    miss_error = 1.0
                    # warning: miss_error is not 10.0
                    for j in range(6):
                        t_float = get_dis_point(x - selected_points[j, 0], y - selected_points[j, 1])
                        if t_float < minFDE:
                            minFDE = t_float
                    if minFDE <= 2.0:
                        miss_error = 0.0
                    sum += minFDE * (1.0 - MRratio) + miss_error * MRratio
            sum /= cnt * cnt
            value += scores[i] * sum

    return value

def set_predict_get_value(goals_2D, scores, selected_points, kwargs=None):
    return _set_predict_get_value(goals_2D, scores, selected_points, kwargs)

def _set_predict_next_step(np.ndarray[np.float32_t, ndim=2] goals_2D,
                           np.ndarray[np.float32_t, ndim=1] scores,
                           np.ndarray[np.float32_t, ndim=2] selected_points,
                           float lr, kwargs):
    cdef:
        int step, j, ok, num_step = 100
        float nxt_expectation, best_expectation = _set_predict_get_value(goals_2D, scores, selected_points, kwargs)
        np.ndarray[np.float32_t, ndim = 2] nxt_points = np.zeros((6, 2), dtype=np.float32)
        np.ndarray[np.float32_t, ndim = 2] best_points = selected_points.copy()

    if kwargs is not None and 'dynamic_label-double' in kwargs:
        num_step = 200

    for step in range(num_step):
        for j in range(6):
            nxt_points[j, 0] = selected_points[j, 0]
            nxt_points[j, 1] = selected_points[j, 1]

        if True:
            while True:
                ok = 0
                for j in range(6):
                    if get_rand(0.0, 1.0) < 0.5:
                        nxt_points[j, 0] += get_rand(-lr, lr)
                        nxt_points[j, 1] += get_rand(-lr, lr)
                        ok = 1
                if ok:
                    break

        nxt_expectation = _set_predict_get_value(goals_2D, scores, nxt_points, kwargs)
        if nxt_expectation < best_expectation:
            best_expectation = nxt_expectation
            for j in range(6):
                best_points[j, 0], best_points[j, 1] = nxt_points[j, 0], nxt_points[j, 1]

    return nxt_expectation, best_points

def set_predict_next_step(goals_2D, scores, selected_points, lr=1.0, kwargs=None):
    return _set_predict_next_step(goals_2D, scores, selected_points, lr, kwargs)
