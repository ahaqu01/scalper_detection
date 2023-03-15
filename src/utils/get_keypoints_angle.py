# utf-8
import math


# 0: 'nose',
# 1: 'left_eye',
# 2: 'right_eye',
# 3: 'left_ear',
# 4: 'right_ear',
# 5: 'left_shoulder',
# 6: 'right_shoulder',
# 7: 'left_elbow',
# 8: 'right_elbow',
# 9: 'left_wrist',
# 10: 'right_wrist',
# 11: 'left_hip',
# 12: 'right_hip',
# 13: 'left_knee',
# 14: 'right_knee',
# 15: 'left_ankle',
# 16: 'right_ankle'

def angle_between_points(p0, p1, p2):
    # 计算角度
    a = (p1[0] - p0[0]) ** 2 + (p1[1] - p0[1]) ** 2
    b = (p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2
    c = (p2[0] - p0[0]) ** 2 + (p2[1] - p0[1]) ** 2
    if a * b == 0:
        return -1.0
    return math.acos((a + b - c) / math.sqrt(4 * a * b)) * 180 / math.pi


def length_between_points(p0, p1):
    # 2点之间的距离
    return math.hypot(p1[0] - p0[0], p1[1] - p0[1])


def get_angle_point(human, pos, kpt_score_thr=0.3):
    # 返回各个部位的关键点
    pnts = []
    if pos == 'left_elbow':
        pos_list = [5, 7, 9]
    elif pos == 'right_elbow':
        pos_list = [6, 8, 10]
    elif pos == 'left_shoulder':
        pos_list = [6, 5, 7]
    elif pos == 'right_shoulder':
        pos_list = [5, 6, 8]
    elif pos == 'left_shoulder_wrist':
        pos_list = [6, 5, 9]
    elif pos == 'right_shoulder_wrist':
        pos_list = [5, 6, 10]
    elif pos == 'left_hip':
        pos_list = [12, 11, 13]
    elif pos == 'right_hip':
        pos_list = [11, 12, 14]
    elif pos == 'left_shoulder_hip_knee':
        pos_list = [5, 11, 13]
    elif pos == 'right_shoulder_hip_knee':
        pos_list = [6, 12, 14]
    elif pos == 'left_knee':
        pos_list = [11, 13, 15]
    elif pos == 'right_knee':
        pos_list = [12, 14, 16]
    elif pos == 'left_hip_shoulder_wrist':
        pos_list = [11, 5, 9]
    elif pos == 'right_hip_shoulder_wrist':
        pos_list = [12, 6, 10]
    else:
        # print('Unknown  [%s]', pos)
        return pnts

    for i in range(3):
        if human[pos_list[i]][2] <= kpt_score_thr:
            # print('component [%d] incomplete' % (pos_list[i]))
            return pnts
        pnts.append((int(human[pos_list[i]][0]), int(human[pos_list[i]][1])))
    return pnts


def get_angle(human, pnts_name='', kpt_score_thr=0.3):
    angle = -1
    pnts = get_angle_point(human, pnts_name, kpt_score_thr=kpt_score_thr)
    if len(pnts) != 3 or pnts is None:
        # print('component incomplete')
        return angle
    angle = angle_between_points(pnts[0], pnts[1], pnts[2])
    # print('{} angle:{}'.format(pnts_name, angle))
    return angle

if __name__ == "__main__":
    angle = angle_between_points((1,2), (1,1), (2,0))
    print(angle)