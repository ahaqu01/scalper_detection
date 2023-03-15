import cv2
import numpy as np

import torch
import torchvision
from models.retinaface.src.face_detect import face_detector
from models.facenet.src.cal_embedding import cal_face_embedding
from models.yolox.src.detecter import Detecter
from models.pose_det.src.pose_detecter import pose_detecter
from models.bytemot.src.deep_reid import DeepReid
from models.image_definition_det.src.musiq import Musiq
from utils.get_keypoints_angle import get_angle
from configs.configs import single_frame_inference as single_frame_inference_cfg


class single_frame_inference(object):
    def __init__(self, device):
        self.model_weight_paths = single_frame_inference_cfg["model_weight_paths"]
        self.config_paths = single_frame_inference_cfg["config_paths"]
        self.other_cfgs = single_frame_inference_cfg["other_cfgs"]

        self.device = device
        self.facenet = cal_face_embedding(
            model_weights=self.model_weight_paths["facenet"],
            device=device
        )

        self.retinaface = face_detector(
            backbone_name=self.config_paths["retinaface"],
            model_weights=self.model_weight_paths["retinafce"],
            confidence_threshold=self.other_cfgs["retinafce"]["confidence_threshold"],
            top_k=self.other_cfgs["retinafce"]["top_k"],
            nms_threshold=self.other_cfgs["retinafce"]["nms_threshold"],
            keep_top_k=self.other_cfgs["retinafce"]["keep_top_k"],
            vis_thres=self.other_cfgs["retinafce"]["vis_thres"],
            device=device
        )
        self.yolox = Detecter(
            input_size=self.other_cfgs["yolox"]["input_size"],
            model_weighs=self.model_weight_paths["yolox"],
            model_config=self.config_paths["yolox"],
            device=device,
            half=False,
            fuse=False
        )
        self.pose_det = pose_detecter(
            model_cfg=self.config_paths["pose_det"],
            model_weights=self.model_weight_paths["pose_det"],
            device=device
        )
        self.mot = DeepReid(
            extractor_config=self.config_paths["mot"]["extractor"],
            extractor_weights=self.model_weight_paths["mot"],
            tracker_config=self.config_paths["mot"]["tracker"],
            device=device
        )
        self.img_definition_f = Musiq(
            musiq_config=self.config_paths["img_definition"],
            musiq_weights=self.model_weight_paths["img_definition"],
            device=device
        )
        tf_list = []
        tf_list.append(torchvision.transforms.Resize([256, 128]))
        tf_list.append(torchvision.transforms.ToTensor())
        tf_list.append(torchvision.transforms.Lambda(lambda x: x * 1.0))
        trans = torchvision.transforms.Compose(tf_list)
        self.img_definition_p = Musiq(
            musiq_config=self.config_paths["img_definition"],
            musiq_weights=self.model_weight_paths["img_definition"],
            device=device,
            trans=trans
        )

    def inference(self, frame):
        # extract data
        # frame = inference_data["resize_frame"]
        # frame_raw = inference_data["raw_frame"]
        # ratio = inference_data["ratio"]
        frame_bgr = frame.copy()
        src_h, src_w = frame_bgr.shape[:2]
        frame_rgb = frame.copy()[:, :, ::-1]

        # human det
        yolovx_pred = self.yolox.inference(frame_bgr)

        # tracker
        if yolovx_pred[0] is not None:
            mot_pred, added_track_ids = self.mot.update(bbox_xyxy=yolovx_pred[0][:, :4].cpu().numpy(),
                                                        confidences=yolovx_pred[0][:, 4].cpu().numpy(),
                                                        ori_img=frame_bgr)

        else:
            mot_pred = {}
            added_track_ids = []

        # face det
        retinaface_pred = self.retinaface.inference_single(img_raw=frame_bgr)
        # del
        bboxes, retinaface_pred = self.retinaface.del_irregular_bboxes(retinaface_pred, frame_bgr)

        # get face features and face imgs
        if bboxes.shape[0] >= 1:
            # get face embedding
            # faces = self.facenet.get_crop_processed_faces(frame_bgr, bboxes)
            # faces_embedding = self.facenet.cal_embedding(faces)
            faces_embedding = self.facenet.get_img_faces_embedding(frame_bgr, bboxes).cpu().numpy()

            # get face definition
            # scale = 3
            # faces_ori = []
            # for bbox in bboxes:
            #     l_x, l_y, r_x, r_y = bbox[0], bbox[1], bbox[2], bbox[3]
            #     face_w_t = scale * (r_x -l_x)
            #     face_h_t = scale * (r_y -l_y)
            #     c_x = (l_x + r_x) // 2
            #     c_y = (l_y + r_y) // 2
            #     l_x_t = int(min(max(c_x - face_w_t // 2, 0), src_w))
            #     r_x_t = int(min(max(c_x + face_w_t // 2, 0), src_w))
            #     l_y_t = int(min(max(c_y - face_h_t // 2, 0), src_h))
            #     r_y_t = int(min(max(c_y + face_h_t // 2, 0), src_h))
            #     faces_ori.append(frame_bgr[l_y_t:r_y_t, l_x_t:r_x_t])
            faces_ori = [frame_rgb[bbox[1]:bbox[3], bbox[0]:bbox[2]] for bbox in bboxes]
            faces_definition = self.img_definition_f.batch_inference(faces_ori).cpu().numpy()
        else:
            faces_embedding = None
            faces_definition = None
            faces_ori = []

        if len(mot_pred) >= 1:
            # pose det
            bbox_xyxy_tracker = np.array([mot_pred[mot_pred_track_id]["bbox"] \
                                          for mot_pred_track_id in list(mot_pred.keys())])
            scores_tracker = np.expand_dims(np.array([mot_pred[mot_pred_track_id]["confidence"] \
                                                      for mot_pred_track_id in list(mot_pred.keys())]), -1)
            tracker_pred = np.concatenate([bbox_xyxy_tracker, scores_tracker], axis=-1)
            pose_det_pred = self.pose_det.inference(frame_rgb, bboxs=torch.from_numpy(tracker_pred))

            # get person definition
            person_imgs_tracker = [mot_pred[mot_pred_track_id]["person_img"] \
                                   for mot_pred_track_id in list(mot_pred.keys())]
            person_imgs_definition = self.img_definition_p.batch_inference(person_imgs_tracker).cpu().numpy()

            # res integration
            mot_pred_track_ids = list(mot_pred.keys())
            frame_res = self.preds_integration(tracker_pred, mot_pred_track_ids,
                                               retinaface_pred, faces_embedding,
                                               faces_ori, faces_definition,
                                               pose_det_pred, person_imgs_definition,
                                               mot_pred)
            return frame_res
        else:
            return {}

    def humanbboxs_facebboxs_match(self, face_bboxs, human_bboxs):
        # key is human det res id, value is a list, list[0] is face det res id, list[1] is face-human-det-bboxs inner ratio
        face_bboxs = face_bboxs.copy().astype(np.int32)
        human_bboxs = human_bboxs.copy().astype(np.int32)

        fx1 = face_bboxs[:, 0]
        fy1 = face_bboxs[:, 1]
        fx2 = face_bboxs[:, 2]
        fy2 = face_bboxs[:, 3]
        face_areas = (fx2 - fx1 + 1) * (fy2 - fy1 + 1)

        match_res = {}
        for i in range(face_bboxs.shape[0]):
            inner_x1 = np.maximum(human_bboxs[:, 0], face_bboxs[i, 0])
            inner_x2 = np.minimum(human_bboxs[:, 2], face_bboxs[i, 2])
            inner_y1 = np.maximum(human_bboxs[:, 1], face_bboxs[i, 1])
            inner_y2 = np.minimum(human_bboxs[:, 3], face_bboxs[i, 3])
            inner_h = np.maximum(0.0, inner_y2 - inner_y1 + 1)
            inner_w = np.maximum(0.0, inner_x2 - inner_x1 + 1)
            inner_area = inner_w * inner_h
            max_inner_ratio = np.max(inner_area) / face_areas[i]
            max_inner_id = np.argmax(inner_area)
            match_res[max_inner_id] = [i, max_inner_ratio]

        return match_res

    def scalper_judge(self, pose_key_points):
        is_scalper = False
        kpt_score_thr = self.other_cfgs["kpt_score_thr"]
        # Judge whether to stand
        # left_hip_angle = get_angle(pose_key_points, pnts_name="left_hip", kpt_score_thr=kpt_score_thr)
        # right_hip_angle = get_angle(pose_key_points, pnts_name="right_hip", kpt_score_thr=kpt_score_thr)
        # stand_prob = -1
        # if left_hip_angle != -1 and right_hip_angle != -1:
        #     left_hip_angle = abs(left_hip_angle - 90)
        #     right_hip_angle = abs(right_hip_angle - 90)
        #     stand_prob = 1 - (left_hip_angle / 90 + right_hip_angle / 90) / 2
        left_shoulder_hip_knee_angle = get_angle(pose_key_points, pnts_name="left_shoulder_hip_knee",
                                                 kpt_score_thr=kpt_score_thr)
        right_shoulder_hip_knee_angle = get_angle(pose_key_points, pnts_name="right_shoulder_hip_knee",
                                                  kpt_score_thr=kpt_score_thr)
        left_knee_angle = get_angle(pose_key_points, pnts_name="left_knee", kpt_score_thr=kpt_score_thr)
        right_knee_angle = get_angle(pose_key_points, pnts_name="right_knee", kpt_score_thr=kpt_score_thr)
        stand_prob = -1
        if left_shoulder_hip_knee_angle != -1 and right_shoulder_hip_knee_angle != -1 and left_knee_angle != -1 and right_knee_angle != -1:
            stand_prob = (
                                 left_shoulder_hip_knee_angle / 180 + right_shoulder_hip_knee_angle / 180 + left_knee_angle / 180 + right_knee_angle / 180) / 4

        # Judge whether Straighten your arms
        left_arm_extension_prob = -1
        left_elbow_angle = get_angle(pose_key_points, pnts_name="left_elbow", kpt_score_thr=kpt_score_thr)
        if left_elbow_angle != -1:
            left_arm_extension_prob = left_elbow_angle / 180

        right_arm_extension_prob = -1
        right_elbow_angle = get_angle(pose_key_points, pnts_name="right_elbow", kpt_score_thr=kpt_score_thr)
        if right_elbow_angle != -1:
            right_arm_extension_prob = right_elbow_angle / 180

        # Judge whether raise your elbow
        left_elbow_raising_prob = -1
        left_hip_shoulder_elbow_angle = get_angle(pose_key_points, pnts_name="left_hip_shoulder_wrist",
                                                  kpt_score_thr=kpt_score_thr)
        if left_hip_shoulder_elbow_angle != -1:
            left_elbow_raising_prob = left_hip_shoulder_elbow_angle / 180

        right_elbow_raising_prob = -1
        right_hip_shoulder_elbow_angle = get_angle(pose_key_points, pnts_name="right_hip_shoulder_wrist",
                                                   kpt_score_thr=kpt_score_thr)
        if right_hip_shoulder_elbow_angle != -1:
            right_elbow_raising_prob = right_hip_shoulder_elbow_angle / 180

        stand_threshold, arm_extension_threshold, elbow_raising_threshold = \
            self.other_cfgs["stand_threshold"], self.other_cfgs["arm_extension_threshold"], self.other_cfgs[
                "elbow_raising_threshold"]
        if stand_prob > stand_threshold and \
                right_arm_extension_prob > arm_extension_threshold and \
                right_elbow_raising_prob > elbow_raising_threshold:
            is_scalper = True
        if stand_prob > stand_threshold and \
                left_arm_extension_prob > arm_extension_threshold and \
                left_elbow_raising_prob > elbow_raising_threshold:
            is_scalper = True
        return is_scalper

    def preds_integration(self,
                          tracker_pred,
                          tracker_ids,
                          retinaface_pred,
                          faces_embedding,
                          faces_ori,
                          faces_definition,
                          pose_det_pred,
                          person_imgs_definition,
                          mot_pred):

        # match human det res and face det res
        humanbboxs_facebboxs_match_res = {}
        if retinaface_pred.shape[0] >= 1 and tracker_pred.shape[0] >= 1:
            humanbboxs_facebboxs_match_res = self.humanbboxs_facebboxs_match(retinaface_pred[:, :4],
                                                                             tracker_pred[:, :4])
        for index, track_id in enumerate(tracker_ids):
            # add face det res & face emdedding & face img % face definition in mot_red
            if index in humanbboxs_facebboxs_match_res:
                retinaface_pred_id = humanbboxs_facebboxs_match_res[index][0]
                mot_pred[track_id]["face_pred"] = retinaface_pred[retinaface_pred_id]
                mot_pred[track_id]["face_embedding"] = faces_embedding[retinaface_pred_id]
                mot_pred[track_id]["face_img"] = faces_ori[retinaface_pred_id]
                mot_pred[track_id]["face_definition"] = faces_definition[retinaface_pred_id]
            # add pose det res in mot_det
            pose_key_points = pose_det_pred[index]["keypoints"]
            mot_pred[track_id]["pose_key_points"] = pose_key_points
            # add person img definition
            mot_pred[track_id]["person_img_definition"] = person_imgs_definition[index]
            # add is_scalper flag
            is_scalper = self.scalper_judge(pose_key_points)
            mot_pred[track_id]["is_scalper"] = is_scalper
        return mot_pred

    def draw_scalper(self, frame, res):
        frame = frame.copy()
        for tracker_id in res:
            if res[tracker_id]["is_scalper"]:
                bbox = res[tracker_id]["bbox"]
                l, t, r, b = bbox
                cv2.rectangle(frame, pt1=(int(l), int(t)), pt2=(int(r), int(b)), color=(0, 0, 255), thickness=1)
                cv2.putText(frame, str(tracker_id), (int((l + r) // 2), int(t)), cv2.FONT_HERSHEY_SIMPLEX, 1,
                            (0, 0, 255), 2)
            else:
                bbox = res[tracker_id]["bbox"]
                l, t, r, b = bbox
                cv2.rectangle(frame, pt1=(int(l), int(t)), pt2=(int(r), int(b)), color=(0, 0, 255), thickness=1)
                cv2.putText(frame, str(tracker_id), (int((l + r) // 2), int(t)), cv2.FONT_HERSHEY_SIMPLEX, 1,
                            (0, 255, 255), 2)
        return frame

    def resize_imgs(self, img, test_input_size):
        H, W, _ = img.shape
        scale = test_input_size / max(H, W)
        resized_img = cv2.resize(img, dsize=(int(W * scale), int(H * scale)), interpolation=cv2.INTER_LINEAR)
        return resized_img, scale
