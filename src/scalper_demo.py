# utf-8
import random
import cv2
import os
import datetime
import numpy as np
from one_segment_inference import one_segment_inference
import torch
from utils.global_features_produce import global_features_produce
from utils.get_global_track_id import correct_track_id
from utils.single_person import single_person
from utils.format_convert import numpy_to_base64
from configs.configs import demo_configs

# for logging
import time
import logging


class scalper_demo(one_segment_inference):
    def __init__(self):
        random.seed(demo_configs["seed"])
        self.demo_configs = demo_configs
        super().__init__(gpuid=self.demo_configs["gpu_id"],
                         read_fps=self.demo_configs["segment_fps"],
                         segment_time=self.demo_configs["segment_time"])
        self.global_features_produce = global_features_produce(
            self.demo_configs["featurelib_path"],
            max_track_id_num=self.demo_configs["max_track_id_num"],
            max_feature_num_every_track_id=self.demo_configs["max_feature_num_every_track_id"])
        self.appear_rate_threshold = self.demo_configs["appear_rate_threshold"]
        self.scalper_num_threshold = self.demo_configs["scalper_num_threshold"]
        self.person_threshold = self.demo_configs["person_threshold"]
        self.face_threshold = self.demo_configs["face_threshold"]

    # !!! can be changed by business needs and real scenario testing
    def get_suspected_scalper_results(self, one_segment_result):
        # get suspected_scalper_results, del low score track id
        suspected_scalper_results = {}
        for track_id in one_segment_result:
            if one_segment_result[track_id]["appear_rate"] >= self.appear_rate_threshold or \
                    sum(one_segment_result[track_id]["scalper_num"]) >= self.scalper_num_threshold:

                # add suspected scalper results
                suspected_scalper_results[track_id] = one_segment_result[track_id]

                # huangniu_confidence calculate
                if one_segment_result[track_id]["appear_rate"] >= self.appear_rate_threshold:
                    suspected_scalper_results[track_id]["huangniu_confidence"] = \
                        one_segment_result[track_id]["appear_rate"] + \
                        (1 - one_segment_result[track_id]["appear_rate"]) * \
                        sum(one_segment_result[track_id]["scalper_num"]) / \
                        len(one_segment_result[track_id]["scalper_num"])
                else:
                    suspected_scalper_results[track_id]["huangniu_confidence"] = \
                        self.appear_rate_threshold + \
                        (1 - self.appear_rate_threshold) * \
                        sum(one_segment_result[track_id]["scalper_num"]) / \
                        len(one_segment_result[track_id]["scalper_num"])
        return suspected_scalper_results

    def result_post_process(self, suspected_scalper_results):
        # get correct_local_global_match_dict = {"track_id": global_id}
        self.load_feature_libs()
        correct_local_global_match_dict = correct_track_id(
            self.global_features_produce,
            suspected_scalper_results,
            self.all_person_features,
            self.all_person_features_track_ids,
            self.all_face_features,
            self.all_face_features_track_ids,
            person_threshold=self.person_threshold,
            face_threshold=self.face_threshold,
            max_track_id_num=self.demo_configs["max_track_id_num"])

        # get global_suspected_scalper_results, difference is the id changed to global id
        global_suspected_scalper_results = {}
        output_results = {"video_segment_results": []}
        for track_id in correct_local_global_match_dict:
            global_person_id = correct_local_global_match_dict[track_id]
            huangniu_confidence = suspected_scalper_results[track_id]["huangniu_confidence"]

            # get max confidence face img if no face then set face img None
            face_img_list = [face_img for face_img in
                             suspected_scalper_results[track_id]["face_img_list"] if
                             face_img is not None]
            face_embedding_list = [face_embedding for face_embedding in
                                   suspected_scalper_results[track_id]["face_embedding_list"] if
                                   face_embedding is not None]
            face_definition_list = [face_definition for face_definition in
                                    suspected_scalper_results[track_id]["face_definition_list"] if
                                    face_definition is not None]
            face_confidence_list = [face_confidence for face_confidence in
                                    suspected_scalper_results[track_id]["face_confidence_list"] if
                                    face_confidence is not None]

            if len(face_img_list) > 0 and len(face_definition_list) > 0:
                f_index = np.argmax(np.array(face_confidence_list))
                face_definition = str(face_definition_list[f_index][0])
                face_img = face_img_list[f_index]
                face_img = numpy_to_base64(face_img)  # RGB
                face_embedding = face_embedding_list[f_index]
            else:
                face_img = None
                face_embedding = None
                face_definition = None

            # get max confidence full body img if no img then set full body img None
            person_img_list = suspected_scalper_results[track_id]["person_img_list"]
            person_feature_list = suspected_scalper_results[track_id]["person_feature_list"]
            person_img_definition_list = suspected_scalper_results[track_id]["person_img_definition_list"]
            person_confidence_list = suspected_scalper_results[track_id]["person_confidence_list"]
            p_index = np.argmax(np.array(person_confidence_list))
            person_img_definition = str(person_img_definition_list[p_index][0])
            full_body_image = person_img_list[p_index]
            full_body_image = numpy_to_base64(full_body_image)  # RGB
            person_feature = person_feature_list[p_index]

            # get person occurrence info
            person_occurrence = {
                "start_time": suspected_scalper_results[track_id]["person_datetime_list"][0],
                "end_time": suspected_scalper_results[track_id]["person_datetime_list"][-1],
                "duration": suspected_scalper_results[track_id]["person_time_list"][-1] -
                            suspected_scalper_results[track_id]["person_time_list"][0]
            }

            # get raise hand action info
            raise_hand_action = {
                "num": sum(suspected_scalper_results[track_id]["scalper_num"]),
                "duration": sum(suspected_scalper_results[track_id]["scalper_num"]) / \
                            (self.demo_configs["segment_fps"]),
            }

            video_segment_results_item = {"global_person_id": str(global_person_id),
                                          "huangniu_confidence": huangniu_confidence,
                                          "face_img": face_img,
                                          "face_embedding": face_embedding,
                                          "face_definition": face_definition,
                                          "full_body_image": full_body_image,
                                          "person_img_definition": person_img_definition,
                                          "person_feature": person_feature,
                                          "person_occurrence": person_occurrence,
                                          "raise_hand_action": raise_hand_action}
            output_results["video_segment_results"].append(video_segment_results_item)
            global_suspected_scalper_results[global_person_id] = suspected_scalper_results[track_id]

        # get conclusion attr
        if len(output_results["video_segment_results"]) > 0:
            output_results["conclusion"] = 1
        else:
            output_results["conclusion"] = 0

        return global_suspected_scalper_results, output_results

    def load_feature_libs(self):
        self.all_person_features, self.all_person_features_track_ids = self.global_features_produce.read_all_person_features()
        self.all_face_features, self.all_face_features_track_ids = self.global_features_produce.read_all_face_features()
        all_person_features_track_ids = list(map(int, self.all_person_features_track_ids))
        if len(all_person_features_track_ids) > 0:
            self.global_features_produce.track_id_next = max(map(int, self.all_person_features_track_ids)) + 1

    def add_features(self, output_results):
        for video_segment_results_item in output_results["video_segment_results"]:
            global_person_id = video_segment_results_item["global_person_id"]
            person_feature = video_segment_results_item["person_feature"]
            face_embedding = video_segment_results_item["face_embedding"]
            # add features
            self.global_features_produce.add_person_feature(global_person_id, person_feature)
            if face_embedding is not None:
                self.global_features_produce.add_face_feature(global_person_id, face_embedding)

    def draw_one_tracker(self, frame, global_tracker_id, bbox, is_scalper):
        if is_scalper:
            # draw raise hand suspected scalper
            l, t, r, b = bbox
            cv2.rectangle(frame, pt1=(int(l), int(t)), pt2=(int(r), int(b)), color=(0, 0, 255), thickness=1)
            cv2.putText(frame, str(global_tracker_id), (int((l + r) // 2), int(t)), cv2.FONT_HERSHEY_SIMPLEX, 1,
                        (0, 0, 255), 2)
        else:
            # draw normal suspected scalper
            l, t, r, b = bbox
            cv2.rectangle(frame, pt1=(int(l), int(t)), pt2=(int(r), int(b)), color=(0, 0, 255), thickness=1)
            cv2.putText(frame, str(global_tracker_id), (int((l + r) // 2), int(t)), cv2.FONT_HERSHEY_SIMPLEX, 1,
                        (0, 255, 255), 2)
        return frame

    def visualize(self, out_root, frames, global_one_segment_result):
        if not os.path.exists(out_root):
            os.makedirs(out_root)

        fps = self.segment_read.read_fps
        size = (frames[0].shape[1], frames[0].shape[0])
        video_name = datetime.datetime.now().strftime("%Y%m%d%H%M%S") \
                     + "_" + str(random.randint(0, 100)).zfill(4) + '.mp4'
        video_path = os.path.join(out_root, video_name)
        videowrite = cv2.VideoWriter(video_path,
                                     cv2.VideoWriter_fourcc('m', 'p', '4', 'v'),
                                     fps,
                                     size)

        persons = [single_person(global_tracker_id,
                                 global_one_segment_result[global_tracker_id]["person_frameid_list"],
                                 global_one_segment_result[global_tracker_id]["person_bboxxyxy_list"],
                                 global_one_segment_result[global_tracker_id]["person_is_scalper"])
                   for global_tracker_id in global_one_segment_result]

        for frame_idx, frame in enumerate(frames):
            for person in persons:
                if person.get_head_frame_id() == frame_idx:
                    _, bbox, is_scalper = person.get_head_attrs()
                    frame = self.draw_one_tracker(frame, person.id, bbox, is_scalper)
                    person.del_head_attrs()
            videowrite.write(frame)
        return video_path

    def demo(self, url, visual=False, video_root=''):
        output_results = {}
        one_segment_result, camera_url, time_range, frames = self.seg_inference(url, visual=visual)
        output_results["time_range"] = time_range
        output_results["video_url"] = camera_url
        # print(list(one_segment_result.keys()))
        logging.info("camera url: {}".format(camera_url))
        logging.info("time range: {} -- {}".format(time_range[0], time_range[1]))
        logging.info("ori tracker ids: {}". \
                     format(" ".join(list(map(str, one_segment_result.keys())))))

        # suspected_scalper_results, with single segment local track id
        suspected_scalper_results = self.get_suspected_scalper_results(one_segment_result)
        # print(list(suspected_scalper_results.keys()))
        logging.info("suspected scalper tracker ids: {}". \
                     format(" ".join(list(map(str, suspected_scalper_results.keys())))))

        # global_one_segment_result, with global track id, it is not very accurate, because of the video quality and reid model
        global_one_segment_result, output_results_sub = self.result_post_process(suspected_scalper_results)
        output_results.update(output_results_sub)
        # print(list(global_one_segment_result.keys()))
        logging.info("global suspected scalper tracker ids: {}". \
                     format(" ".join(list(map(str, global_one_segment_result.keys())))))

        logging.info("adding features")
        self.add_features(output_results)
        logging.info("features added down: {}".format(self.demo_configs["featurelib_path"]))

        if visual and len(frames) > 0:
            if len(global_one_segment_result) > 0:
                logging.info("visualize results")
                video_path = self.visualize(video_root, frames, global_one_segment_result)
                output_results["visual_path"] = video_path
                logging.info("visualize down: {}".format(video_path))
            else:
                logging.info("no suspected scalper!")
                video_path = ""
                output_results["visual_path"] = video_path

        torch.cuda.empty_cache()
        return output_results


if __name__ == "__main__":
    url_list = ["/workspace/huangniu_det/ed40384e00b942ce04da37bd73208082.mp4"] * 2
    #url_list = ["/workspace/huangniu_det/test_shake_hand.mkv"]

    s_d = scalper_demo()
    for i in range(len(url_list)):
        output_results = s_d.demo(url_list[i], visual=True, video_root="/workspace/huangniu_det/visual")
        print(output_results)
