import torch

from single_frame_inference_bytemot_yolox import single_frame_inference
from utils.segment_reader import segment_reader


class one_segment_inference(single_frame_inference):
    def __init__(self, gpuid='0', read_fps=2, segment_time=120):
        device = torch.device('cuda:{}'.format(gpuid) if torch.cuda.is_available() else 'cpu')
        super().__init__(device=device)
        self.one_segment_result = {}
        self.segment_read = segment_reader(read_fps=read_fps, segment_time=segment_time)
        #self.input_size = input_size

    def reset(self):
        self.one_segment_result.clear()
        self.segment_read.reset()
        self.mot.reset()

    def add_res(self, data, frame_res):
        if len(frame_res) > 0:
            for track_id in frame_res:
                if track_id not in self.one_segment_result:
                    self.one_segment_result[track_id] = {
                        "person_time_list": [data["time"]],
                        "person_datetime_list": [data["datetime"]],
                        "person_frameid_list": [data["frame_id"]],
                        "person_bboxxyxy_list": [frame_res[track_id]["bbox"]],
                        "person_confidence_list": [frame_res[track_id]["confidence"]],
                        "person_img_list": [frame_res[track_id]["person_img"]],
                        "person_feature_list": [frame_res[track_id]["feature"]],
                        "person_img_definition_list": [frame_res[track_id]["person_img_definition"]],
                        "person_is_scalper": [frame_res[track_id]["is_scalper"]],
                        "face_frameid_list": [data["frame_id"] \
                                                  if "face_pred" in frame_res[track_id] else -1],
                        "face_bboxxyxy_list": [frame_res[track_id]["face_pred"][:4] \
                                                   if "face_pred" in frame_res[track_id] else None],
                        "face_confidence_list": [frame_res[track_id]["face_pred"][4] \
                                                     if "face_pred" in frame_res[track_id] else None],
                        "face_embedding_list": [frame_res[track_id]["face_embedding"] \
                                                    if "face_embedding" in frame_res[track_id] else None],
                        "face_img_list": [frame_res[track_id]["face_img"] \
                                              if "face_img" in frame_res[track_id] else None],
                        "face_definition_list": [frame_res[track_id]["face_definition"] \
                                                     if "face_definition" in frame_res[track_id] else None],
                    }
                else:
                    self.one_segment_result[track_id]["person_time_list"].append(data["time"])
                    self.one_segment_result[track_id]["person_datetime_list"].append(data["datetime"])
                    self.one_segment_result[track_id]["person_frameid_list"].append(data["frame_id"])
                    self.one_segment_result[track_id]["person_bboxxyxy_list"].\
                        append(frame_res[track_id]["bbox"])
                    self.one_segment_result[track_id]["person_confidence_list"].\
                        append(frame_res[track_id]["confidence"])
                    self.one_segment_result[track_id]["person_img_definition_list"].\
                        append(frame_res[track_id]["person_img_definition"]),
                    self.one_segment_result[track_id]["person_img_list"].\
                        append(frame_res[track_id]["person_img"])
                    self.one_segment_result[track_id]["person_feature_list"].\
                        append(frame_res[track_id]["feature"])
                    self.one_segment_result[track_id]["person_is_scalper"].\
                        append(frame_res[track_id]["is_scalper"])
                    self.one_segment_result[track_id]["face_frameid_list"].\
                        append(data["frame_id"] if "face_pred" in frame_res[track_id] else -1)
                    self.one_segment_result[track_id]["face_bboxxyxy_list"].\
                        append(frame_res[track_id]["face_pred"][:4] if "face_pred" in frame_res[track_id] else None)
                    self.one_segment_result[track_id]["face_confidence_list"].\
                        append(frame_res[track_id]["face_pred"][4] if "face_pred" in frame_res[track_id] else None)
                    self.one_segment_result[track_id]["face_embedding_list"].\
                        append(frame_res[track_id]["face_embedding"] if "face_embedding" in frame_res[track_id] else None)
                    self.one_segment_result[track_id]["face_img_list"].\
                        append(frame_res[track_id]["face_img"] if "face_img" in frame_res[track_id] else None)
                    self.one_segment_result[track_id]["face_definition_list"].\
                        append(frame_res[track_id]["face_definition"] if "face_definition" in frame_res[track_id] else None)

    def segment_scalper_judge(self):
        frames_all = self.segment_read.put_frame_num
        for track_id in self.one_segment_result:
            person_frames = self.one_segment_result[track_id]["person_frameid_list"][-1] - \
                            self.one_segment_result[track_id]["person_frameid_list"][0]
            appear_rate = person_frames / frames_all
            scalper_num = self.one_segment_result[track_id]["person_is_scalper"]
            self.one_segment_result[track_id]["appear_rate"] = appear_rate
            self.one_segment_result[track_id]["scalper_num"] = scalper_num

    def seg_inference(self, video_url, visual=False):
        self.reset()

        seg_start_time = None
        seg_end_time = None
        camera_url = None
        # resized_frames = []
        frames = []

        self.segment_read.start(video_url)
        while True:
            # get frame
            data = self.segment_read.frame_get()
            if data is not None:
                # frame resize, get resize ratio
                # resized_img, ratio = self.resize_imgs(data["frame"], self.input_size)
                frame = data["frame"]

                if data["frame_id"] == 0:
                    seg_start_time = data["datetime"]
                    camera_url = data["url"]

                # inference one frame
                # inference_data = {
                #     "resize_frame": resized_img,
                #     "raw_frame": data["frame"],
                #     "ratio": ratio,
                # }
                frame_res = self.inference(frame)

                # if visual, collect resized_frames
                if visual:
                    # resized_frames.append(resized_img)
                    frames.append(frame)

                # collect one segment results
                self.add_res(data, frame_res)
                seg_end_time = data["datetime"]
                print("---------frame {} produced".format(data["frame_id"]))
            else:
                break

        self.segment_read.join()
        self.segment_scalper_judge()

        return self.one_segment_result, camera_url, [seg_start_time, seg_end_time], frames


if __name__ == "__main__":
    o_s_i = one_segment_inference(gpuid='0')
    video_path = "/workspace/huangniu_det/test_shake_hand.mkv"
    o_s_i.seg_inference(video_path)
