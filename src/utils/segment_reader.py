import queue
import threading
import cv2
import time
import datetime


class segment_reader(object):
    def __init__(self, read_fps=2, segment_time=120):
        self.q = queue.Queue()
        self.read_fps = read_fps
        self.segment_frame_num = segment_time * read_fps
        self.put_lock = threading.Lock()
        self.get_lock = threading.Lock()
        self.put_frame_num = 0
        self.get_frame_num = 0
        self.suffix_lib = ["mp4", "avi", "flv", "mkv"]

    def reset(self):
        self.q.queue.clear()
        self.put_frame_num = 0
        self.get_frame_num = 0

    def _get_video_stream(self, video_url):
        cap = cv2.VideoCapture(video_url)
        assert cap.isOpened(), "{} not open!".format(video_url)

        raw_fps = cap.get(cv2.CAP_PROP_FPS)
        ret = True
        prev_time = time.time()
        while ret and self.put_frame_num < self.segment_frame_num:
            try:
                ret, frame = cap.read()
                channel = frame.shape[2]
            except Exception as e:
                continue

            time_elapsed = time.time() - prev_time
            print("time_elapsed/time_per_frame: {}/{}".format(time_elapsed, 1. / self.read_fps))
            if time_elapsed > 1. / self.read_fps:
                prev_time = time.time()
                # put frame into queue
                with self.put_lock:
                    data = {
                        "frame": frame,
                        "frame_id": self.put_frame_num,
                        "datetime": datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f'),
                        "time": time.time(),
                        "url": video_url,
                    }
                    self.q.put(data)
                    self.put_frame_num += 1
                    print("reading frame {}".format(self.put_frame_num))
            time.sleep(1. / raw_fps)
        cap.release()

    def frame_get(self, out_time_thr=2):
        # handle the situation of qsize==0
        if self.q.qsize() == 0:
            start_time = time.time()
            while True:
                if time.time() - start_time > out_time_thr:
                    return None
                elif time.time() - start_time <= out_time_thr and self.q.qsize() == 0:
                    time.sleep(0.5)
                else:
                    break
        # get data
        with self.get_lock:
            data = self.q.get()
            self.get_frame_num += 1
            return data

    def start(self, video_url, out_time_thr=10):
        """Start frame put thread"""
        self.read_thread = threading.Thread(
            target=self._get_video_stream,
            args=(video_url,),
            name='frameput-Thread',
            daemon=True)
        self.read_thread.start()
        self.wait_url_connect(out_time_thr=out_time_thr)

    def join(self):
        self.read_thread.join()

    def wait_url_connect(self, out_time_thr=10):
        start_time = time.time()
        while True:
            if self.q.qsize() > 0:
                break
            elif time.time() - start_time > out_time_thr:
                raise RuntimeError("url connect out of time")
            else:
                time.sleep(0.5)


if __name__ == "__main__":
    read_fps = 10
    segment_time = 20
    video_path = "http://192.168.1.135:8000/live/B9191252E95617A9479B8F85D16BE56F.flv"
    # video_path = "/workspace/huangniu_det/test_shake_hand.mkv"
    for i in range(3):
        out_path = "./{}.mp4".format(i)
        videowrite = cv2.VideoWriter(out_path,
                                     cv2.VideoWriter_fourcc('m', 'p', '4', 'v'),
                                     read_fps,
                                     (1280, 720))
        s_r = segment_reader(read_fps=read_fps, segment_time=segment_time)
        s_r.start(video_path)
        s_r.wait_url_connect(out_time_thr=20)

        while True:
            data = s_r.frame_get()
            if data is not None:
                print("-------------get frame {}".format(data["frame_id"]))
                videowrite.write(data["frame"])
            else:
                break
        s_r.join()
    # s_r._frame_url_put(video_path)
    # i=0
    # while True:
    #     # get frame
    #     data = s_r.frame_get()
    #     if data is not None:
    #         print(i)
    #         i+=1
