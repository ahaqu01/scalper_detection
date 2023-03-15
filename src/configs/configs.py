demo_configs = {
    "seed": 10,
    "gpu_id": "0",
    "segment_time": 120,  # second
    "segment_fps": 2,
    #"segment_max_length": 1280,
    "max_track_id_num": 50000,
    "max_feature_num_every_track_id": 20,
    "appear_rate_threshold": 0.8,
    "scalper_num_threshold": 10,
    "person_threshold": 0.7,
    "face_threshold": 0.75,
    "img_definition_threshold": 67,
    "featurelib_path": "/workspace/huangniu_det/Featurelibrary",
}

single_frame_inference = {
    "model_weight_paths": {
        "facenet": "/workspace/huangniu_det/src/models/facenet/src/weights/20180402-114759-vggface2.pt",
        "mot": "/workspace/huangniu_det/src/models/bytemot/src/weights/model_final.pth",
        "pose_det": "/workspace/huangniu_det/src/models/pose_det/src/weights/hrnet_w32_coco_256x192-c78dce93_20200708.pth",
        "retinafce": "/workspace/huangniu_det/src/models/retinaface/src/weights/mobilenet0.25_Final.pth",
        "yolox": "/workspace/huangniu_det/src/models/yolox/src/weights/bytetrack_m_mot17.pth.tar",
        "img_definition": "/workspace/huangniu_det/src/models/image_definition_det/src/weights/musiq_paq2piq_ckpt-364c0c84_2.pth",
    },
    "config_paths": {
        "retinaface": "mobile0.25",
        "yolox": "/workspace/huangniu_det/src/models/yolox/src/configs/yolox_m_mix_det.yaml",
        "pose_det": "/workspace/huangniu_det/src/models/pose_det/src/configs/hrnet_w32_coco_256x192.py",
        "mot": {
            "extractor": "/workspace/huangniu_det/src/models/bytemot/src/configs/config-test.yaml",
            "tracker": "/workspace/huangniu_det/src/models/bytemot/src/configs/tracker.yaml",
        },
        "img_definition": "/workspace/huangniu_det/src/models/image_definition_det/src/configs/musiq.yaml",
    },
    "other_cfgs": {
        "retinafce": {
            "confidence_threshold": 0.5,
            "top_k": 5000,
            "nms_threshold": 0.4,
            "keep_top_k": 100,
            "vis_thres": 0.5,
        },
        "yolox": {
            "input_size": (800, 1440)
        },
        "stand_threshold": 0.8,
        "elbow_raising_threshold": 0.30,
        "arm_extension_threshold": 0.7,
        "kpt_score_thr": 0.1,
    }
}
