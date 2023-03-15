# coding=utf-8
import h5py
import os
import numpy as np


class global_features_produce(object):
    def __init__(self, featurelib_path, max_track_id_num=10000, max_feature_num_every_track_id=10):
        self.max_track_id_num = max_track_id_num
        self.max_feature_num_every_track_id = max_feature_num_every_track_id
        self.track_id_next = 0
        self.featurelib_path = featurelib_path

        if not os.path.exists(featurelib_path):
            os.makedirs(featurelib_path)

        if not os.path.exists(os.path.join(featurelib_path, "feature.hdf5")):
            with h5py.File(os.path.join(featurelib_path, "feature.hdf5"), "a") as f:
                f.create_group("person_feature")
                f.create_group("face_feature")

    def add_face_feature(self, track_id, feature):
        # track_id, str
        # feature, ndarray
        with h5py.File(os.path.join(self.featurelib_path, "feature.hdf5"), "a") as f:
            if track_id not in f["face_feature"].keys():
                f["face_feature"].create_group(track_id).create_dataset("0", data=feature)
            else:
                len_face_feature_group_t_i = len(f["face_feature"][track_id])
                if len_face_feature_group_t_i < self.max_feature_num_every_track_id:
                    f["face_feature"][track_id].create_dataset(str(len_face_feature_group_t_i), data=feature)

    def add_person_feature(self, track_id, feature):
        # track_id, str
        # feature, ndarray
        with h5py.File(os.path.join(self.featurelib_path, "feature.hdf5"), "a") as f:
            if track_id not in f["person_feature"].keys():
                f["person_feature"].create_group(track_id).create_dataset("0", data=feature)
            else:
                len_person_features_group_t_i = len(f["person_feature"][track_id])
                if len_person_features_group_t_i < self.max_feature_num_every_track_id:
                    f["person_feature"][track_id].create_dataset(str(len_person_features_group_t_i), data=feature)

    def read_all_face_features(self):
        with h5py.File(os.path.join(self.featurelib_path, "feature.hdf5"), "a") as f:
            all_face_features_track_ids = []
            all_face_features = []
            for track_id in f["face_feature"]:
                all_face_features_track_ids += len(f["face_feature"][track_id]) * [track_id]
            for track_id in f["face_feature"]:
                all_face_features += list(f["face_feature"][track_id].values())
            all_face_features = np.array(all_face_features)
            return all_face_features, all_face_features_track_ids

    def read_all_person_features(self):
        with h5py.File(os.path.join(self.featurelib_path, "feature.hdf5"), "a") as f:
            all_person_features_track_ids = []
            all_person_features = []
            for track_id in f["person_feature"]:
                all_person_features_track_ids += len(f["person_feature"][track_id]) * [track_id]
            for track_id in f["person_feature"]:
                all_person_features += list(f["person_feature"][track_id].values())
            all_person_features = np.array(all_person_features)
            return all_person_features, all_person_features_track_ids
