import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


def correct_track_id(global_features_produce,
                     suspected_scalper_results,
                     person_feature_lib,
                     person_feature_track_ids,
                     face_feature_lib,
                     face_feature_track_ids,
                     person_threshold=0.5,
                     face_threshold=0.6,
                     max_track_id_num=10000):

    # get local_global_match_dict
    local_global_match_dict = {}
    global_person_id_set = set()
    person_feature_track_ids_set = set(person_feature_track_ids)
    if len(person_feature_track_ids_set) > 0:
        for track_id in suspected_scalper_results:

            # person feature match cal
            person_features = np.array(suspected_scalper_results[track_id]["person_feature_list"])
            person_cossim = cosine_similarity(person_features, person_feature_lib)
            person_maximum = np.max(person_cossim)
            person_max_idx = list(np.where(person_cossim == person_maximum.max())[1])[0]

            # face feature match cal
            # face_features = np.array(
            #     [ff for ff in suspected_scalper_results[track_id]["face_embedding_list"] if ff is not None])
            # face_maximum = None
            # face_max_idx = None
            # if face_features.shape[0] >= 1 and face_feature_lib.shape[0] >= 1:
            #     face_cossim = cosine_similarity(face_features, face_feature_lib)
            #     face_maximum = np.max(face_cossim)
            #     face_max_idx = list(np.where(face_cossim == face_maximum.max())[1])[0]

            #  priority to face features
            # if face_maximum is not None:
            #     if face_maximum > face_threshold:
            #         local_global_match_dict[track_id] = int(face_feature_track_ids[face_max_idx])
            #     elif face_maximum <= face_threshold and person_maximum > person_threshold:
            #         local_global_match_dict[track_id] = int(person_feature_track_ids[person_max_idx])
            #     elif face_maximum <= face_threshold and person_maximum <= person_threshold and \
            #             len(person_feature_track_ids_set) < max_track_id_num:
            #         local_global_match_dict[track_id] = -1
            #     else:
            #         local_global_match_dict[track_id] = int(face_feature_track_ids[face_max_idx])
            # else:
            #     if person_maximum > person_threshold:
            #         local_global_match_dict[track_id] = int(person_feature_track_ids[person_max_idx])
            #     elif person_maximum <= person_threshold and len(person_feature_track_ids_set) < max_track_id_num:
            #         local_global_match_dict[track_id] = -1
            #     else:
            #         local_global_match_dict[track_id] = int(person_feature_track_ids[person_max_idx])

            #  priority to person features
            if person_maximum > person_threshold:
                global_track_id = int(person_feature_track_ids[person_max_idx])
                # handle the conflict of two same global_track_id in one frame
                if global_track_id not in global_person_id_set:
                    pass
                elif global_features_produce.track_id_next < max_track_id_num:
                    global_track_id = global_features_produce.track_id_next
                    global_features_produce.track_id_next += 1
                else:
                    raise RuntimeError("person feature library overflow!")
                local_global_match_dict[track_id] = global_track_id
                global_person_id_set.add(global_track_id)

            elif person_maximum <= person_threshold:
                # if face_maximum is not None and face_maximum > face_threshold:
                #     global_track_id = int(face_feature_track_ids[face_max_idx])
                #     # handle the conflict of two same global_track_id in one frame
                #     if global_track_id not in global_person_id_set:
                #         pass
                #     elif global_features_produce.track_id_next < max_track_id_num:
                #         global_track_id = global_features_produce.track_id_next
                #         global_features_produce.track_id_next += 1
                #     else:
                #         raise RuntimeError("person feature library overflow!")
                #     local_global_match_dict[track_id] = global_track_id
                #     global_person_id_set.add(global_track_id)

                # no matched person or face features, so we have to create a new track_id
                if global_features_produce.track_id_next < max_track_id_num:
                    global_track_id = global_features_produce.track_id_next
                    global_features_produce.track_id_next += 1
                    local_global_match_dict[track_id] = global_track_id
                    global_person_id_set.add(global_track_id)

                else:
                    raise RuntimeError("person feature library overflow!")

    # The situation of empty feature libs
    else:
        for track_id in suspected_scalper_results:
            if global_features_produce.track_id_next < max_track_id_num:
                global_track_id = global_features_produce.track_id_next
                global_features_produce.track_id_next += 1
                local_global_match_dict[track_id] = global_track_id
                global_person_id_set.add(global_track_id)
            else:
                raise RuntimeError("person feature library overflow!")

    # handle the conflict of two same global_track_id in one frame
    # global_person_id_set = set()
    # correct_local_global_match_dict = {}
    # for track_id in local_global_match_dict:
    #     global_track_id = local_global_match_dict[track_id]
    #     if global_track_id != -1 and global_track_id not in global_person_id_set:
    #         global_track_id = local_global_match_dict[track_id]
    #     else:
    #         # for not matched and conflict when matching
    #         global_track_id = global_features_produce.track_id_next
    #         global_features_produce.track_id_next += 1
    #     global_person_id_set.add(global_track_id)
    #     correct_local_global_match_dict[track_id] = global_track_id
    return local_global_match_dict
