class single_person(object):
    def __init__(self, id, frame_idx_list, bbox_list, is_scalper_list):
        self.id = id
        self.frame_idx_list = frame_idx_list
        self.bbox_list= bbox_list
        self.is_scalper_list = is_scalper_list

    def get_head_frame_id(self):
        if len(self.frame_idx_list) > 0:
            return self.frame_idx_list[0]
        else:
            return -1

    def get_head_attrs(self):
        return self.frame_idx_list[0], self.bbox_list[0], self.is_scalper_list[0]

    def del_head_attrs(self):
        self.frame_idx_list.pop(0)
        self.bbox_list.pop(0)
        self.is_scalper_list.pop(0)
