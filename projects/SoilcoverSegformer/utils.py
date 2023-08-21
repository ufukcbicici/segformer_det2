import itertools
import os
import pickle
from functools import lru_cache

import cv2


class Utils:

    @staticmethod
    def get_files_under_folder(root_path):
        file_path_list = []
        assert os.path.isdir(root_path)
        files = os.listdir(root_path)
        files = [f for f in files if os.path.isfile(os.path.join(root_path, f))]
        for file in files:
            file_path = os.path.join(root_path, file)
            file_path_list.append(file_path)
        return file_path_list

    @staticmethod
    def ref_conversion(coord, direction):
        direction_mult = {'n': 1, 's': -1, 'e': 1, 'w': -1, 'north': 1, 'south': -1, 'east': 1, 'west': -1}
        if direction in direction_mult:
            return coord * direction_mult[direction]
        else:
            raise ValueError("Expected coordinate reference to be in {0}, "
                             "but found '{1}' instead.".format(direction_mult.keys(), direction))

    @staticmethod
    def pickle_save_to_file(path, file_content):
        f = open(path, "wb")
        pickle.dump(file_content, f)
        f.close()

    @staticmethod
    def pickle_load_from_file(path):
        f = open(path, "rb")
        content = pickle.load(f)
        f.close()
        return content

    @staticmethod
    def divide_array_into_chunks(arr, count):
        chunk_size = len(arr) // count
        curr_index = 0
        list_of_chunks = []
        for idx in range(count):
            if idx != count - 1:
                list_of_chunks.append(arr[curr_index:curr_index + chunk_size])
            else:
                list_of_chunks.append(arr[curr_index:])
            curr_index += chunk_size
        return list_of_chunks

    @staticmethod
    def get_cartesian_product(list_of_lists):
        cartesian_product = list(itertools.product(*list_of_lists))
        return cartesian_product

    @staticmethod
    @lru_cache(maxsize=1000)
    def open_image_cached(image_path):
        img_array = cv2.imread(filename=image_path)
        return img_array

