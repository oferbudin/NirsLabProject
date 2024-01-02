import sys
import numpy as np
from typing import Dict, List

import scipy

from NirsLabProject.config.consts import *
from NirsLabProject.config.subject import Subject
from NirsLabProject.utils import general_utils as utils


# Python 3 program to merge K sorted arrays of size N each.
# A Min heap node
class MinHeapNode:
    element = 0

    # The element to be stored index of the array from which the element is taken
    i = 0

    # index of the next element to be picked from array
    j = 0

    def __init__(self, element, i, j):
        self.element = element
        self.i = i
        self.j = j


# A class for Min Heap
class MinHeap:
    harr = None

    # Array of elements in heap
    heap_size = 0

    # Current number of elements in min heap
    # Constructor: Builds a heap from
    # a given array a[] of given size
    def __init__(self, a, size):
        self.heap_size = size
        self.harr = a
        i = int((self.heap_size - 1) / 2)
        while (i >= 0):
            self.MinHeapify(i)
            i -= 1

    # A recursive method to heapify a subtree with the root at given index
    # This method assumes that the subtrees are already heapified
    def MinHeapify(self, i):
        l = self.left(i)
        r = self.right(i)
        smallest = i
        if (l < self.heap_size and self.harr[l].element[0] < self.harr[i].element[0]):
            smallest = l
        if (r < self.heap_size and self.harr[r].element[0] < self.harr[smallest].element[0]):
            smallest = r
        if (smallest != i):
            self.swap(self.harr, i, smallest)
            self.MinHeapify(smallest)

    # to get index of left child of node at index i
    def left(self, i):
        return (2 * i + 1)

    # to get index of right child of node at index i
    def right(self, i):
        return (2 * i + 2)

    # to get the root
    def getMin(self):
        if (self.heap_size <= 0):
            print("Heap underflow")
            return None
        return self.harr[0]

    # to replace root with new node "root" and heapify() new root
    def replaceMin(self, root):
        self.harr[0] = root
        self.MinHeapify(0)

    # A utility function to swap two min heap nodes
    def swap(self, arr, i, j):
        temp = arr[i]
        arr[i] = arr[j]
        arr[j] = temp

    # A utility function to print array elements
    @staticmethod
    def printArray(arr):
        for i in arr:
            print(str(i) + " ", end="")
        print()

    # This function takes an array of arrays as an argument and All
    # arrays are assumed to be sorted.
    # It merges them together and prints the final sorted output.
    @staticmethod
    def mergeKSortedArrays(arr, K):
        hArr = [None] * (K)
        resultSize = 0
        i = 0
        while (i < len(arr)):
            node = MinHeapNode(arr[i][0], i, 1)
            hArr[i] = node
            resultSize += len(arr[i])
            i += 1

        # Create a min heap with k heap nodes. Every heap
        # node has first element of an array
        mh = MinHeap(hArr, K)
        result = [0] * (resultSize)

        # To store output array
        # Now one by one get the minimum element from min
        # heap and replace it with next element of its
        # array
        i = 0
        while (i < resultSize):

            # Get the minimum element and store it in
            # result
            root = mh.getMin()
            result[i] = root.element

            # Find the next element that will replace
            # current root of heap. The next element
            # belongs to same array as the current root.
            if (root.j < len(arr[root.i])):
                root.element = arr[root.i][root.j]
                root.j += 1
            else:
                root.element = [sys.maxsize, -1]

            # Replace root with next element of array
            mh.replaceMin(root)
            i += 1
        return np.asarray(result)


class Group:
    def __init__(self, group: List[np.ndarray], group_index: int, index_to_channel: Dict[int, str]):
        self._group = group
        self.index = int(group_index)
        self.size = len(group)
        self.fist_event_timestamp = group[0][TIMESTAMP_INDEX]
        self.last_event_timestamp = group[-1][TIMESTAMP_INDEX]
        self.group_event_duration = self.last_event_timestamp - self.fist_event_timestamp
        self.index_to_channel = index_to_channel

        # sorting all the timestamps with the same timestamp of the first event by amlitude
        self.focal_channel = sorted(
            [spike for spike in group if spike[0] == self.fist_event_timestamp],
            key=lambda x: (x[1], x[2])
        )[-1]
        self.focal_channel_index = self.focal_channel[CHANNEL_INDEX]
        self.focal_channel_name = index_to_channel[self.focal_channel_index]
        self.focal_channel_amplitude = self.focal_channel[AMPLITUDE_INDEX]

        self.hemispheres = set()
        self.structures = set()

        _electrode_depths = set()
        for record in group:
            channel = index_to_channel[record[1]]
            electrode_name, contact_number = utils.extract_channel_name_and_contact_number(channel)
            _electrode_depths.add(contact_number)
            # adds the hemisphere 'L' or 'R' and the structure 'HPC', 'EC' etc...
            self.hemispheres.add(electrode_name[0])
            self.structures.add(electrode_name)

        self.deepest_electrode = min(_electrode_depths)
        self.shallowest_electrode = max(_electrode_depths)

        self.group_spatial_spread = self.calculate_group_spatial_spread()

    def calculate_group_spatial_spread(self):

        def polygon_area(poly):
            # shape (N, 3)
            if isinstance(poly, list):
                poly = np.array(poly)
            # all edges
            edges = poly[1:] - poly[0:1]
            # row wise cross product
            cross_product = np.cross(edges[:-1], edges[1:], axis=1)
            # area of all triangles
            area = np.linalg.norm(cross_product, axis=1) / 2
            return sum(area)

        pts = np.array([g[CORD_X_INDEX:CORD_Z_INDEX + 1] for g in self._group if not np.isnan(g[CORD_X_INDEX])])
        value = -1
        if pts.shape[0] > 0:
            if pts.shape[0] == 1:
                value = 0
            elif pts.shape[0] == 2:
                value = np.linalg.norm(pts[0] - pts[1])
            elif pts.shape[0] == 3:
                value = polygon_area(pts)
            else:
                value = polygon_area(pts)
        if value > 2000:
            value = 2000
        return value

    def get_features(self):
        return np.asarray([self.size, self.group_event_duration, self.deepest_electrode, self.shallowest_electrode])

    def print_group_electrodes(self):
        for record in self._group:
            print(self.index_to_channel[record[CHANNEL_INDEX]])

    def __str__(self):
        return f'Group size {self.size} | Focal: {self.focal_channel_name} | Hemisphers: {self.hemispheres} | Stractures: {self.structures} | Time Difrences: {self.group_event_duration}'


def group_spikes(subject: Subject, channels_spikes_features: Dict[str, np.ndarray], index_to_channel: Dict[int, str]):
    group_objets = []
    print('Grouping spikes')

    # Merge all the spikes into one sorted array
    all_spikes = [spikes for spikes in channels_spikes_features.values() if spikes.shape[0] > 0]
    all_spikes_flat = MinHeap.mergeKSortedArrays(all_spikes, len(all_spikes))

    # Group the timestamps based on the window_width
    groups_list = []
    group_index_to_group = {}
    group = [all_spikes_flat[TIMESTAMP_INDEX]]
    for i in range(1, all_spikes_flat.shape[0]):
        # If the next spike is in the window add it to the group
        if group[0][TIMESTAMP_INDEX] + SPIKES_GROUPING_WINDOW_SIZE > all_spikes_flat[i][TIMESTAMP_INDEX]:
            group.append(all_spikes_flat[i])
        else:
            # window is over, start a new group
            groups_list.append(group)
            group = [all_spikes_flat[i]]
    # Add the last group
    groups_list.append(group)

    spike_index = 0
    all_spikes_group_indexes = np.zeros(all_spikes_flat.shape[0], dtype=int)
    all_spikes_group_focal_point = np.zeros(all_spikes_flat.shape[0], dtype=int)
    all_spikes_group_event_duration = np.zeros(all_spikes_flat.shape[0], dtype=int)
    all_spikes_group_event_size = np.zeros(all_spikes_flat.shape[0], dtype=int)
    all_spikes_group_deepest_electrode = np.zeros(all_spikes_flat.shape[0], dtype=int)
    all_spikes_group_shallowest_electrode = np.zeros(all_spikes_flat.shape[0], dtype=int)
    all_spikes_group_group_spatial_spread = np.zeros(all_spikes_flat.shape[0], dtype=int)
    all_spikes_group_focal_point_amplitude = np.zeros(all_spikes_flat.shape[0], dtype=float)
    # Create a group object for each group
    for group_index, group in enumerate(groups_list):
        subject_group_index = f'{subject.p_number}_{group_index}'
        group = Group(group, subject_group_index, index_to_channel)
        group_objets.append(group)
        group_index_to_group[subject_group_index] = group
        for i in range(group.size):
            # Add the group index to the spikes features
            all_spikes_group_indexes[spike_index] = group.index
            all_spikes_group_focal_point[spike_index] = group.focal_channel_index
            all_spikes_group_event_duration[spike_index] = group.group_event_duration
            all_spikes_group_event_size[spike_index] = group.size
            all_spikes_group_deepest_electrode[spike_index] = group.deepest_electrode
            all_spikes_group_shallowest_electrode[spike_index] = group.shallowest_electrode
            all_spikes_group_group_spatial_spread[spike_index] = group.group_spatial_spread
            all_spikes_group_focal_point_amplitude[spike_index] = group.focal_channel_amplitude
            spike_index += 1

    # Add the group index to the spikes array
    all_spikes_flat = np.concatenate(
        (
            all_spikes_flat,
            all_spikes_group_indexes.reshape((-1, 1)),
            all_spikes_group_focal_point.reshape((-1, 1)),
            all_spikes_group_event_duration.reshape((-1, 1)),
            all_spikes_group_event_size.reshape((-1, 1)),
            all_spikes_group_deepest_electrode.reshape((-1, 1)),
            all_spikes_group_shallowest_electrode.reshape((-1, 1)),
            all_spikes_group_group_spatial_spread.reshape((-1, 1))
        ),
        axis=1
    )

    return group_index_to_group, all_spikes_flat, all_spikes_group_focal_point_amplitude, group_objets
