import sys
import numpy as np
from typing import Dict, List

from NirsLabProject.config.consts import *


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
        self.index = group_index
        self.size = len(group)
        self.fist_event_timestamp = group[0][TIMESTAMP_INDEX]
        self.last_event_timestamp = group[-1][TIMESTAMP_INDEX]
        self.group_event_duration = self.last_event_timestamp - self.fist_event_timestamp

        # sorting all the timestamps with the same timestamp of the first event by amlitude
        self.focal_channnel_index = sorted(
            [spike for spike in group if spike[0] == self.fist_event_timestamp],
            key=lambda x: (x[1], x[2])
        )[-1][1]
        self.focal_channnel_name = index_to_channel[self.focal_channnel_index]

        self.hemispheres = set()
        self.structures = set()

        _electrode_depths = set()
        for record in group:
            channel = index_to_channel[record[1]]
            _electrode_depths.add(channel[-1])
            # adds the hemisphere 'L' or 'R' and the structure 'HPC', 'EC' etc...
            self.hemispheres.add(channel[0])
            self.structures.add(channel[:-1])

        self.deepest_electrode = min(_electrode_depths)
        self.shallowest_electrode = max(_electrode_depths)


    def calculate_group_spatial_spread(self):
        # TODO: Implement
        pass

    def get_features(self):
        return np.asarray([self.size, self.group_event_duration, self.deepest_electrode, self.shallowest_electrode])

    def __str__(self):
        return f'Group size {self.size} | Focal: {self.focal_channnel_name} | Hemisphers: {self.hemispheres} | Stractures: {self.structures} | Time Difrences: {self.group_event_duration}'


def group_spikes(channels_spikes_features: Dict[str, np.ndarray]):
    index_to_channel = {}
    for i, channel_name in enumerate(channels_spikes_features.keys()):
        index_to_channel[i] = channel_name

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
            group = [all_spikes_flat[i, :]]
    # Add the last group
    groups_list.append(group)

    spike_index = 0
    all_spikes_group_indexes = np.zeros(all_spikes_flat.shape[0], dtype=int)
    # Create a group object for each group
    for group_index, group in enumerate(groups_list):
        group = Group(group, group_index, index_to_channel)
        group_index_to_group[group_index] = group
        for i in range(group.size):
            # Add the group index to the spikes features
            all_spikes_group_indexes[spike_index] = group.index
            spike_index += 1

    # Add the group index to the spikes array
    all_spikes_group_indexes = all_spikes_group_indexes.reshape((-1, 1))
    all_spikes_flat = np.concatenate((all_spikes_flat, all_spikes_group_indexes), axis=1)

    return group_index_to_group, all_spikes_flat
