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


def group_spikes(channels_spikes: Dict[str, np.ndarray]):
    all_spikes = []
    index_to_channel = {}
    for i, channel_name in enumerate(channels_spikes.keys()):
        channel_spikes = channels_spikes[channel_name]
        all_spikes.append(np.concatenate([channel_spikes, np.full((channel_spikes.shape[0], 1), i)], axis=1))
        index_to_channel[i] = channel_name

    all_spikes_flat = MinHeap.mergeKSortedArrays(all_spikes, len(all_spikes))

    # Group the timestamps based on the window_width
    groups = []
    group = [all_spikes_flat[0]]
    for i in range(1, all_spikes_flat.shape[0]):
        if group[0][0] + SPIKES_GROUPING_WINDOW_SIZE > all_spikes_flat[i][0]:
            group.append(all_spikes_flat[i])
        else:
            groups.append(group)
            group = [all_spikes_flat[i, :]]

    for group in groups:
        print('Group:')
        for record in group:
            print(f'Timestamp: {record[0]} - channel {index_to_channel[record[1]]}')
        print('\n')
