from bisect import bisect_left, bisect_right


def lower_bound(arr, x):
    """
    the index of the first number not smaller than x
    :param arr: [0, 2, 5, 8, 10]
    :param x: 2
    :return: 1
    """
    # binary_search_for_file_index
    i = bisect_left(arr, x)
    return i


def upper_bound(arr, x):
    """
    the index of the first number larger than x
    :param arr: [0, 2, 5, 8, 10]
    :param x: 2
    :return: 2
    """
    # binary_search_for_file_index
    i = bisect_right(arr, x)
    return i


if __name__ == '__main__':
    a = [0, 2, 5, 8, 10]
    for x in [0, 1, 2, 3]:
        print(lower_bound(a, x))  # 0, 1, 1, 2
        print(upper_bound(a, x))  # 1, 1, 2, 2
