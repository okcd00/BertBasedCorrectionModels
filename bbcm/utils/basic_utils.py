from bisect import bisect_left, bisect_right


def binary_search_right(arr, x):
    """
    the index of the first number not smaller than x
    :param a: [0, 2, 5, 8, 10]
    :param x: 2
    :return:
    """
    # binary_search_for_file_index
    i = bisect_left(arr, x)
    if i:
        return i
    else:
        return 0


if __name__ == '__main__':
    r1 = binary_search_right([0, 2, 5, 8, 10], 2)  # => 1
    r2 = binary_search_right([0, 2, 5, 8, 10], 3)  # => 2
    print(r1, r2)
