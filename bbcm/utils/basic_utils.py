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


def colored_text_html(text, color=None, method='background'):
    if color is None:
        return text
    if method == 'font':
        return u"<text style=color:{}>{}</text>".format(color, text)
    elif method == 'background':
        return u"<text style=background-color:{}>{}</text>".format(color, text)
    elif method == 'border':
        return u'<text style="border-style:solid;border-color:{}">{}</text>'.format(color, text)


def highlight_positions(text, positions, color='yellow', p=True, as_html=False):
    """
    将 text 中的 positions处 进行高亮显示

    如果想要把高亮的数据收集起来，进行处理之后再打印，参考 highlight_keyword

    :param text: 可能包含关键字的文本
    :param positions: 要高亮的位置。
      如果是 list 就用 color 染色；
      如果是 dict，{color: positions, ...} 覆盖 color 参数
    :param color:
    :param p: 是否直接打印出来
    :param as_html: True 直接渲染好 html 显示颜色。False 打印出 html 源码
    :type as_html: bool
    :return: html 字符串::

        <text style=color:black>10/0.4kV<text style=color:red>智能箱变</text></text>
    """
    try:
        from IPython.core.display import display, HTML
    except ImportError:
        print('no IPython')

    if not isinstance(positions, dict):
        positions = {color: positions}
    position_to_color = {}
    for c, ps in positions.items():
        for pos in ps:
            position_to_color[pos] = c
    html = u''
    for i, char in enumerate(text):
        if i in position_to_color:
            html += colored_text_html(char, position_to_color[i])
        else:
            html += char
    to_print = colored_text_html(html)
    if p:
        if as_html:
            print(to_print)
        else:
            display(HTML(to_print))
    return to_print


if __name__ == '__main__':
    a = [0, 2, 5, 8, 10]
    for x in [0, 1, 2, 3]:
        print(lower_bound(a, x))  # 0, 1, 1, 2
        print(upper_bound(a, x))  # 1, 1, 2, 2
