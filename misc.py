def dot_vectors(v, w):
    if len(v) != len(w):
        raise Exception("WTF!!!")
    return sum(v_i * w_i for v_i, w_i in zip(v, w))


def get_truth_table() -> list:
    # Truth table
    # x1 or (not x2 -> x3)
    # x1 \/ (!x2 -> x3)
    """
    x1	x2	x3	X
    0	0	0	0
    0	0	1	1
    0	1	0	1
    0	1	1	1
    1	0	0	1
    1	0	1	1
    1	1	0	1
    1	1	1	1
    """
    return [
        # (x1, x2, x3, X)
        (0, 0, 0, 0),
        (0, 0, 1, 1),
        (0, 1, 0, 1),
        (0, 1, 1, 1),
        (1, 0, 0, 1),
        (1, 0, 1, 1),
        (1, 1, 0, 1),
        (1, 1, 1, 1),
    ]
