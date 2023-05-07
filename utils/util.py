import copy


def lerp(l, t1, t2):
    t3 = copy.deepcopy(t2)
    for p in t1:
        t3[p] = (1 - l) * t1[p] + l * t2[p]
    return t3
