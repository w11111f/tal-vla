class CopyU:
    def __init__(self, src_field, out_field):
        self.src_field = src_field
        self.out_field = out_field


class Mean:
    def __init__(self, msg_field, out_field):
        self.msg_field = msg_field
        self.out_field = out_field


def copy_u(src_field, out_field):
    return CopyU(src_field, out_field)


def mean(msg_field, out_field):
    return Mean(msg_field, out_field)
