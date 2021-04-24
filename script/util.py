def same_padding_1d(in_length, kernel_size, stride):
    if in_length % stride == 0:
        pad = max(kernel_size - stride, 0)
    else:
        pad = max(kernel_size - (in_length % stride), 0)
    print("same pad total", pad)
    pad_left = pad // 2
    pad_right = pad - pad_left
    return pad_left, pad_right



if __name__ == '__main__':
    in_length = 63
    kernel_size = 4
    stride = 4
    pad_left, pad_right = same_padding_1d(in_length, kernel_size, stride)
    print(pad_left, pad_right)