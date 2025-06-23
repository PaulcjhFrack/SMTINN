def NMSE(y, y_fit):
    numer_sum = np.sum(np.square(np.abs(y - y_fit)))
    NMSE = numer_sum / np.sum(np.square(np.abs(y)))
    # 转换为 dB 单位
    return 10 * np.log10(NMSE)
