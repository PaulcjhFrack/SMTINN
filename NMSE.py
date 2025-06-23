def NMSE(y, y_fit):
    numer_sum = np.sum(np.square(np.abs(y - y_fit)))
    NMSE = numer_sum / np.sum(np.square(np.abs(y)))

    return 10 * np.log10(NMSE)
