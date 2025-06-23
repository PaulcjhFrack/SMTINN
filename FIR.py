def FIR_filter(X, Y, M):
    X_FIR_filter = X[M:, :]
    input_size = len(X)
    for i in range(1, M):
        X_FIR_filter = np.concatenate([X_FIR_filter, X[M - i:input_size - i, :]], axis=1)
    Y = Y[M:, :]
    return np.array(X_FIR_filter), Y
