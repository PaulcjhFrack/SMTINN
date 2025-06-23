def load_data(filename):

    X_I = []
    X_Q = []
    f = open(filename, 'r')
    for line in f.readlines():
        line = line.strip()  
        dataset_temp = line.split('\t')
        dataset = [float(dataset) for dataset in dataset_temp]
        X_I.append(dataset[0])
        X_Q.append(dataset[1])
    X_I = np.array(X_I).reshape((-1, 1))
    X_Q = np.array(X_Q).reshape((-1, 1))
    # Normalization
    max_X = max(np.sqrt(np.square(X_I) + np.square(X_Q)))
    X_I = X_I / (max_X)
    X_Q = X_Q / (max_X)
    X = np.concatenate((X_I, X_Q), axis=1)
    return X
