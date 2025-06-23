import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.autograd import Variable
from torch.utils.data import TensorDataset
import torch.nn.functional as F
import torch.nn.utils.prune as prune
import sys
from thop import profile
from torchsummary import summary
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import ReduceLROnPlateau


plt.rcParams['font.sans-serif'] = ['SimHei']  
plt.rcParams['axes.unicode_minus'] = False  

# file path
path_in = r"C:\Users\Mr chen\Desktop\SMTINN\20250410\in.txt"
path_out = r"C:\Users\Mr chen\Desktop\SMTINN\20250410\out.txt"

# load
data_in = load_data(path_in)
data_out = load_data(path_out)

# parameters
N = 20000
epochs = 200
M_max = 20
all_taps = list(range(M_max))

order_list = [4]
hidden_list = [20]
batch_size_list = [128]

best_nmse = float('inf')
best_params = {}
nmse_results = []  

for order in order_list:
    for hidden in hidden_list:
        for batch_size in batch_size_list:
            # Extract I/Q data
            I_in = data_in[:N, 0]
            Q_in = data_in[:N, 1]
            I_out = data_out[:N, 0]
            Q_out = data_out[:N, 1]

            # Default tap 0 is added
            selected_taps = [0]
            best_nmse_current = float('inf')
            iteration = 0

            while len(selected_taps) < 20:  # Modify the loop termination condition and filter the first 4 taps
                iteration += 1
                print(f"\n--------------------------------------------------")
                print(f"iteration {iteration}: selected_taps{selected_taps}")
                available_taps = [t for t in all_taps if t not in selected_taps]
                print(f"available_taps: {available_taps[:19]}")
                print(f"==================================================")

                best_nmse_step = float('inf')
                best_tap = None

                for tap in all_taps:
                    if tap in selected_taps:
                        continue

                    current_taps = selected_taps + [tap]
                    current_taps.sort()
                    print(f"\n▶ Evaluation criteria {tap} add {current_taps}")

                    max_tap = max(current_taps)
                    end_index = N - max_tap

                    x_complex = []
                    for t in current_taps:
                        x_complex.append(I_in[t:end_index + t] + 1j * Q_in[t:end_index + t])
                    x_complex = np.array(x_complex).T

                    y_complex = I_out[max_tap:end_index + max_tap] + 1j * Q_out[max_tap:end_index + max_tap]

                    Y_real = np.real(y_complex)
                    Y_imag = np.imag(y_complex)

                    # Construct the input features X and X_abs
                    X = []
                    for t in current_taps:
                        X.append(I_in[t:end_index + t] + 1j * Q_in[t:end_index + t])
                    X = np.array(X).T

                    X_abs_P = []
                    for j in current_taps:
                        for i in range(1, order):
                            X_abs_P.append(np.abs(I_in[j:end_index + j] + 1j * Q_in[j:end_index + j]) ** i)
                    X_abs_P = np.array(X_abs_P).T

                    # Extract the real and imaginary parts of X
                    X_real = np.real(X)
                    X_imag = np.imag(X)

                    # Merge all input features
                    input_vector = np.hstack([X_real, X_imag])

                    # output
                    out_all = np.c_[Y_real, Y_imag]

                    # network model
                    model = RVTDNN(input_vector.shape[1], hidden, 2)  # a input layer two hidden layer a output layer
                    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)  # optimizer
                    loss_func = torch.nn.MSELoss()  # loss function
                    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=4, min_lr=0.0005)

                    # train
                    epoch = []
                    mse = []
                    train_x, train_y, train_loader = Batch_size(input_vector, out_all, batch_size)  # batch_size

                    model.train()

                    for t in range(epochs):  # epochs
                        train_loss = 0
                        j = 1
                        for i, data in enumerate(train_loader):
                            x_data, y_data = data
                            y_fit = model(x_data)
                            loss = loss_func(y_fit, y_data)
                            optimizer.zero_grad()
                            loss.backward()
                            optimizer.step()
                            train_loss = train_loss + loss
                            j = j + 1
                        train_loss = train_loss / j
                        scheduler.step(train_loss)

                        epoch.append(t)
                        mse.append(train_loss.detach().numpy())

                        if t % 1 == 0:
                            print("epochs:", t, "mse:", train_loss)

                    # predict
                    y_pred = model(train_x)

                    # NMSE
                    data_out_complex = data_out[max_tap:end_index + max_tap, 0] + 1j * data_out[max_tap:end_index + max_tap, 1]
                    y_pred_complex = y_pred[:, 0] + 1j * y_pred[:, 1]
                    nmse = NMSE(data_out_complex, y_pred_complex.detach().numpy())

                    # Merge all input features
                    nmse_results.append({
                        'order': order,
                        'hidden': hidden,
                        'batch_size': batch_size,
                        'iteration': iteration,
                        'current_taps': str(current_taps),
                        'tap_added': tap,
                        'nmse': nmse,
                        'num_taps': len(current_taps)
                    })

                    print(f"  ├─ over,MSE: {train_loss.item():.6f}")
                    print(f"  └─ NMSE: {nmse:.2f} dB")

                    if nmse < best_nmse_step:
                        best_nmse_step = nmse
                        best_tap = tap

                if best_nmse_step < best_nmse_current:
                    best_nmse_current = best_nmse_step
                    selected_taps.append(best_tap)
                    print(f"\n★ The best of this round: New tap {best_tap}")
                    print(f"  new combination: {selected_taps}, NMSE: {best_nmse_step:.2f} dB")
                else:
                    print(f"\n☆ continue {selected_taps}，NMSE: {best_nmse_current:.2f} dB")
