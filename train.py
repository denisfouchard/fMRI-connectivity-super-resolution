# import torch
# import torch.nn as nn
# import numpy as np
# import matplotlib.pyplot as plt
# from preprocessing import *
# from model import *

# criterion = nn.MSELoss()

# def train(model, optimizer, subjects_adj, subjects_labels, args):
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     model.to(device)
  
#     i = 0
#     all_epochs_loss = []
#     no_epochs = args.epochs

#     for epoch in range(no_epochs):
#         epoch_loss = []
#         epoch_error = []

#         for lr, hr in zip(subjects_adj, subjects_labels):
#             model.train()
#             optimizer.zero_grad()
          
#             lr = torch.from_numpy(lr).type(torch.FloatTensor).to(device)
#             hr = torch.from_numpy(hr).type(torch.FloatTensor).to(device)
          
#             model_outputs, net_outs, start_gcn_outs, layer_outs = model(lr)
#             model_outputs = unpad(model_outputs, args.padding)

#             padded_hr = pad_HR_adj(hr, args.padding)
#             # eig_val_hr, U_hr = torch.symeig(padded_hr, eigenvectors=True, upper=True)
#             eig_val_hr, U_hr = torch.linalg.eigh(padded_hr, UPLO='U')
          
#             U_hr_reduced = U_hr[:, :model.layer.lr_dim]
#             loss = args.lmbda * criterion(net_outs, start_gcn_outs) + \
#                   criterion(model.layer.weights, U_hr_reduced) + \
#                   criterion(model_outputs, hr)
          
#             # loss = args.lmbda * criterion(net_outs, start_gcn_outs) + \
#             #        criterion(model.layer.weights, U_hr) + \
#             #        criterion(model_outputs, hr) 
          
#             error = criterion(model_outputs, hr)
          
#             loss.backward()
#             optimizer.step()

#             epoch_loss.append(loss.item())
#             epoch_error.append(error.item())
      
#         i += 1
#         print(f"Epoch: {i}, Loss: {np.mean(epoch_loss)}, Error: {np.mean(epoch_error)*100}%")
#         all_epochs_loss.append(np.mean(epoch_loss))

# def test(model, test_adj, test_labels, args):
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     model.to(device)

#     test_error = []
#     preds_list = []
#     g_t = []
  
#     i = 0
#     for lr, hr in zip(test_adj, test_labels):
#         all_zeros_lr = not np.any(lr)
#         all_zeros_hr = not np.any(hr)

#         if not all_zeros_lr and not all_zeros_hr:  
#             lr = torch.from_numpy(lr).type(torch.FloatTensor).to(device)
#             np.fill_diagonal(hr, 1)
#             hr = torch.from_numpy(hr).type(torch.FloatTensor).to(device)

#             preds, _, _, _ = model(lr)
#             preds = unpad(preds, args.padding)
            
#             preds_list.append(preds.flatten().detach().cpu().numpy())

#             error = criterion(preds, hr)
#             g_t.append(hr.flatten().cpu().numpy())
#             print(error.item())
#             test_error.append(error.item())
     
#             i += 1

#     print(f"Test error MSE: {np.mean(test_error)}")

# # Load dataset
# subjects_adj, subjects_labels, test_adj = data()




import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from preprocessing import *
from model import *

# Change loss function to Mean Absolute Error (MAE) for Kaggle
criterion = nn.L1Loss()  # MAE instead of MSE

def train(model, optimizer, subjects_adj, subjects_labels, args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
  
    i = 0
    all_epochs_loss = []
    no_epochs = args.epochs

    for epoch in range(no_epochs):
        epoch_loss = []
        epoch_error = []

        for lr, hr in zip(subjects_adj, subjects_labels):
            model.train()
            optimizer.zero_grad()
          
            lr = torch.from_numpy(lr).type(torch.FloatTensor).to(device)
            hr = torch.from_numpy(hr).type(torch.FloatTensor).to(device)
          
            model_outputs, net_outs, start_gcn_outs, layer_outs = model(lr)
            model_outputs = unpad(model_outputs, args.padding)

            padded_hr = pad_HR_adj(hr, args.padding)
            eig_val_hr, U_hr = torch.linalg.eigh(padded_hr, UPLO='U')
          
            U_hr_reduced = U_hr[:, :model.layer.lr_dim]

            # Use MAE (L1 Loss) instead of MSE
            loss = args.lmbda * criterion(net_outs, start_gcn_outs) + \
                  criterion(model.layer.weights, U_hr_reduced) + \
                  criterion(model_outputs, hr)
          
            error = criterion(model_outputs, hr)  # MAE error calculation
          
            loss.backward()
            optimizer.step()

            epoch_loss.append(loss.item())
            epoch_error.append(error.item())
      
        i += 1
        print(f"Epoch: {i}, Loss: {np.mean(epoch_loss)}, Error (MAE): {np.mean(epoch_error)}")
        all_epochs_loss.append(np.mean(epoch_loss))

def test(model, test_adj, test_labels, args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    test_error = []
    preds_list = []
    g_t = []
  
    i = 0
    for lr, hr in zip(test_adj, test_labels):
        all_zeros_lr = not np.any(lr)
        all_zeros_hr = not np.any(hr)

        if not all_zeros_lr and not all_zeros_hr:  
            lr = torch.from_numpy(lr).type(torch.FloatTensor).to(device)
            np.fill_diagonal(hr, 1)
            hr = torch.from_numpy(hr).type(torch.FloatTensor).to(device)

            preds, _, _, _ = model(lr)
            preds = unpad(preds, args.padding)
            
            preds_list.append(preds.flatten().detach().cpu().numpy())

            error = criterion(preds, hr)  # Compute MAE instead of MSE
            g_t.append(hr.flatten().cpu().numpy())
            print(f"MAE: {error.item()}")  # Print MAE error
            test_error.append(error.item())
     
            i += 1

    print(f"Test error MAE: {np.mean(test_error)}")  # Report MAE instead of MSE

# Load dataset
subjects_adj, subjects_labels, test_adj = data()
