from torch.utils.data import DataLoader
from  dataset import *
from Unet_transformer import  *
import torch
import torch.nn as nn
import torch.optim as optim
import time
from tqdm import tqdm
USE_FLASH_ATTENTION=1
import os
from loss_function import *
import plot_R2
import post555

## use the min_max normalization

def main():
    if not os.path.exists('loss_logs'):
        os.makedirs('loss_logs')
    loss_log_file = os.path.join('loss_logs','training_loss_MSE_log.txt')

    features = ['IniPerm.','IniPoro','Volum','Time','GENER','X','Z']
    labels = ['magnesite']
    dest_file = r"./output/"




    in_channels = 7
    lr = 5e-4
    step_size = 15
    gamma = 0.75
    weight_decay = 1e-4
    num_epochs = 250

    model = UNET(7,1).to('cuda')
    stating_epoch = 0
    minist_loss = 1
    ssim_best = 0

    training_dataset = MyDataset(r"D:\Data_for_CNN1\1e-7\hdf5_files",mode='training',features=features
                                  ,labels=labels,transform=None,
                                  stats_file=r'D:\Data_for_CNN1\1e-7\hdf5_files\output_absmax_values.json'
                                  ,label_stats_file=r'D:\Final\feature_stats.json')# 您需要定义这个类
    train_loader = DataLoader(training_dataset,batch_size=32,shuffle=True)

    test_dataset = MyDataset(r"D:\Data_for_CNN1\1e-7\hdf5_files",mode='testing',features=features
                             ,labels=labels,transform=None,
                             stats_file=r'D:\Data_for_CNN1\1e-7\hdf5_files\output_absmax_values.json'
                             ,label_stats_file=r'D:\Final\feature_stats.json')  # 需要您定义
    test_loader = DataLoader(test_dataset,batch_size = 32,shuffle=False)




    plot_class = post555.Post_Plot(55,r"D:\Data_for_CNN1\1e-7\hdf5_files",dest_file,'',
                           '',[],r'D:\Data_for_CNN1\1e-7\hdf5_files\output_absmax_values.json','testing',labels)
    plot_class.reading_MESH_file()

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")

    optimizer = optim.Adam(model.parameters(),lr=lr,weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,mode='min',factor=0.8,patience=20,threshold=0.005,
                                                           threshold_mode='rel',cooldown=0,min_lr=0,eps=1e-08)





    model.to('cuda')

    mse_loss_fn = nn.MSELoss()
    total_loss = Total_loss(1,1)



    for epoch in range(stating_epoch,num_epochs):
        start = time.time()

        model.train()
        running_loss = 0.0
        runing_mse = 0.0

        with tqdm(total=len(train_loader)) as pbar:
            for inputs, labels, labels_real in train_loader:
                # print(inputs.shape, labels.shape)
                inputs, labels ,labels_real = inputs.to('cuda'), labels.to('cuda') , labels_real.to('cuda')

                optimizer.zero_grad()

                outputs = model(inputs)

                loss = total_loss(labels_real, labels, outputs)


                loss_mse = mse_loss_fn(outputs, labels)


                loss.backward()

                optimizer.step()

                running_loss += loss.item()
                runing_mse += loss_mse.item()

                pbar.set_description(f"Epoch {epoch + 1}, Loss_total: {loss.item():.8f}, Loss_MSE: {loss_mse.item():.8f}")

                pbar.update(1)

        end = time.time()
        print(f"Epoch {epoch+1}, Loss: {running_loss/len(train_loader)} , mse: {runing_mse/len(train_loader)} , f{(start -end)/60}mins")


        model.eval()


        test_loss = 0.0
        test_mse = 0.0

        with torch.no_grad():
            for inputs, labels ,labels_real in test_loader:
                inputs, labels = inputs.to('cuda'), labels.to('cuda')

                outputs = model(inputs)
                loss = total_loss(labels_real, labels, outputs)
                loss_mse = mse_loss_fn(outputs, labels)
                test_mse += loss_mse.item()
                test_loss += loss.item()

            ssim = plot_R2.SSIM_boxplot(test_dataset,model,plot_class,threshold=1e-7)



        avg_loss = test_loss / len(test_loader)
        avg_mse = test_mse / len(test_loader)



        scheduler.step(avg_mse)
        end = time.time()
        if epoch >= 3 and ssim > ssim_best :
            ssim_best = ssim
            torch.save(model.state_dict(), f"model{epoch+1}.pth")
            print("Saved PyTorch Model State to model.pth")
        print(f"Average  Loss: {avg_loss} , testing mse {avg_mse} ,f{(start -end)/60}mins, ssim = {ssim}")
        current_lr = optimizer.param_groups[0]['lr']
        print("Current learning rate is: {}".format(current_lr))
        with open(loss_log_file, 'a') as log_file:

            log_file.write(f"Epoch {epoch + 1}: Training Loss: {running_loss/len(train_loader):.8f}, Test Loss: {avg_loss:.8f} , Training MSE: {runing_mse/len(train_loader):.8f}, Test MSE: {avg_mse:.8f} ,Current learning rate is: {current_lr}, ssim = {ssim}\n")


if __name__ == '__main__':
    os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
    main()
    #test
    """ model = UNET(7,4).to('cuda')
    real = torch.randn(32,4,30,100).to('cuda')
    fake = torch.randn(32,4,30,100).to('cuda')
    inputs= torch.randn(32,7,30,100).to('cuda')
    pred = model(inputs)
    total_loss = Total_loss(4,4)
    loss = total_loss(real,fake,pred)
    loss.backward()
   """
