import h5py
import matplotlib.pyplot as plt
import torch

from post555 import *
from Unet_transformer import  *
from dataset import *
from torch.utils.data import DataLoader
import time


"""
Single case output function that allows selection of output image content:
1. Error map
2. Prediction map
3. Real vs Prediction vs Error map
4. Real map

"""
def single_case_output(case_num, plot_class, model, test_dataset, plot_type='real_pre_error',threshold = 1e-7):
    feature, label = test_dataset.get_case(case_num)
    feature = feature.cuda()
    label = label.cuda()
    prediction = torch.zeros([75, 2, 30, 100]).cuda()  # 预测张量也应该在 GPU 上

    #model.eval()


    with torch.no_grad():
        prediction = model(feature)
        prediction = test_dataset.deprocess_label(prediction,test_dataset.labels).cuda()
        prediction[torch.abs(prediction) < threshold] = 0


    if plot_type == 'error':
        plot_class.plot_error(prediction,case_num)
    elif plot_type == 'prediction':
        plot_class.plot_prediction(prediction,case_num)
    elif plot_type == 'real_pre_error':
        plot_class.plot_real_pre_error(prediction,case_num)
    elif plot_type == 'real':
        plot_class.plot_real(case_num)
    else:
        print("Invalid plot type, please choose from 'error', 'prediction', 'real_pre_error', or 'real'.")


#SSIM boxplot with feature variation
def SSIM_boxplot(test_dataset, model, plot_class,threshold = 1e-7):
    """
    Calculate SSIM scores for each case in the test dataset and plot boxplots.
    1. Iterate through each case in the test dataset.
    2. For each case, retrieve the feature and label data.
    3. Use the model to predict the output based on the feature data.
    4. Deprocess the predicted output to match the original label scale.
    5. Calculate the SSIM score between the predicted output and the label.
    6. Store the SSIM scores for all cases.
    7. Plot boxplots of the SSIM scores using the provided plot_class.

    """
    ssim_scores_list = []
    time_total = 0
    print(len(plot_class.case_dir))
    for case_num in tqdm(plot_class.case_dir):
        case_num = int(case_num.split('_')[1])
        feature, label = test_dataset.get_case(case_num)
        feature = feature.cuda()
        label = label.cuda()

        with torch.no_grad():
            time1 = time.time()
            prediction = model(feature)
            time2 = time.time()
            time_total += time2 - time1
            prediction = test_dataset.deprocess_label(prediction,test_dataset.labels)
           #print(prediction.shape)
            #prediction [torch.abs (prediction) < threshold] = 0
            ssim = plot_class.calculate_criteria(prediction, case_num)
            ssim_scores_list.append(ssim)


    ssim_total = torch.stack(ssim_scores_list, dim=1)
    plot_class.plot_SSIM_boxplots(ssim_total)
    #print(ssim_total.mean())
    #print("SSIM boxplots have been plotted.")
    print(time_total )#平均每个case的时间
    return ssim_total.mean()


def R2_time_boxplot(test_dataset, model, plot_class,threshold = 1e-7):
    """
    Size of R2_scores_list: [case_num, time_num]

    """
    R2_scores_list = []
    for case_num in tqdm(plot_class.case_dir):
        case_num = int(case_num.split('_')[1])
        feature, label = test_dataset.get_case(case_num)
        feature = feature.cuda()
        label = label.cuda()

        with torch.no_grad():
            prediction = model(feature)
            prediction = test_dataset.deprocess_label(prediction,test_dataset.labels)
            label = test_dataset.deprocess_label(label,test_dataset.labels)
            prediction [torch.abs (prediction) < threshold] = 0
            r2 = plot_class.calculate_R_squares(label, prediction)
            R2_scores_list.append(r2)

    R2_total = torch.stack(R2_scores_list, dim=1)
    plot_class.plot_R2_boxplots(R2_total)
    print("R2 temporal  boxplots have been plotted.")


def R2_feature_boxplot(test_dataset, model, plot_class,threshold = 1e-7):
    R2_scores_list = []
    for case_num in tqdm(plot_class.case_dir):
        case_num = int(case_num.split('_')[1])
        feature, label = test_dataset.get_case(case_num)
        feature = feature.cuda()
        label = label.cuda()

        with torch.no_grad():
            prediction = model(feature)
            prediction = test_dataset.deprocess_label(prediction,test_dataset.labels).cuda()
            label = test_dataset.deprocess_label(label,test_dataset.labels).cuda()

            prediction [torch.abs (prediction) < threshold] = 0

            r2 = plot_class.calculate_R_squares_feature(label, prediction)

            R2_scores_list.append(r2)


    R2_total = torch.stack(R2_scores_list, dim=0)
    print(R2_total.shape)
    plot_class.plot_R2_boxplots_features(R2_total)
    print("R2 features boxplots have been plotted.")


def R2_time_casedim_boxplot(test_dataset, model, plot_class,threshold = 1e-7):

    case_dim= []

    for case_num in tqdm(plot_class.case_dir):
        case_num = int(case_num.split('_')[1])
        feature, label = test_dataset.get_case(case_num)
        feature = feature.cuda()
        label = label.cuda()

        with torch.no_grad():
            prediction = model(feature)
            prediction = test_dataset.deprocess_label(prediction,test_dataset.labels)
            prediction [torch.abs (prediction) < threshold] = 0
            label [torch.abs (label) < threshold] = 0
            y_real , y_pre = plot_class.casedim_sum(label,prediction)
            case_dim.append(torch.stack([y_real,y_pre],dim=0))
    # size of case_dim: [case_num, 2, time_num, feature_num]
    # size of case_dim[0]: [2, time_num, feature_num]
    print(len(case_dim))
    print(case_dim[0].shape)
    case_dim = torch.stack(case_dim, dim=0)

    print(case_dim.shape)

    case_dim = case_dim.permute(1,2,3,0)
    plot_class.plot_R2_casedim(case_dim)
    print("R2 casedim boxplots have been plotted.")


def pcolormesh_plot(case_num, plot_class, model, test_dataset, plot_type='real_pre_error',threshold = 1e-7):
    feature,label = test_dataset.get_case(case_num)
    feature = feature.cuda()
    label = label.cuda()
    prediction = torch.zeros([75,2,30,100]).cuda()



    with torch.no_grad():
        prediction = model(feature)
        prediction = test_dataset.deprocess_label(prediction,test_dataset.labels).cuda()
        prediction[torch.abs(prediction) < threshold] = 0
        print(prediction)

    if plot_type == 'error':
        plot_class.plot_error(prediction,case_num)
    elif plot_type == 'prediction':
        plot_class.plot_prediction(prediction,case_num)
    elif plot_type == 'real_pre_error':
        plot_class.plot_real_pre_pcolor(prediction,case_num)
    elif plot_type == 'real':
        plot_class.plot_real(case_num)
    elif plot_type == 'plot_real_pre_error':
        plot_class.plot_real_pre_error(prediction,case_num)
    else:
        print("Invalid plot type, please choose from 'error', 'prediction', 'real_pre_error', or 'real'.")


def R2_sum_of_domain(test_dataset, model, plot_class, threshold=1e-5):
    case_dim = []
    volume = plot_class.Volum
    volume = torch.tensor(volume).cuda()

    for case_num in tqdm(plot_class.case_dir):
        case_num = int(case_num.split('_')[1])
        feature, label = test_dataset.get_case(case_num)
        feature = feature.cuda()
        label = label.cuda()

        with torch.no_grad():
            prediction = model(feature)
            prediction = test_dataset.deprocess_label(prediction, test_dataset.labels)

            prediction[torch.abs(prediction) < threshold] = 0

            label = test_dataset.deprocess_label(label, test_dataset.labels)
            label[torch.abs(label) < threshold] = 0

            label = label[:,:,:29,:]
            prediction = prediction[:,:,:29,:]
            volume_reduced = volume[:29,:]


            y_real = torch.sum(label * volume_reduced, dim=(2, 3))
            y_pre = torch.sum(prediction * volume_reduced, dim=(2, 3))

            case_dim.append(torch.stack([y_real, y_pre], dim=0))


    case_dim = torch.stack(case_dim, dim=0)

    case_dim = case_dim.squeeze(-1)

    case_dim = case_dim.permute(1, 2, 0)
    print(case_dim.shape)


    y_real = case_dim[0]
    print(y_real)

    y_pre = case_dim[1]
    print(y_pre)
    r2_values = []

    for t in range(y_real.shape[0]):
        ss_res = torch.sum((y_real[t] - y_pre[t]) ** 2)
        ss_tot = torch.sum((y_real[t] - torch.mean(y_real[t])) ** 2)
        r2 = 1 - ss_res / ss_tot
        r2_values.append(r2.item())


    plt.figure(figsize=(10, 5))
    plt.plot(r2_values[20:], marker='o', linestyle='-', color='b')
    plt.xlabel('Time Step', fontsize=15)
    plt.ylabel('R²', fontsize=15)
    plt.title('R² over Time', fontsize=18)
    plt.grid(True)
    plt.savefig(f"R2_over_time.png", dpi=300, bbox_inches='tight')


    print("R² values have been plotted.")


def R2_overall(test_dataset ,model ,plot_class ,threshold=1e-7):
    y_real_list = []
    y_pre_list = []
    volume = plot_class.Volum
    volume = torch.tensor (volume).cuda ()

    conversion_factors = {
        'calcite': 0.0369 * 0.001 ,  # m^3/mol
        'magnesite': 0.0280 * 0.001 ,
        'siderite': 0.0294 * 0.001
    }

    for case_num in tqdm (plot_class.case_dir):
        case_num = int (case_num.split ('_') [1])
        feature ,label = test_dataset.get_case (case_num)
        feature = feature.cuda ()
        label = label.cuda ()

        with torch.no_grad ():
            prediction = model (feature)
            prediction = test_dataset.deprocess_label (prediction ,test_dataset.labels)
            prediction [torch.abs (prediction) < threshold] = 0
            label = test_dataset.deprocess_label (label ,test_dataset.labels)
            label [torch.abs (label) < threshold] = 0

            label = label [: ,: ,: ,:96]
            prediction = prediction [: ,: ,: ,:96]
            volume_reduced = volume [: ,:96]


            y_real = torch.sum (label * volume_reduced ,dim=(2 ,3))
            y_pre = torch.sum (prediction * volume_reduced ,dim=(2 ,3))

            y_real = y_real.cpu().numpy() / conversion_factors [test_dataset.labels[0]]
            y_pre = y_pre.cpu().numpy() / conversion_factors [test_dataset.labels[0]]

            y_real = torch.tensor (y_real)
            y_pre = torch.tensor (y_pre)

            y_real_list.append (y_real)
            y_pre_list.append (y_pre)



    y_real_all = torch.cat (y_real_list ,dim=0)
    y_pre_all = torch.cat (y_pre_list ,dim=0)


    ss_res = torch.sum ((y_real_all - y_pre_all) ** 2)
    ss_tot = torch.sum ((y_real_all - torch.mean (y_real_all)) ** 2)
    r2_overall = 1 - ss_res / ss_tot
    print (f"Overall R²: {r2_overall.item ()}")


    min_val = min (y_real_all.min () ,y_pre_all.min ())
    max_val = max (y_real_all.max () ,y_pre_all.max ())


    plt.figure (figsize=(12 ,8))

    plt.plot ([min_val ,max_val] ,[min_val ,max_val] ,linestyle='--' ,color='r' ,linewidth=2)


    plt.scatter (y_real_all ,y_pre_all ,alpha=0.5)
    plt.xlabel ('Simulation Values (mol)' ,fontsize=25)
    plt.ylabel ('Predicted Values (mol)' ,fontsize=25)
    # plt.title (f'Scatter Plot of Real vs Predicted Volume of Magnesite\nOverall R²: {r2_overall.item ():.4f}' ,
    #            fontsize=18)
    plt.xticks(fontsize=23)
    plt.yticks(fontsize=23)
    plt.ticklabel_format(style='sci', axis='both', scilimits=(0,0))

    plt.gca ().yaxis.get_offset_text ().set_size (23)
    plt.gca().xaxis.get_offset_text().set_size(23)

    plt.grid (True)
    plt.savefig (f"r2 _scatter_plot.png" ,dpi=300 ,bbox_inches='tight')
    #plt.show()

    print ("Scatter plot has been plotted.")


if __name__ =="__main__":

    case_num = 110
    inchannels = 7
    ouchannels = 1
    features = ['IniPerm.','IniPoro','Volum','Time','GENER','X','Z']
    labels = ['magnesite']
    model_path = r"model215.pth"
    dest_file = r"./output/"
    channel = [16,32,64,128]

    plot_class = Post_Plot(case_num,r"D:\Data_for_CNN1\1e-7\hdf5_files",dest_file,'',
                           '',[],r'D:\Data_for_CNN1\1e-7\hdf5_files\output_absmax_values.json','testing',labels)

    test_dataset = MyDataset(r"D:\Data_for_CNN1\1e-7\hdf5_files",mode='testing',features=features
                             ,labels=labels,transform=None,
                             stats_file=r'D:\Data_for_CNN1\1e-7\hdf5_files\output_absmax_values.json'
                             ,label_stats_file=r'D:\Final\feature_stats.json')



    plot_class.reading_MESH_file ()


    model = UNET(in_channels=inchannels, out_channels= ouchannels)
    model.load_state_dict(torch.load(model_path))
    model = model.cuda()
    model.eval()

    #R2_time_casedim_boxplot(test_dataset, model, plot_class,threshold = 0)
    #single_case_output = single_case_output(550, plot_class, model, test_dataset, plot_type='real_pre_error',threshold = 0)
    ssim =SSIM_boxplot(test_dataset, model, plot_class,threshold = 1e-7)
    #R2_time_boxplot(test_dataset, model, plot_class,threshold = 1e-7)
    #R2_feature_boxplot(test_dataset, model, plot_class,threshold = 1e-7)
    #pcolormesh_plot(550, plot_class, model, test_dataset, plot_type='plot_real_pre_error',threshold = 1e-7)
    #R2_sum_of_domain(test_dataset, model, plot_class,threshold = 1e-7)
    #R2_overall(test_dataset, model, plot_class, threshold=1e-7)
    #print(ssim)