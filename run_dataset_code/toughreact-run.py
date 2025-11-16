import subprocess
import toughio
import gstools as gs
import numpy as np
from scipy.stats import qmc
import matplotlib.pyplot as plt
import os
import shutil
import time
import numpy as np
import scipy.interpolate
import matplotlib.colors as colors

project_path = os.getcwd()


def anisotropy_angle():
    np.random.seed(55)
    angles = np.pi*np.random.uniform(low=0, high=1, size=(5000, 1))
    np.savetxt('angles.txt',     angles)


def new_folders(case_num,):
    folder_name = 'case_' + str(case_num)
    if not os.path.exists(folder_name):
        os.mkdir(folder_name)
    shutil.copy('treact_eco2n.exe', folder_name)
    shutil.copy('flow.inp', folder_name)
    shutil.copy('MESH', folder_name)
    shutil.copy('CO2TAB', folder_name)
    shutil.copy('concrt140.dll', folder_name)
    shutil.copy('chemical.inp', folder_name)
    shutil.copy('solute.inp', folder_name)
    shutil.copy('license.lic', folder_name)
    shutil.copy('thermodb.txt', folder_name)
    shutil.copy('rlm922.dll', folder_name)
    shutil.copy('SAVE', folder_name)
    shutil.copy('GENER', folder_name)


def change_dir(case_num):
    folder_name = 'case_' + str(case_num)
    os.chdir(project_path + '\\' + folder_name)


def generate_random_fields(case_num):
    # generate random permeability fields
    color_style = 'jet'
    x = y = range(64)
    cond_pos = [[32, 32]]
    cond_val = [np.log(100.0)]
    # angle_file = np.loadtxt('angles.txt')
    model = gs.Exponential(dim=2, var=0.5, len_scale=10)
    krige = gs.Krige(model, cond_pos=cond_pos, cond_val=cond_val, mean=np.log(100.))
    cond_srf = gs.CondSRF(krige, seed=case_num*55)
    # field = cond_srf.set_pos([x, y], "structured")
    #
    # srf = gs.SRF(model, mean=np.log(80.0), seed=case_num*77)
    field = cond_srf.structured([x, y])
    data_array = cond_srf.post_field(field=field, process=False)
    data_processed_1 = np.exp(data_array)
    data_processed_2 = data_processed_1 * 1.0e-15
    data_flipped = np.flip(data_processed_2, axis=0)    # flip in order to match TOUGH MESH

    plt.imshow(data_processed_2, cmap=color_style)    # not right, should have show non-flipped perm image !
    plt.colorbar()
    plt.xticks([])
    plt.yticks([])
    plt.title(f'mean: {np.mean(data_flipped)}\nmax: {np.max(data_flipped)}\nmin: {np.min(data_flipped)}')
    plt.savefig('hetero_perm_field.png', dpi=300)
    plt.close()
    plt.imshow(data_array, cmap=color_style)
    plt.colorbar()
    plt.xticks([])
    plt.yticks([])
    plt.title(f'mean: {np.mean(data_processed_1)}\nmax: {np.max(data_processed_1)}\nmin: {np.min(data_processed_1)}')
    plt.savefig('hetero_perm_field (log).tiff', dpi=1000, bbox_inches='tight', pad_inches=0.1)
    plt.close()

    # save permeability
    np.savetxt('perm_field.txt', data_flipped.flatten())

    # save porosity according to permeability
    porosity = np.log10(data_processed_1) * 0.05 + 0.15
    porosity_flipped = np.flip(porosity, axis=0)
    np.savetxt('poro_field.txt', porosity_flipped.flatten())

    plt.imshow(porosity, cmap=color_style)
    plt.colorbar()
    plt.xticks([])
    plt.yticks([])
    plt.title(f'mean: {np.mean(porosity_flipped)}\nmax: {np.max(porosity_flipped)}\nmin: {np.min(porosity_flipped)}')
    plt.savefig('hetero_poro_field.png', dpi=300)
    plt.close()

def generate_random_fields555(case_num):

    mesh = toughio.read_mesh("MESH")

    ##first generate the ramdon field, then assign to each element
    model = gs.Gaussian(dim=2, var=0.2, len_scale=10)
    srf = gs.SRF(model, seed=case_num * 55)

    xz_coords = np.array([elem['center'][::2] for elem in mesh['elements'].values()])

    random_field_values = srf(xz_coords.T)

    for value, element in zip(random_field_values, mesh['elements'].values()):
        element['random_field_value'] = value
        element['random_permeability_field'] = 8.75 * 10 ** (-14 + value)
        element['random_porosity_field'] = (np.exp(value)) ** (1 / 3) * 0.1


    ##plot the kriging interpolation result
    xz_coords = np.array([elem['center'][::2] for elem in mesh['elements'].values()])
    interp_vals = np.array([elem['random_field_value'] for elem in mesh['elements'].values()])

    grid_x, grid_y = np.meshgrid(np.linspace(xz_coords[:, 0].min(), xz_coords[:, 0].max(), 100),
                                 np.linspace(xz_coords[:, 1].min(), xz_coords[:, 1].max(), 100))

    grid_z = scipy.interpolate.griddata(xz_coords, interp_vals, (grid_x, grid_y), method='linear')

    plt.figure(figsize=(40, 10))
    contour = plt.contourf(grid_x, grid_y, grid_z, cmap='viridis', levels=20)
    plt.colorbar(contour, label='Interpolated Random Field Value')
    plt.xlabel('X')
    plt.ylabel('Z')
    plt.title('Kriging Interpolation on Regular Grid')
    plt.savefig(f"{case_num}random_field.png")
    plt.close()

    ## generate permeability field and plot and txt
    interp_vals = np.array([elem['random_permeability_field'] for elem in mesh['elements'].values()])

    grid_x, grid_y = np.meshgrid(np.linspace(xz_coords[:, 0].min(), xz_coords[:, 0].max(), 100),
                                 np.linspace(xz_coords[:, 1].min(), xz_coords[:, 1].max(), 100))

    grid_z = scipy.interpolate.griddata(xz_coords, interp_vals, (grid_x, grid_y), method='linear')

    levels = np.logspace(np.log10(interp_vals.min()), np.log10(interp_vals.max()), 20)

    plt.figure(figsize=(40, 10))
    contour = plt.contourf(grid_x, grid_y, grid_z, cmap='viridis', levels=levels,
                           norm=colors.SymLogNorm(linthresh=interp_vals.min(), linscale=0.1,
                                                  vmin=interp_vals.min(), vmax=interp_vals.max(), base=10))

    cbar = plt.colorbar(contour)
    cbar.set_label('Random Permeability Field Value')

    min_exp = np.log10(interp_vals.min())
    max_exp = np.log10(interp_vals.max())

    num_ticks = 5
    exp_range = np.linspace(min_exp, max_exp, num_ticks)
    ticks = np.power(10, exp_range)

    cbar.set_ticks(ticks)
    cbar.set_ticklabels([f'$10^{{ {int(e)} }}$' for e in exp_range])

    plt.xlabel('X')
    plt.ylabel('Z')
    plt.title(f'Random Permeability Field,mean:{interp_vals.mean()},max:{interp_vals.max()},min:{interp_vals.min()}')
    plt.savefig(f"case:{case_num}Random Permeability Field.png")

    plt.close()
    np.savetxt('perm_field.txt', interp_vals)

    ## generate porosity field and plot and txt
    interp_vals = np.array([elem['random_porosity_field'] for elem in mesh['elements'].values()])
    # 定义规则网格
    grid_x, grid_y = np.meshgrid(np.linspace(xz_coords[:, 0].min(), xz_coords[:, 0].max(), 100),
                                 np.linspace(xz_coords[:, 1].min(), xz_coords[:, 1].max(), 100))


    grid_z = scipy.interpolate.griddata(xz_coords, interp_vals, (grid_x, grid_y), method='linear')

    plt.figure(figsize=(40, 10))
    contour = plt.contourf(grid_x, grid_y, grid_z, cmap='viridis', levels=20)
    plt.colorbar(contour, label='Interpolated Random Porosity Value')
    plt.xlabel('X')
    plt.ylabel('Z')
    plt.title(f'Random Porosity Field,mean:{interp_vals.mean()},max:{interp_vals.max()},min:{interp_vals.min()}')
    plt.savefig(f"case:{case_num}Random Porosity Field.png")
    np.savetxt('poro_field.txt', interp_vals)
    plt.close()





def ini_incon(seed_num):
    # modify SAVE file poro & perm from the 2 txt files
    parameters = toughio.read_input('SAVE')
    element_dict = parameters.values()
    element_dict = list(element_dict)[0]

    f1 = open('poro_field.txt', 'r')
    f2 = open('perm_field.txt', 'r')

    porosity_lines = f1.readlines()
    permeability_lines = f2.readlines()
    # np.random.seed(seed_num * 77)
    # ini_pressure = np.random.uniform(low=20.1, high=28.1, size=1) * 1.e6
    # ini_pressure_array = np.full((64*64, 1), ini_pressure, dtype=np.float64)
    # np.savetxt('ini_pressure_[20,28].txt', ini_pressure_array)

    line_i = 0
    for values in element_dict.values():
        values['porosity'] = float(porosity_lines[line_i])
        values['userx'][0] = float(permeability_lines[line_i])
        values['userx'][1] = float(permeability_lines[line_i])
        values['userx'][2] = float(permeability_lines[line_i])
        # values['values'][0] = ini_pressure
        line_i += 1
    f1.close()
    f2.close()

    toughio.write_input('INCON', parameters)

    with open('INCON', 'r+') as f:
        lines = f.readlines()
        f.seek(0)
        f.writelines(lines[1:-2])
        f.truncate()
        f.write('\n')


def ini_incon555(seed_num):
    parameters = toughio.read_input("SAVE", file_format="toughreact-flow")
    element_dict = parameters.values()
    element_dict = list(element_dict)[0]

    f1 = open('poro_field.txt', 'r')
    f2 = open('perm_field.txt', 'r')

    porosity_lines = f1.readlines()
    permeability_lines = f2.readlines()

    line_i = 0
    for values in element_dict.values():
        values['porosity'] = float(porosity_lines[line_i])
        values['permeability'] = float(permeability_lines[line_i])
        # values['values'][0] = ini_pressure
        line_i += 1
    f1.close()
    f2.close()

    toughio.write_input('INCON', parameters, file_format="toughreact-flow")

    with open("INCON", "r") as file:
        lines = file.readlines()


    with open("INCON", "w") as file:
        file.writelines(lines[1:])








def run_tough(case_num,):
    folder_name = 'case_' + str(case_num)
    command = f'wsl bash -c "/mnt/d/original_tough/{folder_name}/tough3-eco2n input.txt output.txt"'
    p = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    while p.poll() is None:
        with open('cmd_out.txt', 'a') as f:
            line = p.stdout.readline().decode("utf8")
            f.write(line)

def run_toughreact555(case_num):

    command = r"C:\Users\49110\Desktop\toughreact-run\treact_eco2n "


    process = subprocess.Popen(command, shell=True)


    process.wait()

def latin_sampling(l_bounds, u_bounds, num_samples, seed_num):
    sampler = qmc.LatinHypercube(d=1, seed=seed_num)
    sample = sampler.random(n=num_samples)
    sample_scaled = qmc.scale(sample, l_bounds, u_bounds)
    sample_scaled = np.round(sample_scaled, 2)
    return sample_scaled



def make_dynamic_injection(case_num):
    f = toughio.read_input('input0.txt')
    time_seq = list(np.arange(0., 155520000. + 2592000., 2592000.))
    tmp = latin_sampling(2.1, 15.1, len(time_seq), case_num * 55)
    rates_seq = []
    for rate in tmp:
        rates_seq.append(rate.item())
    f['generators'][0]['times'] = time_seq
    f['generators'][0]['rates'] = rates_seq
    f['times'] = list(np.arange(2592000., 155520000. + 2592000., 2592000.))
    np.savetxt('injection_rates_dynamic_[1,15].txt', rates_seq, fmt='%.02f')
    toughio.write_input('input.txt', f)


def make_dynamic_injection555(case_num, num_wells, group_size, num_cases_per_group):
    f = toughio.read_input('GENER', file_format="toughreact-flow")

    # possible dynamic injection time points for further research
    # change the injection rate every 5 days, for 1 year + 5 days
    time_seq = list(np.arange(0., 86400 * 365 + 86400 * 5, 86400 * 5))
    # the injection rate is between 3.0 and 7.0 kg/s
    total_rate = latin_sampling(3., 7., 1, case_num * 55)
    # the injection wells are divided into several groups, each group has group_size wells
    # each case only activates one group of wells
    group_index = (case_num - 1) // num_cases_per_group
    start_well = group_index
    end_well = start_well + group_size

    well_rates_seq = np.zeros((num_wells, len(time_seq)))
    for i in range(len(time_seq)):
        group_rate = total_rate / group_size
        well_rates_seq[start_well:end_well, i] = group_rate

    for i in range(num_wells):
        f['generators'][i]['times'] = time_seq
        f['generators'][i]['rates'] = well_rates_seq[i].tolist()
        f['generators'][i]['specific_enthalpy'] = list(np.zeros(len(time_seq)))
    toughio.write_input("GENER", f)

    with open("GENER", "r") as file:
        lines = file.readlines()


    with open("GENER", "w") as file:
        file.writelines(lines[1:])





def gene_routin(case_num, pro_path):
    print(f'---------- case {case_num} begins ----------')
    case_begin_time = time.time()
    new_folders(case_num)
    change_dir(case_num)
    # anisotropy_angle()
    generate_random_fields555(case_num)
    ini_incon555(case_num)
    make_dynamic_injection555(case_num , 30 , 10 , 55)
    run_toughreact555(case_num)
    os.chdir(pro_path)
    case_end_time = time.time()
    print(f'case {case_num} finished. {(case_end_time - case_begin_time) / 60.:.2f} mins used.' + '\n')
    return case_end_time - case_begin_time


if __name__ == '__main__':
    enerate_random_fields555(1)




