from toughreact_fun import *
import os
import time

working_path = os.getcwd()

with open('errorcase.txt', 'r') as f:
    case_numbers = [int(line.strip()) for line in f]

begin_time = time.time()

case_seed = 555
for ii in case_numbers:
    cost_time = gene_routin(ii, working_path,case_seed=case_seed)

    case_dir = os.path.join(working_path, f'case_{ii}')
    os.chdir(case_dir)

    with open('error_info.txt', 'w') as f:
        f.write(f'Case seed  {case_seed} \n')

    os.chdir(working_path)

end_time = time.time()

print('='*50)
print(f'total seconds is: {end_time - begin_time:.2f}s')
print(f'total minutes is: {(end_time - begin_time)/60.:.2f}min')
print(f'total hours is: {(end_time - begin_time)/3600.:.2f}h')
print('='*50)