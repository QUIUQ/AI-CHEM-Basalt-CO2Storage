from toughreact_fun import *
import os
import time

working_path = os.getcwd()

begin_time = time.time()

for ii in range(104, 221):
    cost_time = gene_routin(ii, working_path,case_seed=55)

end_time = time.time()

print('='*50)
print('='*50)
print(f'total seconds is: {end_time - begin_time:.2f}s')
print(f'total minutes is: {(end_time - begin_time)/60.:.2f}min')
print(f'total hours is: {(end_time - begin_time)/3600.:.2f}h')
print('='*50)
print('='*50)
