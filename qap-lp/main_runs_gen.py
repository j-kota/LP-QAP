import numpy as np

file_name = "main_runs.in"
file=open(file_name, 'w')

count = 0
for lr in [1e-2,1e-3,1e-4]:
    for quad_reg in [1e-2,1e-3]:
        for num_examples_train in [4000]:
            for clip_grad_norm in [40.0,10000.0]:
                for generative_model in ['ErdosRenyi','Regular']:
                    for noise in [0.03,0.05,0.1,0.25]:
                        for align in [1]:

                            print(  lr,
                                    quad_reg,
                                    num_examples_train,
                                    clip_grad_norm,
                                    generative_model,
                                    noise,
                                    align,
                                    count,
                                    sep=',',
                                    file=file
                             )

                            count = count + 1

file.close()
print("Done")
