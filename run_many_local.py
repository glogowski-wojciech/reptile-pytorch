import os

for _ in range(5):
    os.system('neptune run train_omniglot.py --ex specs/reptile_extensions.py')
