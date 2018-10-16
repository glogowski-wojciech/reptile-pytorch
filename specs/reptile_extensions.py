from mrunner.experiment import Experiment
import sys, os
sys.path.append(os.path.join(os.getcwd(), 'specs', 'some_utils'))
from spec_utils import get_git_head_info, get_combinations
# It might be a good practice to not change specification files if run
# successfully, to keep convenient history of experiments. When you want to run
# the same experiment with different hyper-parameters, just copy it.
# Starting name with (approximate) date of run is also helpful.

def create_experiment_for_spec(parameters):
    script = 'train_omniglot.py log/'
    # this will be also displayed in jobs on prometheus
    name = 'wg_reptile'
    project_name = "my-project"
    python_path = '.:specs/some_utils'
    paths_to_dump = ''  # e.g. 'plgrid tensor2tensor', do we need it?
    tags = 'multithreading'.split(' ')
    parameters['git_head'] = get_git_head_info()
    return Experiment(project=project_name, name=name, script=script,
                      parameters=parameters, python_path=python_path,
                      paths_to_dump=paths_to_dump, tags=tags,
                      time='0-1:0'  # days-hours:minutes
                      )

# Set params_configurations, eg. as combinations of grid.
# params are also good place for e.g. output path, or git hash
params_grid = dict(
    root=['/net/archive/groups/plggluna/wglogowski/omniglot/'],
#     delta=[1.0],
#     alpha=['a',],
#     classes=[5],
#     shots=[5],
#     train_shots=[10],
#     meta_iters=[1000], # 100000
#     train_iters=[5],
#     test_iters=[50],
#     meta_batch=[1], # 5
#     meta_lr=[0.2],
#     lr=[1e-3],
#     transductive=[True],
)
params_configurations = get_combinations(params_grid)


def spec():
    experiments = [create_experiment_for_spec(params)
                   for params in params_configurations]
    return experiments
