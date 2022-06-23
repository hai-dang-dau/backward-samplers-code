"""
Workflow for numerical experiments.

General scheme:
csv file -(csv_parser)-> dict -(executor)-> intermediate objects -(serializer)-> ouput files.

Wrapping schemes:
DictProcessor <- executor + serializer
start_workflow <- csv file + DictProcessor
"""
# Module fully tested
from abc import ABC, abstractmethod
import typing as tp
import pandas as pd
from pandas.errors import EmptyDataError
from libs_new.utils import random_string, read_json, parallelly_evaluate, memory_protection, iter_data_frame_by_row, \
    zip_with_assert, estimate_size_json
import os
import numpy as np
from itertools import count
from datetime import datetime
import json
from glob import glob
from tqdm import tqdm

class DictProcessor(ABC):
    """
    Process a dictionary that represents one line of the input csv file.
    """
    def __init__(self, d: dict):
        self.failed_to_serialise = None
        random_path_prefix = d['path_random_prefix']
        objs = self.executor(d)
        self.serializer(objs, random_path_prefix, json_maxsize=d['json_maxsize'])

    @abstractmethod
    def executor(self, d: dict):
        """
        Executes the action mandated by the dictionary, then returns intermediate objects
        """
        ...

    def serializer(self, objs, random_path_prefix: str, json_maxsize=np.inf) -> None:
        # tested, even with successive runs (i.e. output_0, output_1, etc).
        """
        Take objects `objs` and serializing them to disk, using path information in the `random_path_prefix` variable.
        """
        esj = estimate_size_json(objs)
        if esj + 0.01 > json_maxsize:
            self.failed_to_serialise = objs
            raise IOError('Json file is too big: {} MB'.format(esj))

        filename = next_full_path(random_path_prefix, 'output', 'json')
        with open(filename['new_name'], 'w') as f:
            json.dump(objs, f)

class MultipleRunCommand(tp.NamedTuple):
    params: dict
    ntimes: int

class MultipleRuns(DictProcessor, ABC):
    # tested
    """
    Class specifically adapted to running the same (random) experiment multiple times to assess the variability of results
    """
    def executor(self, d: dict):
        np.random.seed(d['seed_execution'])
        args_command = [MultipleRunCommand(d, d['nruns_per_core']) for _ in range(d['ncores'])]
        if d['ncores'] > 1:
            res = parallelly_evaluate(self.run_multiple_times, args_command, d['ncores'], 'fork', print_progress=True)
        else:
            res = list(map(self.run_multiple_times, tqdm(args_command)))
        return sum(res, [])

    @abstractmethod
    def one_run(self, d: dict):
        ...

    def run_multiple_times(self, command: MultipleRunCommand):
        res = []
        for _ in range(command.ntimes):
            memory_protection(command.params['maxmem'])
            res.append(self.one_run(command.params))
        return res

def csv_parser(csv_path: str) -> dict:
    # tested
    """
    Reads one line of the file specified by `csv_path`, then convert its into a dictionary. Creates a new file with a random name in the same folder and include the random prefix in the resulting dictionary. The line is then removed from the file.

    If the file is initially empty, a EOFError is raised.
    """
    jobs = pd.read_csv(csv_path)
    d: dict = jobs.iloc[0,:].to_dict()
    for k, v in d.items():
        d[k] = _convert_to_pure_python(_remove_float(_booleanize(v)))

    path_random_prefix = d.get('path_random_prefix')
    if not isinstance(path_random_prefix, str):
        path_random_prefix = random_string()

    real_path = os.path.dirname(os.path.realpath(csv_path))
    path_random_prefix = real_path + '/' + path_random_prefix + '_'
    d['path_random_prefix'] = _convert_to_pure_python(path_random_prefix)

    with open(path_random_prefix + 'input.json', 'w') as f:
        json.dump(d, f)

    jobs = jobs.iloc[1:,:]
    if len(jobs) == 0:
        with open(csv_path, 'w') as _:
            pass
    else:
        jobs.to_csv(csv_path, index=False)

    return d

def _remove_float(v):
    if isinstance(v, float) and (not np.isinf(v)) and (not np.isnan(v)) and (v - int(v) == 0):
        return int(v)
    else:
        return v

def _booleanize(v):
    if v == 'True':
        return True
    elif v == 'False':
        return False
    else:
        return v

def _convert_to_pure_python(v):
    for np_type, py_type in [(np.integer, int), (np.floating, float), (np.str_, str)]:
        if isinstance(v, np_type):
            return py_type(v)
    return v

def start_workflow(csv_path: str, dict_processor: tp.Type[DictProcessor], verbose=True):
    while True:
        try:
            d = csv_parser(csv_path)
        except EmptyDataError:
            break
        else:
            if verbose:
                print('At {}, we process the following:\n{}'.format(datetime.now(), d))
            dict_processor(d)

def next_full_path(prefix: str, suffix: str, extension: str) -> tp.Dict[str, tp.Union[str, int]]:
    # tested
    """
    Determining the next file name in a folder. File names are of the form `prefix_suffix_number.extension` and the function searchs for the next number such that the file name is valid.
    """
    assert prefix.endswith('_')
    assert not suffix.endswith('_')
    assert not extension.startswith('.')
    for i in count():
        intended_name = prefix + suffix + '_' + str(i) + '.' + extension
        if not os.path.exists(intended_name):
            return dict(new_name=intended_name, new_index=i)

class show_first_completion:
    """
    Wrapper around an iterable to print the date and time of the first completion
    :param proba: the probability to print the date and time
    """
    def __init__(self, iterable: tp.Iterable, proba: float):
        self.iterable = iterable
        self.proba = proba

    def __iter__(self):
        self.iterator = iter(self.iterable)
        self.i = 0
        return self

    def __next__(self):
        if self.i == 1 and np.random.rand() < self.proba:
            print('First iteration completed at {}'.format(datetime.now()))
        self.i += 1
        return next(self.iterator)

FilteredResults = tp.NamedTuple('FilteredResults', [('inputs', pd.DataFrame), ('outputs', tp.List)])

def filter_results(path: str, criteria: tp.List[tp.Dict]) -> FilteredResults:
    # tested
    """
    Note: only works with default specifications (i.e., the file couple is *_input.json and *_output_0.json).
    :param path: the folder where results are located
    :param criteria: will return all results whose input dict matches one of the criteria
    """
    assert not path.endswith('/')
    input_jsons = glob(path + '/*input*.json', recursive=False)
    input_dicts = []
    for f in input_jsons:
        d = read_json(f)
        d['_source'] = f
        input_dicts.append(d)

    filtered_dicts = _filter_dicts(criteria, input_dicts)
    output_jsons = [read_json(d['_source'][:-10] + 'output_0.json') for d in filtered_dicts]
    return FilteredResults(inputs=pd.DataFrame(list(filtered_dicts)), outputs=output_jsons)

def _is_subset(a: dict, b:dict):
    return all((a.get(k) == b.get(k)) for k in a.keys())

def _filter_dicts(_criteria: tp.List[dict], dicts: tp.List[dict]):
    # tested
    def _is_ok(_d: dict):
        return any(_is_subset(criterion, _d) for criterion in _criteria)
    return list(filter(_is_ok, dicts))

def iterate_over_filtered_results(filtered_results: FilteredResults):
    input_iterator = iter_data_frame_by_row(filtered_results.inputs)
    return zip_with_assert(input_iterator, filtered_results.outputs)