import argparse
import os
import pandas as pd
import subprocess
from pathlib import Path

from core.utils import load_yaml


def _get_config_list(config, *keys, default=None):
    for key in keys:
        if key in config and config[key] is not None:
            return config[key]
    return default


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--exp_id', '-e',
        type=str,
        required=True,
        help='experiment config id from configs/opt/<exp_id>.yaml'
    )
    parser.add_argument(
        '--local', '-l',
        action='store_true',
        help='whether to run locally instead of submitting jobs'
    )
    args = parser.parse_args()

    local_run = args.local
    config = load_yaml(f'../configs/opt/{args.exp_id}.yaml')
    skip = config.get('skip', {})

    datas = _get_config_list(config, 'datas', 'data_types', default=[])
    pretrain_fractions = _get_config_list(config, 'pretrain_fractions', 'pretrain_fractions_c', default=[])
    config_ids = _get_config_list(config, 'config_ids', 'config_ids_c', default=[])
    methods2opt = _get_config_list(config, 'methods2opt', 'methods', default=[])

    batch_size = config.get('batch_size', 10)
    iters = config.get('iters', 1000)
    output_id_base = config.get('output_id_base', 1000)
    output_id_step = config.get('output_id_step', 100)

    records = []

    for method in methods2opt:
        for i in range(len(config_ids)):
            current_data = datas[i]
            current_pf = pretrain_fractions[i]
            current_config = config_ids[i]

            if method in skip and current_config in skip[method]:
                print(f"Skipping method {method} with config {current_config}")
                continue

            current_outputid = output_id_base + i + methods2opt.index(method) * output_id_step
            current_expid = current_outputid

            current_record = {
                'method': method,
                'data': current_data,
                'pf': current_pf,
                'config': current_config,
                'output_id': current_outputid,
                'exp_id': current_expid,
                'iters': iters,
                'batch_size': batch_size
            }
            records.append(current_record)

            if local_run:
                cmd = (
                    f'python opt.py -r -c {current_config} -o {current_outputid} '
                    f'-m {method} -d {current_data} -pf {current_pf} -e {current_expid} '
                    f'-i {iters} -b {batch_size}'
                )
                print(f"Running command locally: {cmd}")
                print("parent", Path(__file__).parent.parent)
                subprocess.run(cmd, shell=True, cwd=Path(__file__).parent.parent)
            else:
                os.system(
                    f'sbatch gl_opt.sh {current_config} {current_outputid} {method} '
                    f'{current_data} {current_pf} {current_expid} {iters} {batch_size}'
                )

    print("Submitted all jobs.")
    df = pd.DataFrame(records)
    df.to_csv('opt_job_records.csv', index=False)
