import argparse
import os
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from core.utils import load_yaml

def get_run_command(data_types, methods, exp_ids, config_ids, pretrain_fractions, method2run=None, exp2run=None, consistency_exp=False, plot=False):
    suffix = '-ce' if consistency_exp else ''
    if plot:
        suffix += ' --plot'
    commands = []
    for i in range(len(methods)):
        if method2run is not None and methods[i] not in method2run:
            continue
        if exp2run is not None and exp_ids[i] not in exp2run:
            continue 
        data_type = data_types[i] 
        pretrain_fraction = pretrain_fractions[i]
        method = methods[i]
        exp_id = exp_ids[i]
        config_id = config_ids[i]
        if pretrain_fraction > 0:
            cmd_str = f"python main.py -d {data_type} -m {method} -e {exp_id} -c {config_id} -pf {pretrain_fraction} {suffix}"
        else:
            cmd_str = f"python main.py -d {data_type} -m {method} -e {exp_id} -c {config_id} {suffix}"
        commands.append(cmd_str.strip())
    return commands

def run(data_types, methods, exp_ids, config_ids, pretrain_fractions, method2run=None, exp2run=None, consistency_exp=False, max_workers=1):
    suffix = '-ce' if consistency_exp else ''
    commands = []
    for i in range(len(methods)):
        if method2run is not None and methods[i] not in method2run:
            continue
        if exp2run is not None and exp_ids[i] not in exp2run:
            continue 
        data_type = data_types[i] 
        pretrain_fraction = pretrain_fractions[i]
        method = methods[i]
        exp_id = exp_ids[i]
        config_id = config_ids[i]
        if pretrain_fraction > 0:
            cmd_str = f"python main.py -d {data_type} -m {method} -e {exp_id} -c {config_id} -pf {pretrain_fraction} {suffix}"
        else:
            cmd_str = f"python main.py -d {data_type} -m {method} -e {exp_id} -c {config_id} {suffix}"
        commands.append(cmd_str.strip())

    if max_workers <= 1:
        for cmd_str in commands:
            print(f"Running: {cmd_str}")
            os.system(cmd_str)
        return

    print(f"Running {len(commands)} jobs with max_workers={max_workers}")
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_cmd = {
            executor.submit(subprocess.run, cmd_str, shell=True): cmd_str
            for cmd_str in commands
        }
        for future in as_completed(future_to_cmd):
            cmd_str = future_to_cmd[future]
            result = future.result()
            if result.returncode != 0:
                print(f"Command failed ({result.returncode}): {cmd_str}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--config', '-c',
        type=str,
        nargs='+',
        default=None,
        help='one or more config names from configs/eval'
    )
    parser.add_argument(
        '--method2run', '-m',
        type=str,
        nargs='+',
        default=None,
        help='method name to run, if not specified, will run all methods in the config'
    )
    parser.add_argument(
        '--expid2run', '-e',
        type=int,
        nargs='+',
        default=None,
        help='experiment to run, if not specified, will run all experiments'
    )
    parser.add_argument(
        '--skip_first', '-s',
        type=int,
        default=0,
        help='number of initial configs to skip'
    )
    parser.add_argument(
        '--plot', '-p',
        action='store_true',
    )
    parser.add_argument(
        '--max_workers', '-j',
        type=int,
        default=10,
        help='maximum number of concurrent jobs'
    )
    args = parser.parse_args()

    all_run_commands = []
    for cfg in args.config:
        config = load_yaml(f'configs/eval/{cfg}.yaml')
        methods = config['methods']
        exp_ids = config['exp_ids']
        config_ids = config['config_ids']
        if 'data_types' in config:
            data_types = config['data_types']
            pretrain_fractions = config['pretrain_fractions']
        else:
            data_type = config['data_type']
            pretrain_fraction = config.get('pretrain_fraction', 0)
            data_types = [data_type]*len(methods)
            pretrain_fractions = [pretrain_fraction]*len(methods)
        if args.skip_first > 0:
            methods = methods[args.skip_first:]
            exp_ids = exp_ids[args.skip_first:]
            config_ids = config_ids[args.skip_first:]
            data_types = data_types[args.skip_first:]
            pretrain_fractions = pretrain_fractions[args.skip_first:]
        run_commands = get_run_command(
            data_types,
            methods,
            exp_ids,
            config_ids,
            pretrain_fractions,
            args.method2run,
            args.expid2run,
            config.get('consistency_exp', False),
            args.plot,
        )
        all_run_commands.extend(run_commands)
    
    # Execute all commands
    max_workers = args.max_workers
    print(f"Running {len(all_run_commands)} jobs with max_workers={max_workers}")
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_cmd = {
            executor.submit(subprocess.run, cmd_str, shell=True): cmd_str
            for cmd_str in all_run_commands
        }
        for future in as_completed(future_to_cmd):
            cmd_str = future_to_cmd[future]
            result = future.result()
            if result.returncode != 0:
                print(f"Command failed ({result.returncode}): {cmd_str}")