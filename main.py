import os
import argparse
import yaml


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/SMPL.yaml', help='category_name')
    args = parser.parse_args()
    
    with open(args.config, 'r') as yaml_file:
        config = yaml.safe_load(yaml_file)
    print("Experiment: " + args.config)


    cmd = "python optctrlpoints.py"
    cmd += f" --source_data_path={config['source_data_path']}"
    cmd += f" --target_data_path={config['target_data_path']}"
    cmd += f" --log_dir={config['log_dir']}"
    cmd += f" --dump_dir={config['dump_dir']}"
    cmd += f" --num_keypoints={config['num_keypoints']}"
    cmd += f" --calc_w_method={config['calc_w_method']}"
    cmd += f" --eval_metric={config['eval_metric']}"
    cmd += f" --seed={config['seed']}"
    cmd += f" --gpu={config['gpu']}"
    

    print("================")
    print(cmd)
    os.system(cmd)
