
# *****************
# Modified by youmebangbang

import argparse
import time
import torch
import sys
import subprocess

if __name__ == '__main__':
    parser = argparse.ArgumentParser() 

    parser.add_argument('-o', '--output_directory', type=str,
                        help='directory to save checkpoints')
    parser.add_argument('-l', '--log_directory', type=str,
                        help='directory to save tensorboard logs')
    parser.add_argument('-c', '--checkpoint_path', type=str, default=None,
                        required=False, help='checkpoint path')
    parser.add_argument('--warm_start', action='store_true',
                        help='load model weights only, ignore specified layers')
    parser.add_argument('--n_gpus', type=int, default=1,
                        required=False, help='number of gpus')
    parser.add_argument('--rank', type=int, default=0,
                        required=False, help='rank of current gpu')
    parser.add_argument('--group_name', type=str, default='group_name',
                        required=False, help='Distributed group name')
    parser.add_argument('--hparams', type=str,
                        required=False, help='comma separated name=value pairs')

    args = parser.parse_args()

    command_line = ['python', '-u', 'train.py']

    num_gpus = torch.cuda.device_count()
    if num_gpus > 1:     
        print("STARTING MULTI-GPU WITH {} GPUS".format(num_gpus))
        args.n_gpus = num_gpus
        workers = []
        job_id = time.strftime("%Y_%m_%d-%H%M%S")
        args.group_name = "group_{}".format(job_id)

        args.rank = num_gpus - 1

        vargs = vars(args)
        vargs_list = vargs.items()
        for v in vargs_list:
            if "warm_start" in v[0]:
                if v[1]:
                    command_line.extend(["--{}".format(v[0])])
            else:
                command_line.extend(["--{}".format(v[0])])
                command_line.extend([str(v[1])])

        print("COMMANDLINE: {}".format(command_line))

        for i in range(num_gpus):
            stdout = None if i == 0 else open("{}/logs/{}_GPU_{}.log".format(args.output_directory, job_id, i),"w")
            p_command_line = command_line
            p_command_line.extend(["--rank", str(i), "--n_gpus", str(num_gpus)])
            p = subprocess.Popen(p_command_line, stdout=stdout)
            workers.append(p)
            
        for p in workers:
            p.wait()