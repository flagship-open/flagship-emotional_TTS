import time
import torch
import sys
import subprocess
import argparse
import os

if __name__ == '__main__':
    argslist = list(sys.argv)[1:]
    
    if "--enable_gpus" not in argslist:
        enable_gpus = '0,1,2,3,4,5,6,7'
    else:
        enable_gpus = argslist[argslist.index("--enable_gpus")+1]
    
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   
    os.environ["CUDA_VISIBLE_DEVICES"]=enable_gpus
    
    gpus_list = enable_gpus.split(',')
    
    if "--group_id" not in argslist:
        group_id = time.strftime("%Y_%m_%d-%H%M%S")
    else:
        group_id = argslist[argslist.index("--group_id")+1]
        
    workers = []
    argslist.append('--n_gpus={}'.format(len(gpus_list)))
    argslist.append("--enable_gpus={}".format(enable_gpus))
    argslist.append("--group_name=group_{}".format(group_id))
    
    hparams = [s for s in argslist if "hparams" in s]
    if not hparams:
        argslist.append("--hparams=distributed_run=True")
    elif 'distributed_run=True' not in hparams[0]:
        argslist[argslist.index(hparams[0])] = hparams[0] + ',distributed_run=True'        
    
    for i, _ in enumerate(gpus_list):
        argslist.append('--rank={}'.format(i))
        stdout = None if i == 0 else open("/data3/sejikpark/.jupyter/workspace/desktop/tts/mellotron/GPU.log".format(group_id, i),"w")
        print(argslist)
        p = subprocess.Popen([str(sys.executable)]+argslist, stdout=stdout)
        workers.append(p)
        argslist = argslist[:-1]
    
    for p in workers:
        p.wait()