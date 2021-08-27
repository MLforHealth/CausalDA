import getpass
import os
import torch
from pathlib import Path

def save_checkpoint(model, optimizer, scheduler, sampler_dict, start_step, es, rng):   
    slurm_job_id = os.environ.get('SLURM_JOB_ID')        
    
    if slurm_job_id is not None and Path('/checkpoint/').exists():        
        torch.save({'model_dict': model.state_dict(),
                    'optimizer_dict': optimizer.state_dict(),
                    'scheduler_dict': scheduler.state_dict(),
                    'sampler_dict': sampler_dict,
                    'start_step': start_step,
                    'es': es,
                    'rng': rng
        } 
                   , 
                   Path(f'/checkpoint/{getpass.getuser()}/{slurm_job_id}/chkpt').open('wb')                  
                  )
        
        
def has_checkpoint():
    slurm_job_id = os.environ.get('SLURM_JOB_ID')
    if slurm_job_id is not None and Path(f'/checkpoint/{getpass.getuser()}/{slurm_job_id}/chkpt').exists():
        return True
    return False       


def load_checkpoint():   
    slurm_job_id = os.environ.get('SLURM_JOB_ID')
    if slurm_job_id is not None and Path('/checkpoint/').exists():
        return torch.load(f'/checkpoint/{getpass.getuser()}/{slurm_job_id}/chkpt')
    
def delete_checkpoint():   
    slurm_job_id = os.environ.get('SLURM_JOB_ID')
    chkpt_file = Path(f'/checkpoint/{getpass.getuser()}/{slurm_job_id}/chkpt')
    if slurm_job_id is not None and chkpt_file.exists():
        return chkpt_file.unlink()   
        
    