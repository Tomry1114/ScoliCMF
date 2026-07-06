"""I2SB finetune on paired preop->postop. Reuses train.py's option parser + Runner.
--corrupt mixture => sample_batch takes the 3-tuple path (preop used directly as x1, NO corruption op).
--cond-x1 => condition on preop. Network inits from ADM 256x256_diffusion_uncond.pt."""
import os, sys, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from logger import Logger
from distributed_util import init_processes
from i2sb import Runner, download_ckpt
from train import create_training_options, set_seed
from dataset.scoli_paired import PairedScoli
import colored_traceback.always

def main(opt):
    log = Logger(opt.global_rank, opt.log_dir)
    log.info("========== I2SB paired preop->postop finetune ==========")
    if opt.seed is not None:
        set_seed(opt.seed + opt.global_rank)
    opt.corrupt = "mixture"                       # force 3-tuple path in Runner.sample_batch
    train_dataset = PairedScoli("train", opt.image_size)
    val_dataset   = PairedScoli("val",   opt.image_size)
    corrupt_method = None                         # never called in the mixture branch
    run = Runner(opt, log)
    run.train(opt, train_dataset, val_dataset, corrupt_method)
    log.info("Finish!")

if __name__ == "__main__":
    opt = create_training_options()
    assert opt.corrupt is not None
    download_ckpt("data/")                        # ADM init (already present)
    torch.cuda.set_device(0)
    opt.global_rank = 0; opt.local_rank = 0; opt.global_size = 1
    init_processes(0, opt.n_gpu_per_node, main, opt)
