"""Full-model Qwen3-4B fine-tuning on re-tokenized local FineWeb shards."""
import os
os.environ['PYTORCH_ALLOC_CONF'] = 'expandable_segments:True'
import argparse
from datetime import datetime
import json
import math
from pathlib import Path
import time

import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from LCT.components.layers import dense_weight
from LCT.components.sparse_utils import SparseAdamW, SparseMuon
from LCT.tensor_buffer import TensorBuffer, _free_regions_snapshot
from qwen.components import Compression, LinearFunction, configure_compression, named_trainable_tensors, dense_state_dict
from qwen.data import TokenStream
from qwen.dist_configs import momentum_dist
from qwen.modeling_qwen3 import Qwen3ForCausalLM
from qwen.optimizer_in_backward import OptimizerInBackward

ROOT = Path(__file__).resolve().parents[1]
MODEL_ID = 'Qwen/Qwen3-4B'
DATA_DIR = ROOT / 'artefacts/qwen/fineweb'
LOG_ROOT = ROOT / 'artefacts/qwen/runs'
COMPRESS_WEIGHTS = True
COMPRESS_ACTIVATIONS = True
COMPRESS_OPTIMISER = True  # Lossless GPU Muon momentum; AdamW moments remain dense on GPU.
BUFFER = False  # Private exact-size overflow allocations avoid an undersized shared arena.
BUFFER_SIZE_MIB = 1024
COMPILE = False
CHECKPOINT_LAYERS = False
CHECKPOINT_HEAD = False
SEQUENCE_LENGTH = 512  # Fine-tuning tokens per sequence; also configurable with --sequence-length.
MICROBATCH_SEQUENCES = 1
OPTIMIZER_IN_BACKWARD = True
TOKENS_PER_STEP = MICROBATCH_SEQUENCES * SEQUENCE_LENGTH  # Hook mode: one microbatch per update.
MAX_GRAD_NORM = None  # Global clipping requires unfused optimization.
TRAIN_STEPS = 1000
WARMUP_STEPS = 50
ADAM_LR = 2e-5
MUON_LR = 1e-3
WEIGHT_DECAY = 0.01
OPTIMIZER = 'muon'  # Muon matrices + AdamW embedding/norms; 'adamw' needs more VRAM.
HEAD_CHUNK_TOKENS = 128
EVAL_EVERY = 100
EVAL_BATCHES = 8
SAVE_EVERY = 0  # Always save the final model; 0 disables intermediate checkpoints.
SEED = 0


@torch.compile(dynamic=False, fullgraph=True)
def cross_entropy(logits, targets):
    return F.cross_entropy(logits.float().reshape(-1, logits.shape[-1]), targets.flatten(), reduction='sum')


def loss_for_batch(model, inputs, targets):
    # Targets from TokenStream are already shifted once.
    hidden = model.model(input_ids=inputs, use_cache=False).last_hidden_state
    weight = model.lm_head.weight
    decoded = dense_weight(weight)
    def project_and_loss(x, y):
        logits = LinearFunction.apply(x, weight, None, None, decoded)
        return cross_entropy(logits, y)
    hidden, targets = hidden.reshape(-1, hidden.shape[-1]), targets.flatten()
    total = hidden.new_zeros((), dtype=torch.float32)
    for start in range(0, len(targets), HEAD_CHUNK_TOKENS):
        x, y = hidden[start:start+HEAD_CHUNK_TOKENS], targets[start:start+HEAD_CHUNK_TOKENS]
        if CHECKPOINT_HEAD and torch.is_grad_enabled():
            total = total + checkpoint(project_and_loss, x, y, use_reentrant=True, preserve_rng_state=False)
        else:
            total = total + project_and_loss(x, y)
    return total


def make_optimizers(model, buffer=None, parameterwise=False):
    named = list(named_trainable_tensors(model))
    def adam(params):
        return SparseAdamW([dict(params=params, lr=ADAM_LR)],betas=(.9,.95),
                           weight_decay=WEIGHT_DECAY,compressed=False)
    def muon(params):
        return SparseMuon(params,lr=MUON_LR,weight_decay=WEIGHT_DECAY,
                          compressed=COMPRESS_OPTIMISER,buffer=buffer,
                          distribution=momentum_dist,match_weight_update=True)
    if OPTIMIZER not in ('adamw', 'muon'):
        raise ValueError("OPTIMIZER must be 'muon' or 'adamw'")
    if parameterwise:
        return [muon([p]) if OPTIMIZER == 'muon' and n.startswith('model.layers.') and p.ndim == 2
                else adam([p]) for n,p in named]
    if OPTIMIZER == 'adamw':
        return [adam([p for _,p in named])]
    muon_params = [p for n,p in named if n.startswith('model.layers.') and p.ndim==2]
    ids = {id(p) for p in muon_params}
    adam_params = [p for _,p in named if id(p) not in ids]
    return [adam(adam_params), muon(muon_params)]


def save(model, step, path):
    torch.save(dict(model=dense_state_dict(model),step=step,model_id=MODEL_ID,
                    config=model.config.to_dict()),path)


def main(steps=TRAIN_STEPS):
    if not torch.cuda.is_available():
        raise RuntimeError('LCT Qwen training requires CUDA')
    if TOKENS_PER_STEP % (MICROBATCH_SEQUENCES*SEQUENCE_LENGTH):
        raise ValueError('TOKENS_PER_STEP must divide evenly into microbatches')
    if OPTIMIZER_IN_BACKWARD and TOKENS_PER_STEP != MICROBATCH_SEQUENCES * SEQUENCE_LENGTH:
        raise ValueError('Optimizer-in-backward requires one microbatch per update; disable it for gradient accumulation')
    if OPTIMIZER_IN_BACKWARD and MAX_GRAD_NORM is not None:
        raise ValueError('Global gradient clipping is incompatible with optimizer-in-backward')
    # Validate data before downloading/loading the large model.
    stream = TokenStream(DATA_DIR,'train',MODEL_ID)
    torch.manual_seed(SEED)
    model = Qwen3ForCausalLM.from_pretrained(MODEL_ID,dtype=torch.bfloat16,
                                           attn_implementation='sdpa').cuda().train()
    buffer = TensorBuffer(BUFFER_SIZE_MIB*2**20,device='cuda') if BUFFER and (COMPRESS_WEIGHTS or COMPRESS_ACTIVATIONS or COMPRESS_OPTIMISER) else None
    configure_compression(model,Compression(COMPRESS_WEIGHTS,COMPRESS_ACTIVATIONS,buffer=buffer))
    if CHECKPOINT_LAYERS:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': True})
    else:
        model.gradient_checkpointing_disable()
    params = [p for _,p in named_trainable_tensors(model)]
    optimizers = make_optimizers(model, buffer, parameterwise=OPTIMIZER_IN_BACKWARD)
    fused = OptimizerInBackward(model, optimizers) if OPTIMIZER_IN_BACKWARD else None
    if COMPILE:model.model.compile()
    run_dir = LOG_ROOT/datetime.now().strftime('%Y%m%d_%H%M%S_%f');run_dir.mkdir(parents=True)
    (run_dir/'settings.json').write_text(json.dumps(dict(model=MODEL_ID,steps=steps,optimizer=OPTIMIZER,
        weights=COMPRESS_WEIGHTS,activations=COMPRESS_ACTIVATIONS,momentum=COMPRESS_OPTIMISER,
        checkpoint_layers=CHECKPOINT_LAYERS,
        checkpoint_head=CHECKPOINT_HEAD,
        optimizer_in_backward=OPTIMIZER_IN_BACKWARD,max_grad_norm=MAX_GRAD_NORM,
        sequence_length=SEQUENCE_LENGTH,
        tokens_per_step=TOKENS_PER_STEP,microbatch=MICROBATCH_SEQUENCES,seed=SEED),indent=2))
    def log(message):
        print(message,flush=True)
        with (run_dir/'train.log').open('a') as f:f.write(message+'\n')
    log(f'{sum(p.numel() for p in params):,} trainable parameters; {run_dir}')
    accumulation=TOKENS_PER_STEP//(MICROBATCH_SEQUENCES*SEQUENCE_LENGTH)
    for step in range(1,steps+1):
        torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();start=time.perf_counter()
        warmup=min(WARMUP_STEPS,max(1,steps//10))
        ratio=min(1.,step/warmup)*(.1+.9*.5*(1+math.cos(math.pi*max(0,step-warmup)/max(1,steps-warmup))))
        for opt in optimizers:
            if isinstance(opt,SparseMuon):
                opt.lr=MUON_LR*ratio;opt.neg_lr.fill_(-opt.lr);opt.decay.fill_(1-opt.lr*opt.weight_decay)
            else:
                for group in opt.param_groups:group['lr']=ADAM_LR*ratio
        if fused is not None:
            fused.begin(math.ceil(TOKENS_PER_STEP / HEAD_CHUNK_TOKENS), checkpoint_head=CHECKPOINT_HEAD)
        total=torch.zeros((),device='cuda')
        for _ in range(accumulation):
            x,y=stream.batch(MICROBATCH_SEQUENCES,SEQUENCE_LENGTH)
            loss=loss_for_batch(model,x,y)
            total+=loss.detach()
            (loss/TOKENS_PER_STEP).backward()
            del loss
        if fused is not None:
            fused.finish()
        else:
            if MAX_GRAD_NORM is not None:
                torch.nn.utils.clip_grad_norm_(params,MAX_GRAD_NORM,error_if_nonfinite=True)
            for opt in optimizers:opt.step()
            for p in params:p.grad=None
        torch.cuda.synchronize()
        buffer_text=''
        if buffer is not None:
            used=buffer.capacity_bytes-sum(size for _,size in _free_regions_snapshot(buffer))
            buffer_text=f' buffer={used/2**20:.1f}/{buffer.capacity_bytes/2**20:.1f}MiB'
        log(f'step={step} loss={total.item()/TOKENS_PER_STEP:.5f} time={time.perf_counter()-start:.2f}s '
            f'peak={torch.cuda.max_memory_allocated()/2**20:.1f}MiB{buffer_text}')
        if step%EVAL_EVERY==0 or step==steps:
            validation=TokenStream(DATA_DIR,'val',MODEL_ID);model.eval()
            with torch.no_grad():
                val=0.
                for _ in range(EVAL_BATCHES):
                    x,y=validation.batch(MICROBATCH_SEQUENCES,SEQUENCE_LENGTH)
                    val+=loss_for_batch(model,x,y).item()
            log(f'step={step} validation_loss={val/(EVAL_BATCHES*MICROBATCH_SEQUENCES*SEQUENCE_LENGTH):.5f}')
            model.train()
        if step==steps or (SAVE_EVERY and step%SAVE_EVERY==0):
            save(model,step,run_dir/f'{step}.pt')
    if fused is not None:
        fused.remove()


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--steps',type=int,default=TRAIN_STEPS)
    parser.add_argument('--sequence-length',type=int,default=SEQUENCE_LENGTH)
    args = parser.parse_args()
    if args.sequence_length < 1:
        parser.error('--sequence-length must be positive')
    SEQUENCE_LENGTH = args.sequence_length
    if OPTIMIZER_IN_BACKWARD:
        TOKENS_PER_STEP = MICROBATCH_SEQUENCES * SEQUENCE_LENGTH
    main(args.steps)
