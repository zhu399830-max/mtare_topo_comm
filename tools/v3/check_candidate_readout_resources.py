"""Fixed sealed example forward/backward only; no optimizer or model writes."""
import _bootstrap
import json,resource,time,argparse
from pathlib import Path
import torch
from mtare_topo.data.gse_candidate_training_examples import load_training_examples
from mtare_topo.representation.gse_candidate_readout import SharedCandidateReadout
from mtare_topo.representation.gse_candidate_objective import balanced_candidate_loss


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--set-loss',action='store_true');a=parser.parse_args()
    if not torch.cuda.is_available():raise RuntimeError('CUDA unavailable')
    torch.set_num_threads(1);torch.manual_seed(0)
    torch.use_deterministic_algorithms(True)
    torch.cuda.set_per_process_memory_fraction(28*1024**3/torch.cuda.get_device_properties(0).total_memory)
    examples=load_training_examples(Path(__file__).resolve().parents[2])
    selected=next(e for e in examples if e.observation_id=='four_way__circle__view2')
    rows=[]
    for path in 'ABC':
        torch.manual_seed(0);torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats()
        m=SharedCandidateReadout(path).cuda();args,y,known=selected.on_device('cuda')
        start=time.monotonic();prediction=m(*args)
        if a.set_loss:
            from mtare_topo.representation.gse_candidate_set_objective import candidate_set_loss
            loss,counts=candidate_set_loss(prediction.presence_logits,selected.positions_m.numpy(),selected.expected_positions_m)
        else:loss,counts=balanced_candidate_loss(prediction.presence_logits,y,known)
        loss.backward();torch.cuda.synchronize()
        if any(p.grad is None or not torch.isfinite(p.grad).all() for p in m.parameters()):raise ValueError('gradient failure')
        peak=torch.cuda.max_memory_allocated()
        if peak>28*1024**3 or resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024>32*1024**3:raise MemoryError('resource cap')
        rows.append(dict(path=path,elapsed_s=time.monotonic()-start,peak_allocated_bytes=peak,
            parameters=sum(p.numel() for p in m.parameters()),counts=counts))
        del m,args,y,known,prediction,loss
    print(json.dumps(dict(observations=len(examples),candidates=sum(len(e.positions_m) for e in examples),
        positive_support=sum(int(((e.target>0)&e.target_known).sum()) for e in examples),
        negative_background=sum(int(((e.target==0)&e.target_known).sum()) for e in examples),
        selected_case=selected.observation_id,selected_candidates=len(selected.positions_m),
        maximum_candidates=max(len(e.positions_m) for e in examples),rows=rows,
        peak_host_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
        optimizer_steps=0,encoder_inference=0)),flush=True)


if __name__=='__main__':main()
