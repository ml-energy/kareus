from cfuser.scheduler.scheduler import NaiveScheduler, ScalingEfficientScheduler, DecoupledScheduler
from cfuser.scheduler.request import InputConfig
import torch


def test_naive_scheduler():
    print("------------------NaiveScheduler------------------")
    scheduler = NaiveScheduler(ranks=list(range(torch.cuda.device_count())))
    scheduler.add_request(0, InputConfig(batch_size=1, height=1024, width=1024, num_inference_steps=10))
    while scheduler.remaining_requests() > 0:
        scheduled_requests = scheduler.schedule()
        print(f"scheduled requests: {scheduled_requests[0].requests}")


def test_scaling_efficiency_scheduling(scheduler: ScalingEfficientScheduler):
    print("------------------ScalingEfficientScheduler------------------")
    batch_size = 2
    seq_len = 4096
    scheduler.add_request(
        0,
        InputConfig(
            prompt=["a"] * batch_size,
            batch_size=batch_size,
            height=256,
            width=seq_len * 16 * 16 // 256,
            num_inference_steps=10,
            output_type="latent",
        ),
    )
    scheduler.add_request(
        1,
        InputConfig(
            prompt=["a"] * batch_size,
            batch_size=batch_size,
            height=256,
            width=seq_len * 16 * 16 // 256,
            num_inference_steps=10,
            output_type="latent",
        ),
    )
    while scheduler.remaining_requests() > 0:
        scheduled_requests = scheduler.fcfs_scaling_schedule()
        for sch_reqs in scheduled_requests:
            for req in sch_reqs.requests:
                print(
                    f"scheduled requests: {req.req_ids}, attn_ranks: {req.attn_ranks}, non_attn_ranks: {req.non_attn_ranks}"
                )
            print("-------------")
            scheduler.add_ranks(sch_reqs.non_attn_ranks)
        print("--------------------------------")


def test_decoupled_scaling_efficiency_scheduling(scheduler: DecoupledScheduler):
    print("------------------DecoupledScheduler------------------")
    batch_size = 2
    seq_len = 4096
    for i in range(50):
        scheduler.add_request(
            i,
            InputConfig(
                prompt=["a"] * batch_size,
                batch_size=batch_size,
                height=256,
                width=seq_len * 16 * 16 // 256,
                num_inference_steps=10,
                output_type="latent",
            ),
        )
    while scheduler.remaining_requests() > 0:
        scheduled_requests = scheduler.fcfs_schedule()
        if len(scheduled_requests) == 0:
            continue
        for sch_reqs in scheduled_requests:
            for req in sch_reqs.requests:
                print(
                    f"scheduled requests: {req.req_ids}, attn_ranks: {req.attn_ranks}, non_attn_ranks: {req.non_attn_ranks}, estimated_time: {sch_reqs.estimated_time}s"
                )
            print("-------------")
            scheduler.add_ranks(sch_reqs.non_attn_ranks)
        print("--------------------------------")


if __name__ == "__main__":
    LOG_DIR = "log_A100x4_80GB/benchmark/component_scaling_efficiency"
    # LOG_DIR = "/workspaces/CrossServe/log_4xA40_48GB/benchmark/component_scaling_efficiency"
    # LOG_DIR = "log_4xA40_48GB/benchmark/component_scaling_efficiency"
    scheduler = DecoupledScheduler(
        ranks=list(range(torch.cuda.device_count())),
        ring_scaling_efficiency_path=f"{LOG_DIR}/ring_scaling_efficiency/ring_scaling_efficiency.json",
        non_attn_scaling_efficiency_path=f"{LOG_DIR}/non_attn_efficiency/non_attn_efficiency.json",
    )
    # test_naive_scheduler()
    # test_scaling_efficiency_scheduling(scheduler)
    test_decoupled_scaling_efficiency_scheduling(scheduler)
