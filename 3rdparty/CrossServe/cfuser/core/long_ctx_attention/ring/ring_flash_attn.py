import torch
from flash_attn.flash_attn_interface import _flash_attn_forward
from cfuser.core.long_ctx_attention.comm import RingComm
from cfuser.core.long_ctx_attention.utils import update_out_and_lse


def ring_flash_attn_forward(
    process_group,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    softmax_scale,
    dropout_p=0,
    causal=True,
    window_size=(-1, -1),
    alibi_slopes=None,
    deterministic=False,
    attn_layer=None,
    joint_tensor_key=None,
    joint_tensor_value=None,
    joint_tensor_query=None,
    joint_strategy="none",
):
    supported_joint_strategy = ["none", "front", "rear"]
    if joint_strategy not in supported_joint_strategy:
        raise ValueError(
            f"joint_strategy: {joint_strategy} not supprted. supported joint strategy: {supported_joint_strategy}"
        )
    elif joint_strategy != "none" and (joint_tensor_key is None or joint_tensor_value is None):
        raise ValueError(f"joint_tensor_key & joint_tensor_value must not be None when joint_strategy is not None")

    comm = RingComm(process_group)

    out = None
    lse = None

    next_k, next_v = None, None

    for step in range(comm.world_size):
        if step + 1 != comm.world_size:
            next_k: torch.Tensor = comm.send_recv(k)
            next_v: torch.Tensor = comm.send_recv(v)
            comm.commit()

        # if joint_strategy == "rear":
        #     if step + 1 == comm.world_size:
        #         key = torch.cat([k, joint_tensor_key], dim=1)
        #         value = torch.cat([v, joint_tensor_value], dim=1)
        #         if joint_tensor_query is not None:
        #             q = torch.cat([q, joint_tensor_query], dim=1)
        #     else:
        #         key, value = k, v
        if joint_strategy == "front":
            if step == 0:
                key = torch.cat([joint_tensor_key, k], dim=1)
                value = torch.cat([joint_tensor_value, v], dim=1)
                if joint_tensor_query is not None:
                    q = torch.cat([joint_tensor_query, q], dim=1)
            else:
                key, value = k, v
        else:
            raise NotImplementedError("Flux only supports front cond joint for now")

        # elif joint_strategy == "none":
        #     key, value = k, v

        if not causal or step <= comm.rank:
            block_out, _, _, _, _, block_lse, _, _ = _flash_attn_forward(
                q,
                key,
                value,
                dropout_p,
                softmax_scale,
                causal=causal and step == 0,
                window_size=window_size,
                softcap=0.0,
                alibi_slopes=alibi_slopes,
                return_softmax=True and dropout_p > 0,
            )
            out, lse = update_out_and_lse(out, lse, block_out, block_lse)
        if step + 1 != comm.world_size:
            comm.wait()
            k = next_k
            v = next_v

    out = out.to(q.dtype)
    lse = lse.squeeze(dim=-1).transpose(1, 2)
    return out, lse


def ring_flash_attn_func(
    q,
    k,
    v,
    dropout_p=0.0,
    softmax_scale=None,
    causal=False,
    window_size=(-1, -1),
    alibi_slopes=None,
    deterministic=False,
    return_attn_probs=False,
    group=None,
    attn_layer=None,
    joint_tensor_key=None,
    joint_tensor_value=None,
    joint_tensor_query=None,
    joint_strategy="none",
):
    if softmax_scale is None:
        softmax_scale = q.shape[-1] ** (-0.5)

    assert alibi_slopes is None
    if attn_layer is None:
        k = k.contiguous()
        v = v.contiguous()
    out, softmax_lse = ring_flash_attn_forward(
        group,
        q,
        k,
        v,
        softmax_scale=softmax_scale,
        dropout_p=dropout_p,
        causal=causal,
        window_size=window_size,
        alibi_slopes=alibi_slopes,
        deterministic=False,
        attn_layer=attn_layer,
        joint_tensor_key=joint_tensor_key,
        joint_tensor_value=joint_tensor_value,
        joint_tensor_query=joint_tensor_query,
        joint_strategy=joint_strategy,
    )

    return out if not return_attn_probs else (out, softmax_lse, None)
