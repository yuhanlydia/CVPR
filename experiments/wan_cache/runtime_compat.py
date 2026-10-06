"""Keep Wan's FP32 islands and qualify memory-efficient attention on Turing."""


def configure_runtime(model, wan_model_module):
    import torch

    # Pinned Wan disables autocast in time_embedding, time_projection and Head.
    # Its affine LayerNorm also consumes FP32 inputs. Retain native FP32 weights.
    model.float()
    capability = torch.cuda.get_device_capability(0)
    backend = "native_flash_attention"
    import wan.modules.attention as attention_module
    if capability[0] < 8 or not (attention_module.FLASH_ATTN_2_AVAILABLE or attention_module.FLASH_ATTN_3_AVAILABLE):
        from torch.nn.attention import SDPBackend, sdpa_kernel
        from torch.nn.functional import scaled_dot_product_attention

        def turing_attention(q, k, v, q_lens=None, k_lens=None,
                             dropout_p=0.0, softmax_scale=None, q_scale=None,
                             causal=False, window_size=(-1, -1),
                             deterministic=False, dtype=None, version=None):
            if window_size != (-1, -1):
                raise ValueError("This adapter requires global attention")
            for lengths, tensor in ((q_lens, q), (k_lens, k)):
                if lengths is not None and not bool((lengths == tensor.size(1)).all()):
                    raise ValueError("Padded sequences require a separately qualified mask")
            if deterministic:
                raise ValueError("Deterministic kernel support is not qualified by this adapter")
            result_dtype = q.dtype
            qh = q.transpose(1, 2).to(torch.float16)
            if q_scale is not None:
                qh = qh * q_scale
            kh = k.transpose(1, 2).to(torch.float16)
            vh = v.transpose(1, 2).to(torch.float16)
            # Fail if this host/wheel cannot run the efficient kernel. A full
            # quadratic math-attention fallback can exceed this card's memory.
            with sdpa_kernel(backends=[SDPBackend.EFFICIENT_ATTENTION]):
                result = scaled_dot_product_attention(
                    qh, kh, vh, dropout_p=dropout_p, is_causal=causal,
                    scale=softmax_scale)
            return result.transpose(1, 2).contiguous().to(result_dtype)

        # WanModel imports this symbol directly. Patching only attention.py
        # would leave the original FA2 call active in its attention layers.
        wan_model_module.flash_attention = turing_attention
        backend = "torch_sdpa_efficient_fp16_global_unpadded"
    return {"autocast_dtype": "float16", "parameter_dtype": "float32",
            "compute_capability": list(capability), "attention_backend": backend,
            "qualification": "requires_real_input_native_forward_parity"}
