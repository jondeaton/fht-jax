# fwht-jax
Fast Walsh-Hadamard Transform CUDA bindings for JAX

*Credit to Tri Dao for the [CUDA kernel implementation](https://github.com/HazyResearch/structured-nets/blob/master/pytorch/structure/hadamard_cuda/hadamard_cuda_kernel.cu) from [HazyResearch/structured-nets](https://github.com/HazyResearch/structured-nets)*


also a demonstration of the simplest possible version of
https://jax.readthedocs.io/en/latest/Custom_Operation_for_GPUs.html#gpu-ops-code-listing


is it faster than a basic JAX implementation? not sure... maybe?


```bash
pip install .
```

TODO
- make a simple implementation (for non-GPU)
- benchmark fused vs simple implementaiton
- vmap rules
- only supports float32, can se also support bfloat16??
- async stream for better performance?