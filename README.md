# fwht-jax
Fast Walsh-Hadamard Transform CUDA bindings for JAX

*Credit to Tri Dao for the [CUDA kernel implementation](https://github.com/HazyResearch/structured-nets/blob/master/pytorch/structure/hadamard_cuda/hadamard_cuda_kernel.cu) from [HazyResearch/structured-nets](https://github.com/HazyResearch/structured-nets)*


```bash
pip install .
```

TODO
- make a simple implementation (for non-GPU)
- benchmark fused vs simple implementaiton
- vmap rules
- only supports float32, can se also support bfloat16??
- async stream for better performance?