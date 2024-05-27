# credit to chatgpt for unit tests... except the gradient one

import pytest
import jax
import jax.numpy as jnp
import numpy as np
from scipy.linalg import hadamard
import fwht

# Define the shapes, batch sizes, and PRNG keys to test
test_params = [
    (1, 8, 0),
    (2, 8, 1),
    (4, 16, 42),
    (8, 32, 3),
    (16, 64, 4),
    (1024, 2048, 4),
]

@pytest.mark.parametrize("batch_size, size, seed", test_params)
def test_fwht_correctness(batch_size, size, seed):
    key = jax.random.PRNGKey(seed)
    x = jax.random.uniform(key, shape=(batch_size, size))
    out = jax.jit(fwht.fwht)(x)
    
    # Calculate the expected output using scipy's hadamard function
    hadamard_matrix = jnp.array(hadamard(size), dtype=jnp.float32)
    expected = jnp.dot(hadamard_matrix, x.T).T
    
    # Assert that the output is correct
    np.testing.assert_allclose(out, expected, atol=5e-3, err_msg="The output of fwht does not match the expected result.")

@pytest.mark.parametrize("batch_size, size, seed", test_params)
def test_fwht_gradient(batch_size, size, seed):
    key = jax.random.PRNGKey(seed)
    x = jax.random.uniform(key, shape=(batch_size, size))
    
    # Compute the gradient using JAX
    grad_fn = jax.jit(jax.grad(lambda x: fwht.fwht(x).sum()))
    grad = grad_fn(x)

    # hadamard transform its a symmetric linear operator, so its gradient is the same
    # as applying the Hadamard transform again.
    expected_grad = fwht.fwht(jnp.ones_like(x))
    
    # Assert that the gradient is correct
    np.testing.assert_allclose(grad, expected_grad, atol=5e-3, err_msg="The gradient of fwht does not match the expected result.")


@pytest.mark.parametrize("batch_size, size, seed", test_params)
def test_fwht_orthogonality(batch_size, size, seed):
    key = jax.random.PRNGKey(seed)
    x = jax.random.uniform(key, shape=(batch_size, size))
    out = fwht.fwht(x)
    
    # Check orthogonality property of the Hadamard transform
    inv_out = fwht.fwht(out) / size  # Applying FWHT again and normalizing by the dimension
    np.testing.assert_allclose(inv_out, x, atol=5e-3, err_msg="FWHT does not satisfy the orthogonality property.")

@pytest.mark.parametrize("batch_size, size, seed", test_params)
def test_fwht_shape(batch_size, size, seed):
    key = jax.random.PRNGKey(seed)
    x = jax.random.uniform(key, shape=(batch_size, size))
    out = fwht.fwht(x)
    
    # Assert the output shape is the same as the input shape
    assert out.shape == x.shape, f"The output shape {out.shape} does not match the input shape {x.shape}."

if __name__ == "__main__":
    pytest.main()