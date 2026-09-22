"""The public surface: what it exports, and what one forward pass guarantees."""
import pytest
import torch

import qfield
from qfield import QuadratureField, gaussian_target, mmd_sq, solve_weights

SIGMA = 0.5


def _reference(d=2, m=64, seed=20260612):
    rho = gaussian_target(SIGMA, d=d)
    z0 = rho.sampler(m, torch.Generator().manual_seed(seed))
    return rho, z0


def test_the_documented_names_are_exported():
    for name in ("QuadratureField", "ConditionalQuadratureField", "solve_weights",
                 "mmd_sq", "gram", "select_emission", "gaussian_target", "emit"):
        assert hasattr(qfield, name), name


def test_weights_sum_to_one_and_are_signed():
    rho, z0 = _reference()
    w = solve_weights(z0, rho.mu_fn(z0), SIGMA)
    assert w.shape == (z0.shape[0],)
    assert abs(float(w.sum()) - 1.0) < 1e-9
    assert float(w.min()) < 0.0, "the solve is unconstrained in sign"


def test_reweighting_the_samples_beats_equal_weights():
    rho, z0 = _reference()
    m = z0.shape[0]
    mu = rho.mu_fn(z0)
    w_eq = torch.full((m,), 1.0 / m, dtype=z0.dtype)
    w = solve_weights(z0, mu, SIGMA)
    assert float(mmd_sq(z0, w, mu, rho.c_rho, SIGMA)) <= float(
        mmd_sq(z0, w_eq, mu, rho.c_rho, SIGMA)
    )


def test_one_forward_pass_is_never_worse_than_the_samples():
    """The returned quadrature beats the samples it was handed, at every seed."""
    rho = gaussian_target(SIGMA, d=2)
    torch.manual_seed(0)
    field = QuadratureField(d=2).to(torch.float64)
    for seed in range(5):
        z0 = rho.sampler(64, torch.Generator().manual_seed(seed))
        out = field.emit(z0, sigma=SIGMA, mu_fn=rho.mu_fn)
        w_eq = torch.full((64,), 1.0 / 64, dtype=z0.dtype)
        floor = float(mmd_sq(z0, w_eq, rho.mu_fn(z0), rho.c_rho, SIGMA))
        ours = float(mmd_sq(out["z"], out["w"], rho.mu_fn(out["z"]), rho.c_rho, SIGMA))
        assert ours <= floor
        assert abs(float(out["w"].sum()) - 1.0) < 1e-9
        assert out["name"] in ("samples", "reweighted", "moved")


def test_an_untrained_field_leaves_the_samples_where_they_are():
    """The head is zero-initialized, so the displacement starts at zero."""
    torch.manual_seed(0)
    field = QuadratureField(d=2).to(torch.float64)
    _, z0 = _reference()
    assert torch.allclose(field(z0.unsqueeze(0)).squeeze(0), z0)


def test_the_field_is_permutation_equivariant():
    torch.manual_seed(0)
    field = QuadratureField(d=2).to(torch.float64)
    _, z0 = _reference()
    perm = torch.randperm(z0.shape[0])
    a = field(z0.unsqueeze(0)).squeeze(0)[perm]
    b = field(z0[perm].unsqueeze(0)).squeeze(0)
    assert torch.allclose(a, b, atol=1e-10)


def test_the_bandwidth_is_optional_and_taken_from_the_samples():
    """Without a bandwidth the field uses the median heuristic on the input."""
    from qfield import gaussian_reference, median_bandwidth

    rho, z0 = gaussian_reference(d=2, m=64, seed=20260612)
    torch.manual_seed(0)
    field = QuadratureField(d=2).to(z0.dtype)

    out = field.emit(z0, mu_fn=rho.mu_fn)
    assert out["sigma"] == median_bandwidth(z0)

    fixed = field.emit(z0, mu_fn=rho.mu_fn, sigma=out["sigma"])
    assert torch.allclose(out["w"], fixed["w"])
    assert out["name"] == fixed["name"]


def test_the_reference_helper_takes_its_bandwidth_from_its_samples():
    from qfield import gaussian_reference, median_bandwidth

    rho, z0 = gaussian_reference(d=2, m=32, seed=1)
    assert z0.shape == (32, 2)
    mu = rho.mu_fn(z0)
    assert mu.shape == (32,) and float(mu.min()) > 0.0
    assert median_bandwidth(z0) > 0.0


def _conditional_field(dy=3, dtype=torch.float64):
    from qfield import ConditionalQuadratureField

    torch.manual_seed(0)
    return ConditionalQuadratureField(
        d=2, cond_kind="vector", cond_vec_dim=dy).to(dtype)


def test_the_conditional_toy_runs_one_pass():
    from qfield import conditional_gaussian_reference

    rho, z0, y = conditional_gaussian_reference(d=2, dy=3, m=64, seed=0)
    assert z0.shape == (64, 2) and y.shape == (3,)
    out = _conditional_field().emit(z0, y, mu_fn=rho.mu_fn)
    assert out["z"].shape == z0.shape
    assert abs(float(out["w"].sum()) - 1.0) < 1e-9
    assert out["name"] in ("samples", "reweighted", "moved")


def test_an_untrained_conditional_field_emits_no_displacement():
    """The head is zero-initialized, so the observation cannot move anything yet."""
    from qfield import conditional_gaussian_reference

    rho, z0, y = conditional_gaussian_reference(d=2, dy=3, m=64, seed=0)
    field = _conditional_field()
    assert torch.allclose(field.emit(z0, y, mu_fn=rho.mu_fn)["z"], z0)


def test_the_conditional_field_reads_the_observation():
    """With a live head, two observations of the family give different nodes."""
    from qfield import conditional_gaussian_reference

    rho, z0, y = conditional_gaussian_reference(d=2, dy=3, m=64, seed=0)
    field = _conditional_field()
    with torch.no_grad():
        # the head and every cross-attention output start at zero, which is
        # what makes an untrained field return the samples unchanged
        for mod in field.modules():
            if isinstance(mod, torch.nn.Linear) and float(mod.weight.abs().sum()) == 0.0:
                mod.weight.normal_(0.0, 0.1)
    # the displaced nodes, not the selection: with a random head the moved
    # candidate loses the comparison, which is the guarantee doing its job
    a = field(z0.unsqueeze(0), y.unsqueeze(0))
    b = field(z0.unsqueeze(0), (y + 1.0).unsqueeze(0))
    assert not torch.allclose(a, b)


def test_a_conditional_field_will_not_run_unconditioned():
    """Forgetting the observation must fail loudly, not silently drop it."""
    from qfield import emit

    rho, z0 = _reference()
    with pytest.raises(ValueError):
        emit(_conditional_field(), z0, mu_fn=rho.mu_fn)


def test_an_unconditional_field_refuses_an_observation():
    from qfield import emit

    rho, z0 = _reference()
    torch.manual_seed(0)
    field = QuadratureField(d=2).to(z0.dtype)
    with pytest.raises(ValueError):
        emit(field, z0, y=torch.zeros(3, dtype=z0.dtype))


def test_the_kernel_mean_is_estimated_from_the_samples():
    """No reference object: extra samples beyond the nodes supply the mean."""
    from qfield import gaussian_reference, kernel_mean

    from qfield import gaussian_target

    _, pool = gaussian_reference(d=2, m=2064, seed=3)
    torch.manual_seed(0)
    field = QuadratureField(d=2).to(pool.dtype)

    out = field.emit(pool, m=64)
    assert out["z"].shape == (64, 2)
    assert abs(float(out["w"].sum()) - 1.0) < 1e-9

    # the estimate agrees with the exact mean, read at the same bandwidth
    z, sigma = pool[:64], out["sigma"]
    est = kernel_mean(z, pool[64:], sigma)
    exact = gaussian_target(sigma, d=2).mu_fn(z)
    assert float((est - exact).abs().max()) < 2e-2       # sampling noise
    assert abs(float((est - exact).mean())) < 5e-3       # and no bias


def test_asking_for_more_nodes_than_samples_fails():
    from qfield import gaussian_reference

    _, pool = gaussian_reference(d=2, m=32, seed=0)
    torch.manual_seed(0)
    field = QuadratureField(d=2).to(pool.dtype)
    with pytest.raises(ValueError):
        field.emit(pool, m=64)
