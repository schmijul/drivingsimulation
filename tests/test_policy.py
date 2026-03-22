import numpy as np

from drivesim.ml.policy import fit_linear_policy


def test_fit_linear_policy_predict_shape() -> None:
    rng = np.random.default_rng(7)
    x = rng.normal(size=(120, 10)).astype(np.float32)
    true_w = rng.normal(size=(10, 2)).astype(np.float32)
    y = x @ true_w

    policy = fit_linear_policy(x, y, l2_reg=1e-4)
    pred = policy.predict(x[:8])

    assert pred.shape == (8, 2)
    assert np.isfinite(pred).all()
