import numpy as np

from drivesim.ml.policy import features_from_observation, fit_linear_policy


def test_fit_linear_policy_predict_shape() -> None:
    rng = np.random.default_rng(7)
    x = rng.normal(size=(120, 10)).astype(np.float32)
    true_w = rng.normal(size=(10, 2)).astype(np.float32)
    y = x @ true_w

    policy = fit_linear_policy(x, y, l2_reg=1e-4)
    pred = policy.predict(x[:8])

    assert pred.shape == (8, 2)
    assert np.isfinite(pred).all()


def test_fit_linear_policy_learns_bias_term() -> None:
    rng = np.random.default_rng(11)
    x = rng.normal(size=(240, 6)).astype(np.float32)
    true_w = rng.normal(size=(6, 2)).astype(np.float32)
    true_b = np.array([0.75, -0.35], dtype=np.float32)
    y = x @ true_w + true_b

    policy = fit_linear_policy(x, y, l2_reg=1e-6)
    pred = policy.predict(x)

    assert np.allclose(pred, y, atol=1e-3)


def test_features_from_observation_returns_fixed_size_vector() -> None:
    obs = {
        "pose": np.array([10.0, 5.0, 0.2, 3.0], dtype=np.float32),
        "goal": np.array([40.0, 25.0], dtype=np.float32),
        "lidar": np.linspace(2.0, 80.0, num=41, dtype=np.float32),
    }
    feats = features_from_observation(obs)
    assert feats.shape == (14,)
    assert np.isfinite(feats).all()
