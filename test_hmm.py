"""
Tests for hmm.py — Hidden Markov Model library.
Autor: Edson Ricardo Pedroza Olivera.

Run with pytest:   python -m pytest test_hmm.py -v
Run without pytest: python test_hmm.py
"""

import numpy as np
from hmm import HMM

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------
A_vit  = np.array([[0.7, 0.3], [0.4, 0.6]])
B_vit  = np.array([[0.8, 0.2], [0.4, 0.6]])
pi_vit = np.array([0.8, 0.2])

A_fwd  = np.array([[0.7, 0.3], [0.4, 0.6]])
B_fwd  = np.array([[0.9, 0.1], [0.2, 0.8]])
pi_fwd = np.array([0.6, 0.4])

OBS_SHORT = [0, 0, 1, 0]
OBS_TRAIN = [0, 0, 1, 0, 0, 1, 0, 0]
OBS_MULTI = np.array([0, 0, 1, 0, 0, 1, 0, 1, 1, 0])
LENGTHS   = [4, 3, 3]

# ---------------------------------------------------------------------------
# Minimal assertion helpers (no pytest required)
# ---------------------------------------------------------------------------
def assert_true(cond, msg=""):
    if not cond:
        raise AssertionError(msg)

def assert_raises(exc_type, fn, *args, **kwargs):
    try:
        fn(*args, **kwargs)
        raise AssertionError(f"Expected {exc_type.__name__} but no exception raised.")
    except exc_type:
        pass

# ===========================================================================
# 1. Constructor
# ===========================================================================
class TestConstructor:
    def setup(self):
        pass

    def test_explicit_parameters(self):
        hmm = HMM(2, 2, A=A_fwd, B=B_fwd, pi=pi_fwd)
        assert_true(np.allclose(hmm.A, A_fwd) and np.allclose(hmm.B, B_fwd) and np.allclose(hmm.pi, pi_fwd))

    def test_random_init_is_stochastic(self):
        hmm = HMM(3, 4)
        assert_true(np.allclose(hmm.A.sum(axis=1), 1.0))
        assert_true(np.allclose(hmm.B.sum(axis=1), 1.0))
        assert_true(np.allclose(hmm.pi.sum(), 1.0))

    def test_log_matrices_precomputed(self):
        hmm = HMM(2, 2, A=A_fwd, B=B_fwd, pi=pi_fwd)
        assert_true(np.allclose(hmm._log_A, np.log(A_fwd + hmm._EPS)))

    def test_invalid_n_estados(self):
        assert_raises(ValueError, HMM, 0, 2)

    def test_invalid_n_obs(self):
        assert_raises(ValueError, HMM, 2, -1)

    def test_wrong_A_shape(self):
        assert_raises(ValueError, HMM, 2, 2, np.array([[0.7, 0.3, 0.0], [0.4, 0.6, 0.0]]))

    def test_non_stochastic_A(self):
        bad_A = np.array([[0.7, 0.5], [0.4, 0.6]])
        assert_raises(ValueError, HMM, 2, 2, bad_A, B_fwd, pi_fwd)

    def test_non_stochastic_pi(self):
        bad_pi = np.array([0.3, 0.3])
        assert_raises(ValueError, HMM, 2, 2, A_fwd, B_fwd, bad_pi)

# ===========================================================================
# 2. viterbi
# ===========================================================================
class TestViterbi:
    def setup(self):
        self.hmm = HMM(2, 2, A=A_vit, B=B_vit, pi=pi_vit)

    def test_known_result(self):
        path, _ = self.hmm.viterbi(OBS_SHORT)
        assert_true(path == [0, 0, 0, 0], f"got {path}")

    def test_path_length_and_range(self):
        path, _ = self.hmm.viterbi(OBS_SHORT)
        assert_true(len(path) == len(OBS_SHORT))
        assert_true(all(0 <= s < 2 for s in path))

    def test_return_log_false_gives_probability(self):
        _, prob = self.hmm.viterbi(OBS_SHORT, return_log=False)
        assert_true(0.0 < prob <= 1.0)

    def test_return_log_true_gives_log_probability(self):
        _, lp = self.hmm.viterbi(OBS_SHORT, return_log=True)
        assert_true(lp <= 0.0)

    def test_return_log_consistency(self):
        _, prob = self.hmm.viterbi(OBS_SHORT, return_log=False)
        _, lp   = self.hmm.viterbi(OBS_SHORT, return_log=True)
        assert_true(np.isclose(np.exp(lp), prob))

    def test_empty_sequence(self):
        path, prob = self.hmm.viterbi([])
        assert_true(path == [] and prob == 0.0)

    def test_type_error(self):
        assert_raises(TypeError, self.hmm.viterbi, [[0, 1], [1, 0]])

    def test_value_error(self):
        assert_raises(ValueError, self.hmm.viterbi, [0, 5, 1])

# ===========================================================================
# 3. forward
# ===========================================================================
class TestForward:
    def setup(self):
        self.hmm = HMM(2, 2, A=A_fwd, B=B_fwd, pi=pi_fwd)

    def test_log_alpha_shape(self):
        log_alpha, _ = self.hmm.forward(OBS_SHORT, return_log=True)
        assert_true(log_alpha.shape == (len(OBS_SHORT), 2))

    def test_known_log_likelihood(self):
        _, log_lik = self.hmm.forward(OBS_SHORT, return_log=True)
        assert_true(round(log_lik, 4) == -2.6427, f"got {round(log_lik,4)}")

    def test_linear_likelihood_in_unit_interval(self):
        _, lik = self.hmm.forward(OBS_SHORT, return_log=False)
        assert_true(0.0 < lik <= 1.0)

    def test_return_log_consistency(self):
        _, lik = self.hmm.forward(OBS_SHORT, return_log=False)
        _, ll  = self.hmm.forward(OBS_SHORT, return_log=True)
        assert_true(np.isclose(np.exp(ll), lik))

    def test_empty_sequence(self):
        alpha, _ = self.hmm.forward([], return_log=False)
        assert_true(alpha.shape == (0, 2))

    def test_type_error(self):
        assert_raises(TypeError, self.hmm.forward, [[0, 1]])

    def test_value_error(self):
        assert_raises(ValueError, self.hmm.forward, [0, 99])

# ===========================================================================
# 4. backward
# ===========================================================================
class TestBackward:
    def setup(self):
        self.hmm = HMM(2, 2, A=A_fwd, B=B_fwd, pi=pi_fwd)

    def test_shape(self):
        lb = self.hmm.backward(OBS_SHORT, return_log=True)
        assert_true(lb.shape == (len(OBS_SHORT), 2))

    def test_last_row_log(self):
        lb = self.hmm.backward(OBS_SHORT, return_log=True)
        assert_true(np.allclose(lb[-1], 0.0))

    def test_last_row_linear(self):
        b = self.hmm.backward(OBS_SHORT, return_log=False)
        assert_true(np.allclose(b[-1], 1.0))

    def test_return_log_consistency(self):
        lb = self.hmm.backward(OBS_SHORT, return_log=True)
        b  = self.hmm.backward(OBS_SHORT, return_log=False)
        assert_true(np.allclose(np.exp(lb), b))

    def test_likelihood_matches_forward(self):
        obs = np.asarray(OBS_SHORT, dtype=int)
        _, ll_fwd = self.hmm.forward(obs, return_log=True)
        lb = self.hmm.backward(obs, return_log=True)
        log_emit0 = self.hmm._log_B[:, obs[0]]
        ll_bwd = float(self.hmm._log_sum_exp(self.hmm._log_pi + log_emit0 + lb[0], axis=0))
        assert_true(np.isclose(ll_fwd, ll_bwd, atol=1e-6))

    def test_empty_sequence(self):
        b = self.hmm.backward([], return_log=False)
        assert_true(b.shape == (0, 2))

    def test_type_error(self):
        assert_raises(TypeError, self.hmm.backward, [[0, 1]])

    def test_value_error(self):
        assert_raises(ValueError, self.hmm.backward, [0, 99])

# ===========================================================================
# 5. forward_backward
# ===========================================================================
class TestForwardBackward:
    def setup(self):
        self.hmm = HMM(2, 2, A=A_fwd, B=B_fwd, pi=pi_fwd)

    def test_shape(self):
        gamma = self.hmm.forward_backward(OBS_SHORT)
        assert_true(gamma.shape == (len(OBS_SHORT), 2))

    def test_rows_sum_to_one(self):
        gamma = self.hmm.forward_backward(OBS_SHORT)
        assert_true(np.allclose(gamma.sum(axis=1), 1.0))

    def test_values_in_unit_interval(self):
        gamma = self.hmm.forward_backward(OBS_SHORT)
        assert_true((gamma >= 0).all() and (gamma <= 1).all())

    def test_return_log_consistency(self):
        g  = self.hmm.forward_backward(OBS_SHORT, return_log=False)
        lg = self.hmm.forward_backward(OBS_SHORT, return_log=True)
        assert_true(np.allclose(np.exp(lg), g))

    def test_type_error(self):
        assert_raises(TypeError, self.hmm.forward_backward, [[0, 1]])

    def test_value_error(self):
        assert_raises(ValueError, self.hmm.forward_backward, [0, 99])

# ===========================================================================
# 6. baum_welch
# ===========================================================================
class TestBaumWelch:
    def setup(self):
        self.hmm = HMM(2, 2, A=A_fwd, B=B_fwd, pi=pi_fwd)

    def test_log_likelihood_improves(self):
        _, logL0 = self.hmm.forward(OBS_TRAIN, return_log=True)
        logLf = self.hmm.baum_welch(OBS_TRAIN, max_iter=50)
        assert_true(logLf > logL0, f"logLf={logLf:.4f} not > logL0={logL0:.4f}")

    def test_returns_float(self):
        assert_true(isinstance(self.hmm.baum_welch(OBS_TRAIN, max_iter=5), float))

    def test_return_log_history(self):
        logL, hist = self.hmm.baum_welch(OBS_TRAIN, max_iter=10, return_log_history=True)
        assert_true(isinstance(hist, list) and len(hist) >= 1)
        assert_true(logL == hist[-1])

    def test_history_monotone(self):
        _, hist = self.hmm.baum_welch(OBS_TRAIN, max_iter=50, return_log_history=True)
        assert_true((np.diff(hist) >= -1e-9).all())

    def test_parameters_stochastic_after_training(self):
        self.hmm.baum_welch(OBS_TRAIN, max_iter=20)
        assert_true(np.allclose(self.hmm.A.sum(axis=1), 1.0, atol=1e-9))
        assert_true(np.allclose(self.hmm.B.sum(axis=1), 1.0, atol=1e-9))
        assert_true(np.allclose(self.hmm.pi.sum(),       1.0, atol=1e-9))

    def test_empty_raises(self):
        assert_raises(ValueError, self.hmm.baum_welch, [])

    def test_type_error(self):
        assert_raises(TypeError, self.hmm.baum_welch, [[0, 1]])

    def test_value_error(self):
        assert_raises(ValueError, self.hmm.baum_welch, [0, 99])

# ===========================================================================
# 7. baum_welch_mult
# ===========================================================================
class TestBaumWelchMult:
    def setup(self):
        self.hmm = HMM(2, 2, A=A_fwd, B=B_fwd, pi=pi_fwd)

    def test_log_likelihood_improves(self):
        starts = np.cumsum([0] + LENGTHS[:-1])
        logL0 = sum(self.hmm.forward(OBS_MULTI[s:s+l], return_log=True)[1]
                    for s, l in zip(starts, LENGTHS))
        logLf = self.hmm.baum_welch_mult(OBS_MULTI, LENGTHS, max_iter=50)
        assert_true(logLf > logL0, f"logLf={logLf:.4f} not > logL0={logL0:.4f}")

    def test_returns_float(self):
        assert_true(isinstance(self.hmm.baum_welch_mult(OBS_MULTI, LENGTHS, max_iter=5), float))

    def test_return_log_history(self):
        logL, hist = self.hmm.baum_welch_mult(OBS_MULTI, LENGTHS, max_iter=10, return_log_history=True)
        assert_true(isinstance(hist, list) and len(hist) >= 1)
        assert_true(logL == hist[-1])

    def test_history_monotone(self):
        _, hist = self.hmm.baum_welch_mult(OBS_MULTI, LENGTHS, max_iter=50, return_log_history=True)
        assert_true((np.diff(hist) >= -1e-9).all())

    def test_parameters_stochastic_after_training(self):
        self.hmm.baum_welch_mult(OBS_MULTI, LENGTHS, max_iter=20)
        assert_true(np.allclose(self.hmm.A.sum(axis=1), 1.0, atol=1e-9))
        assert_true(np.allclose(self.hmm.B.sum(axis=1), 1.0, atol=1e-9))
        assert_true(np.allclose(self.hmm.pi.sum(),       1.0, atol=1e-9))

    def test_empty_raises(self):
        assert_raises(ValueError, self.hmm.baum_welch_mult, [], [])

    def test_lengths_mismatch_raises(self):
        assert_raises(ValueError, self.hmm.baum_welch_mult, OBS_MULTI, [4, 4])

    def test_negative_length_raises(self):
        assert_raises(ValueError, self.hmm.baum_welch_mult, OBS_MULTI, [5, -2, 7])

    def test_type_error(self):
        assert_raises(TypeError, self.hmm.baum_welch_mult, [[0, 1]], [2])

    def test_value_error(self):
        assert_raises(ValueError, self.hmm.baum_welch_mult, [0, 99, 1], [3])

# ===========================================================================
# 8. viterbi_teorico
# ===========================================================================
class TestViterbiTeorico:
    def setup(self):
        self.hmm = HMM(2, 2, A=A_vit, B=B_vit, pi=pi_vit)

    def test_known_result(self):
        path, _ = self.hmm.viterbi_teorico(OBS_SHORT)
        assert_true(path == [0, 0, 0, 0], f"got {path}")

    def test_agrees_with_optimized(self):
        po, pr_o = self.hmm.viterbi(OBS_SHORT, return_log=False)
        pt, pr_t = self.hmm.viterbi_teorico(OBS_SHORT)
        assert_true(po == pt and np.isclose(pr_o, pr_t, rtol=1e-5))

    def test_type_error(self):
        assert_raises(TypeError, self.hmm.viterbi_teorico, [[0, 1]])

    def test_value_error(self):
        assert_raises(ValueError, self.hmm.viterbi_teorico, [0, 99])

# ===========================================================================
# 9. forward_teorico
# ===========================================================================
class TestForwardTeorico:
    def setup(self):
        self.hmm = HMM(2, 2, A=A_fwd, B=B_fwd, pi=pi_fwd)

    def test_known_probability(self):
        _, prob = self.hmm.forward_teorico(OBS_SHORT)
        assert_true(round(prob, 6) == 0.071168, f"got {round(prob,6)}")

    def test_agrees_with_optimized(self):
        ao, lo = self.hmm.forward(OBS_SHORT, return_log=False)
        at, lt = self.hmm.forward_teorico(OBS_SHORT)
        assert_true(np.allclose(ao, at, rtol=1e-5) and np.isclose(lo, lt, rtol=1e-5))

    def test_type_error(self):
        assert_raises(TypeError, self.hmm.forward_teorico, [[0, 1]])

# ===========================================================================
# 10. backward_teorico
# ===========================================================================
class TestBackwardTeorico:
    def setup(self):
        self.hmm = HMM(2, 2, A=A_fwd, B=B_fwd, pi=pi_fwd)

    def test_shape_and_last_row(self):
        beta = self.hmm.backward_teorico(OBS_SHORT)
        assert_true(beta.shape == (len(OBS_SHORT), 2))
        assert_true(np.allclose(beta[-1], 1.0))

    def test_agrees_with_optimized(self):
        bo = self.hmm.backward(OBS_SHORT, return_log=False)
        bt = self.hmm.backward_teorico(OBS_SHORT)
        assert_true(np.allclose(bo, bt, rtol=1e-5))

    def test_type_error(self):
        assert_raises(TypeError, self.hmm.backward_teorico, [[0, 1]])

# ===========================================================================
# 11. forward_backward_teorico
# ===========================================================================
class TestForwardBackwardTeorico:
    def setup(self):
        self.hmm = HMM(2, 2, A=A_fwd, B=B_fwd, pi=pi_fwd)

    def test_rows_sum_to_one(self):
        gamma = self.hmm.forward_backward_teorico(OBS_SHORT)
        assert_true(np.allclose(gamma.sum(axis=1), 1.0))

    def test_agrees_with_optimized(self):
        go = self.hmm.forward_backward(OBS_SHORT, return_log=False)
        gt = self.hmm.forward_backward_teorico(OBS_SHORT)
        assert_true(np.allclose(go, gt, rtol=1e-5))

    def test_type_error(self):
        assert_raises(TypeError, self.hmm.forward_backward_teorico, [[0, 1]])

# ===========================================================================
# 12. baum_welch_teorico
# ===========================================================================
class TestBaumWelchTeorico:
    def setup(self):
        self.hmm = HMM(2, 2, A=A_fwd, B=B_fwd, pi=pi_fwd)

    def test_log_likelihood_improves(self):
        _, logL0 = self.hmm.forward(OBS_TRAIN, return_log=True)
        logLf = self.hmm.baum_welch_teorico(OBS_TRAIN, max_iter=20)
        assert_true(logLf > logL0, f"logLf={logLf:.4f} not > logL0={logL0:.4f}")

    def test_return_log_history(self):
        logL, hist = self.hmm.baum_welch_teorico(OBS_TRAIN, max_iter=10, return_log_history=True)
        assert_true(isinstance(hist, list) and logL == hist[-1])

    def test_parameters_stochastic_after_training(self):
        self.hmm.baum_welch_teorico(OBS_TRAIN, max_iter=10)
        assert_true(np.allclose(self.hmm.A.sum(axis=1), 1.0, atol=1e-9))
        assert_true(np.allclose(self.hmm.B.sum(axis=1), 1.0, atol=1e-9))
        assert_true(np.allclose(self.hmm.pi.sum(),       1.0, atol=1e-9))

    def test_empty_raises(self):
        assert_raises(ValueError, self.hmm.baum_welch_teorico, [])

    def test_type_error(self):
        assert_raises(TypeError, self.hmm.baum_welch_teorico, [[0, 1]])

# ===========================================================================
# 13. Cross-method consistency
# ===========================================================================
class TestCrossMethodConsistency:
    def setup(self):
        self.hmm = HMM(2, 2, A=A_fwd, B=B_fwd, pi=pi_fwd)

    def test_baum_welch_and_mult_agree_on_single_sequence(self):
        hs = HMM(2, 2, A=A_fwd.copy(), B=B_fwd.copy(), pi=pi_fwd.copy())
        hm = HMM(2, 2, A=A_fwd.copy(), B=B_fwd.copy(), pi=pi_fwd.copy())
        ls = hs.baum_welch(OBS_TRAIN, max_iter=30)
        lm = hm.baum_welch_mult(np.array(OBS_TRAIN), [len(OBS_TRAIN)], max_iter=30)
        assert_true(np.isclose(ls, lm, atol=1e-6), f"single={ls:.6f} vs mult={lm:.6f}")

    def test_viterbi_path_valid_after_training(self):
        self.hmm.baum_welch(OBS_TRAIN, max_iter=20)
        path, prob = self.hmm.viterbi(OBS_TRAIN)
        assert_true(len(path) == len(OBS_TRAIN))
        assert_true(all(0 <= s < 2 for s in path) and prob > 0.0)

    def test_gamma_mass_conserved(self):
        gamma = self.hmm.forward_backward(OBS_SHORT)
        assert_true(np.allclose(gamma.sum(axis=1), 1.0, atol=1e-9))


# ===========================================================================
# Entry point
# ===========================================================================
if __name__ == "__main__":
    import sys
    try:
        import pytest
        sys.exit(pytest.main([__file__, "-v"]))
    except ImportError:
        pass

    test_classes = [
        TestConstructor, TestViterbi, TestForward, TestBackward,
        TestForwardBackward, TestBaumWelch, TestBaumWelchMult,
        TestViterbiTeorico, TestForwardTeorico, TestBackwardTeorico,
        TestForwardBackwardTeorico, TestBaumWelchTeorico,
        TestCrossMethodConsistency,
    ]

    passed = failed = 0
    for cls in test_classes:
        instance = cls()
        methods = sorted(m for m in dir(cls) if m.startswith("test_"))
        for name in methods:
            instance.setup()
            try:
                getattr(instance, name)()
                print(f"  PASS  {cls.__name__}::{name}")
                passed += 1
            except Exception as exc:
                print(f"  FAIL  {cls.__name__}::{name}")
                print(f"        → {exc}")
                failed += 1

    total = passed + failed
    print(f"\n{'='*60}")
    print(f"  {passed}/{total} tests passed", end="")
    print(" ✓" if failed == 0 else f"  —  {failed} FAILED")
    print(f"{'='*60}")
    sys.exit(0 if failed == 0 else 1)
