from brainiak.eventseg.event import EventSegment
from scipy.special import comb
import numpy as np
import pytest
from sklearn.exceptions import NotFittedError


def test_create_event_segmentation():
    es = EventSegment(5)
    assert es, "Invalid EventSegment instance"


def test_fit_shapes():
    K = 5
    V = 3
    T = 10
    es = EventSegment(K, n_iter=2)
    sample_data = np.random.rand(V, T)
    es.fit(sample_data.T)

    assert es.segments_[0].shape == (T, K), "Segmentation from fit " \
                                            "has incorrect shape"
    assert np.isclose(np.sum(es.segments_[0], axis=1), np.ones(T)).all(), \
        "Segmentation from learn_events not correctly normalized"

    T2 = 15
    sample_data2 = np.random.rand(V, T2)
    test_segments, test_ll = es.find_events(sample_data2.T)

    assert test_segments.shape == (T2, K), "Segmentation from find_events " \
                                           "has incorrect shape"
    assert np.isclose(np.sum(test_segments, axis=1), np.ones(T2)).all(), \
        "Segmentation from find_events not correctly normalized"

    es_invalid = EventSegment(K)
    with pytest.raises(ValueError):
        es_invalid.model_prior(K-1)
        # ``with`` block is about to end with no error.
        pytest.fail("T < K should cause error")
    with pytest.raises(ValueError):
        es_invalid.set_event_patterns(np.zeros((V, K-1)))
        pytest.fail("#Events < K should cause error")


def test_simple_boundary():
    es = EventSegment(2)
    random_state = np.random.RandomState(0)

    sample_data = np.array([[1, 1, 1, 0, 0, 0, 0], [0, 0, 0, 1, 1, 1, 1]]) + \
        random_state.rand(2, 7) * 10
    es.fit(sample_data.T)

    events = np.argmax(es.segments_[0], axis=1)
    assert np.array_equal(events, [0, 0, 0, 1, 1, 1, 1]),\
        "Failed to correctly segment two events"

    events_predict = es.predict(sample_data.T)
    assert np.array_equal(events_predict, [0, 0, 0, 1, 1, 1, 1]), \
        "Error in predict interface"


def test_event_transfer():
    es = EventSegment(2)
    sample_data = np.asarray([[1, 1, 1, 0, 0, 0, 0], [0, 0, 0, 1, 1, 1, 1]])

    with pytest.raises(NotFittedError):
        seg = es.find_events(sample_data.T)[0]
        pytest.fail("Should need to set variance")

    with pytest.raises(NotFittedError):
        seg = es.find_events(sample_data.T, np.asarray([1, 1]))[0]
        pytest.fail("Should need to set patterns")

    es.set_event_patterns(np.asarray([[1, 0], [0, 1]]))
    seg = es.find_events(sample_data.T, np.asarray([1, 1]))[0]

    events = np.argmax(seg, axis=1)
    assert np.array_equal(events, [0, 0, 0, 1, 1, 1, 1]),\
        "Failed to correctly transfer two events to new data"


def test_weighted_var():
    es = EventSegment(2)

    D = np.zeros((8, 4))
    for t in range(4):
        D[t, :] = (1/np.sqrt(4/3)) * np.array([-1, -1, 1, 1])
    for t in range(4, 8):
        D[t, :] = (1 / np.sqrt(4 / 3)) * np.array([1, 1, -1, -1])
    mean_pat = D[[0, 4], :].T

    weights = np.zeros((8, 2))
    weights[:, 0] = [1, 1, 1, 1, 0, 0, 0, 0]
    weights[:, 1] = [0, 0, 0, 0, 1, 1, 1, 1]
    assert np.array_equal(
        es.calc_weighted_event_var(D, weights, mean_pat), [0, 0]),\
        "Failed to compute variance with 0/1 weights"

    weights[:, 0] = [1, 1, 1, 1, 0.5, 0.5, 0.5, 0.5]
    weights[:, 1] = [0.5, 0.5, 0.5, 0.5, 1, 1, 1, 1]
    true_var = (4 * 0.5 * 12)/(6 - 5/6) * np.ones(2) / 4
    assert np.allclose(
        es.calc_weighted_event_var(D, weights, mean_pat), true_var),\
        "Failed to compute variance with fractional weights"


def test_sym():
    es = EventSegment(4)

    evpat = np.repeat(np.arange(10).reshape(-1, 1), 4, axis=1)
    es.set_event_patterns(evpat)

    D = np.repeat(np.arange(10).reshape(1, -1), 20, axis=0)
    ev = es.find_events(D, var=1)[0]

    # Check that events 1-4 and 2-3 are symmetric
    assert np.all(np.isclose(ev[:, :2], np.fliplr(np.flipud(ev[:, 2:])))),\
        "Fit with constant data is not symmetric"


def test_chains():
    es = EventSegment(5, event_chains=np.array(['A', 'A', 'B', 'B', 'B']))
    sample_data = np.array([[0, 0, 0], [1, 1, 1]])

    with pytest.raises(RuntimeError):
        seg = es.fit(sample_data.T)[0]
        pytest.fail("Can't use fit() with event chains")

    es.set_event_patterns(np.array([[1, 1, 0, 0, 0],
                                    [0, 0, 1, 1, 1]]))
    seg = es.find_events(sample_data.T, 0.1)[0]

    ev = np.nonzero(seg > 0.99)[1]
    assert np.array_equal(ev, [2, 3, 4]),\
        "Failed to fit with multiple chains"


def test_prior():
    K = 10
    T = 100

    es = EventSegment(K)
    mp = es.model_prior(T)[0]

    p_bound = np.zeros((T, K-1))
    norm = comb(T-1, K-1)
    for t in range(T-1):
        for k in range(K-1):
            # See supplementary material of Neuron paper
            # https://doi.org/10.1016/j.neuron.2017.06.041
            p_bound[t+1, k] = comb(t, k) * comb(T-t-2, K-k-2) / norm
    p_bound = np.cumsum(p_bound, axis=0)

    mp_gt = np.zeros((T, K))
    for k in range(K):
        if k == 0:
            mp_gt[:, k] = 1 - p_bound[:, 0]
        elif k == K - 1:
            mp_gt[:, k] = p_bound[:, k-1]
        else:
            mp_gt[:, k] = p_bound[:, k-1] - p_bound[:, k]

    assert np.all(np.isclose(mp, mp_gt)),\
        "Prior does not match analytic solution"


def test_split_merge():
    ev = np.array(
        [0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3,
         3, 3, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4])
    random_state = np.random.RandomState(0)
    ev_pat = random_state.rand(5, 10)
    D = np.zeros((len(ev), 10))
    for t in range(len(ev)):
        D[t, :] = ev_pat[ev[t], :] + 0.1*random_state.rand(10)

    hmm_sm = EventSegment(5, split_merge=True, split_merge_proposals=2)
    hmm_sm.fit(D)
    hmm_events = np.argmax(hmm_sm.segments_[0], axis=1)

    assert np.all(ev == hmm_events),\
        "Merge/split fails to find highly uneven events"


def test_sym_ll():
    ev = np.array([0, 0, 0, 1, 1, 1, 1, 1, 1, 2, 2])
    random_state = np.random.RandomState(0)
    ev_pat = random_state.rand(3, 10)
    D_forward = np.zeros((len(ev), 10))
    for t in range(len(ev)):
        D_forward[t, :] = ev_pat[ev[t], :] + 0.1 * random_state.rand(10)
    D_backward = np.flip(D_forward, axis=0)

    hmm_forward = EventSegment(3)
    hmm_forward.set_event_patterns(ev_pat.T)
    _, ll_forward = hmm_forward.find_events(D_forward, var=1)

    hmm_backward = EventSegment(3)
    hmm_backward.set_event_patterns(np.flip(ev_pat.T, axis=1))
    _, ll_backward = hmm_backward.find_events(D_backward, var=1)

    assert np.allclose(ll_forward, ll_backward),\
        "Log-likelihood not symmetric forward/backward"
