import brainiak.eventseg.event
import numpy as np


def test_create_event_segmentation():
    es = brainiak.eventseg.event.EventSegment(5)
    assert es, "Invalid EventSegment instance"


def test_fit_shapes():
    K = 5
    V = 3
    T = 10
    es = brainiak.eventseg.event.EventSegment(K, n_iter=2)
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


def test_simple_boundary():
    es = brainiak.eventseg.event.EventSegment(2)
    sample_data = np.asarray([[1, 1, 1, 0, 0, 0, 0], [0, 0, 0, 1, 1, 1, 1]])
    es.fit(sample_data.T)

    events = np.argmax(es.segments_[0], axis=1)
    assert np.array_equal(events, [0, 0, 0, 1, 1, 1, 1]),\
        "Failed to correctly segment two events"


def test_event_transfer():
    es = brainiak.eventseg.event.EventSegment(2)
    es.set_event_patterns(np.asarray([[1, 0], [0, 1]]))
    sample_data = np.asarray([[1, 1, 1, 0, 0, 0, 0], [0, 0, 0, 1, 1, 1, 1]])
    seg = es.find_events(sample_data.T, np.asarray([1, 1]))[0]

    events = np.argmax(seg, axis=1)
    assert np.array_equal(events, [0, 0, 0, 1, 1, 1, 1]),\
        "Failed to correctly transfer two events to new data"


def test_weighted_var():
    es = brainiak.eventseg.event.EventSegment(2)

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
