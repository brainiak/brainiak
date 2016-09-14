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

    assert es.segments_[0].shape == (T,K), "Segmentation from fit " \
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
    es = brainiak.eventseg.event.EventSegment(2, n_iter=10)
    sample_data = np.asarray([[1, 1, 1, 0, 0, 0, 0], [0, 0, 0, 1, 1, 1, 1]])
    es.fit(sample_data.T)

    events = np.argmax(es.segments_[0], axis=1)
    assert np.array_equal(events, [0, 0, 0, 1, 1, 1, 1]),\
        "Failed to correctly segment two events"
