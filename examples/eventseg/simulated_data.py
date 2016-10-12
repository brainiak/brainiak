"""Example of finding event segmentations on simulated data

This code generates simulated datasets that have temporally-clustered
structure (with the same series of latent event patterns). An event
segmentation is learned on the first dataset, and then we try to find the same
series of events in other datasets. We measure how well we find the latent
boundaries and the log-likelihood of the fits, and compare to a null model
in which the event order is randomly shuffled.
"""
import brainiak.eventseg.event
import numpy as np
from scipy import stats
import logging
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.DEBUG)


def generate_event_labels(T, K, length_std):
    event_labels = np.zeros(T, dtype=int)
    start_TR = 0
    for e in range(K - 1):
        length = round(
            ((T - start_TR) / (K - e)) * (1 + length_std * np.random.randn()))
        length = min(max(length, 1), T - start_TR - (K - e))
        event_labels[start_TR:(start_TR + length)] = e
        start_TR = start_TR + length
    event_labels[start_TR:] = K - 1

    return event_labels


def generate_data(V, T, event_labels, event_means, noise_std):
    simul_data = np.empty((V, T))
    for t in range(T):
        simul_data[:, t] = stats.multivariate_normal.rvs(
            event_means[:, event_labels[t]], cov=noise_std, size=1)

    simul_data = stats.zscore(simul_data, axis=1, ddof=1)
    return simul_data


# Parameters for creating small simulated datasets
V = 10
K = 10
T = 500
T2 = 300

# Generate the first dataset
np.random.seed(1)
event_means = np.random.randn(V, K)
event_labels = generate_event_labels(T, K, 0.1)
simul_data = generate_data(V, T, event_labels, event_means, 1)

# Find the events in this dataset
simul_seg = brainiak.eventseg.event.EventSegment(K)
simul_seg.fit(simul_data.T)

# Generate other datasets with the same underlying sequence of event
# patterns, and try to find matching events
test_loops = 10
bound_match = np.empty((2, test_loops))
LL = np.empty((2, test_loops))
for test_i in range(test_loops):
    # Generate data
    event_labels2 = generate_event_labels(T2, K, 0.5)
    simul_data2 = generate_data(V, T2, event_labels2, event_means, 0.1)

    # Find events matching previously-learned events
    gamma, LL[0, test_i] = simul_seg.find_events(simul_data2.T)
    est_events2 = np.argmax(gamma, axis=1)
    bound_match[0, test_i] = 1 - np.sum(abs(np.diff(event_labels2) -
                                            np.diff(est_events2))) / (2 * K)

    # Run again, but with the order of events shuffled so that it no longer
    # corresponds to the training data
    gamma, LL[1, test_i] = simul_seg.find_events(simul_data2.T, scramble=True)
    est_events2 = np.argmax(gamma, axis=1)
    bound_match[1, test_i] = 1 - np.sum(abs(np.diff(event_labels2) -
                                            np.diff(est_events2))) / (2 * K)

# Across the testing datasets, print how well we identify the true event
# boundaries and the log-likehoods in real vs. shuffled data
print("Boundary match: {:.2} (null: {:.2})".format(
    np.mean(bound_match[0, :]), np.mean(bound_match[1, :])))
print("Log-likelihood: {:.3} (null: {:.3})".format(
    np.mean(LL[0, :]), np.mean(LL[1, :])))

plt.figure()
plt.subplot(2, 1, 1)
plt.imshow(simul_data2, interpolation='nearest', cmap=plt.cm.bone,
           aspect='auto')
plt.xlabel('Timepoints')
plt.ylabel('Voxels')
plt.subplot(2, 1, 2)
gamma, LL[0, test_i] = simul_seg.find_events(simul_data2.T)
est_events2 = np.argmax(gamma, axis=1)
plt.plot(est_events2)
plt.xlabel('Timepoints')
plt.ylabel('Event label')
plt.show()
