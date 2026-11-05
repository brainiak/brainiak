"""Exercise noise-file parsing without generating a simulated volume."""

import numpy as np
import pytest

from brainiak.utils import fmrisim_real_time_generator as gen


class NoiseParametersLoaded(Exception):
    pass


@pytest.mark.parametrize("as_file", [False, True])
def test_literal_noise_parameters_reach_simulation(tmp_path, monkeypatch, as_file):
    template = np.ones((2, 2, 2))
    noise = b"{'fwhm': 4.0, 'matched': 1}"
    if as_file:
        path = tmp_path / "noise.txt"
        path.write_bytes(noise)
        noise = str(path)
    monkeypatch.setattr(gen, "_get_input_names", lambda settings:
                        (template, template, template, noise))
    monkeypatch.setattr(gen.sim, "mask_brain", lambda **kwargs:
                        (template, template))

    def capture_noise(**kwargs):
        assert kwargs["noise_dict"] == {"fwhm": 4.0, "matched": 0}
        raise NoiseParametersLoaded

    monkeypatch.setattr(gen.sim, "generate_noise", capture_noise)
    with pytest.raises(NoiseParametersLoaded):
        gen.generate_data(str(tmp_path / "output"), {"numTRs": 1})


@pytest.mark.parametrize("expression", [
    "{'fwhm': np.float64(4.0)}",
    "(__import__('pathlib').Path({marker!r}).write_text('executed'), {{}})[1]",
])
def test_noise_expressions_are_rejected(tmp_path, monkeypatch, expression):
    template = np.ones((2, 2, 2))
    marker = tmp_path / "executed.txt"
    # Only the marker expression needs formatting; the literal braces above
    # remain part of the NumPy-constructor input.
    if "{marker!r}" in expression:
        expression = expression.format(marker=str(marker))
    monkeypatch.setattr(gen, "_get_input_names", lambda settings:
                        (template, template, template, expression.encode()))
    monkeypatch.setattr(gen.sim, "mask_brain", lambda **kwargs:
                        (template, template))

    def unexpected_noise(**kwargs):
        pytest.fail("Expression reached noise generation")

    monkeypatch.setattr(gen.sim, "generate_noise", unexpected_noise)
    with pytest.raises(ValueError):
        gen.generate_data(str(tmp_path / "output"), {"numTRs": 1})
    assert not marker.exists()
