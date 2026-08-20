"""Contract of :class:`TemperatureLikelihood`'s handling of the *shared* residual.

The evaluator's own arithmetic needs a populated sampler state on a GPU and is checked
by the live run of section 9 of ``_dev/prior_tempering.md``. What is checked here is the
part that has consequences *outside* the model move, and that a live run would not
report as a failure -- it would report it as a slightly wrong chain:

* the shared analysis containers hold one residual per walker, built by subtracting the
  coldest temperature's templates, and every other move reads it. A swap that moves a
  state into temperature 0 invalidates it, and :meth:`restore_residual` is what puts it
  back. Getting this wrong does not raise; it silently hands the galactic-binary move
  data that no longer matches ``coords[0]``.
* ``data_splits`` tells the waveform generator which device each template's buffer lives
  on. A zero there is not a "no split" marker but *device 0*, so on any other device
  every template is dropped and the injection writes nothing -- again silently.

Both are exercised against a mock generator, so no waveform is produced. Importing
``lisatools.globalfit`` pulls in CuPy and GBGPU, so the module still needs a CUDA
capable node to import at all.
"""

import numpy as np
import pytest

cupy = pytest.importorskip("cupy")
if not cupy.cuda.is_available():
    # importing lisatools.globalfit initialises CUDA and aborts without a device,
    # which pytest cannot turn into a skip -- so bail out before importing it
    pytest.skip("requires a CUDA device", allow_module_level=True)

hypermove = pytest.importorskip("lisatools.globalfit.moves.hypermove")

TemperatureLikelihood = hypermove.TemperatureLikelihood


NWALKERS = 3
NCHANNELS = 2
DATA_LENGTH = 4
NLEAVES = 5
NDIM = 2
DEVICE = 3  # deliberately not 0, which is what the bug this pins looked like


class MockWaveGen:
    """Records its calls and writes a deterministic, invertible "template"."""

    def __init__(self, block: int, gpus):
        self.block = block
        self.gpus = gpus
        self.calls = []

    def generate_global_template(
        self, params, group_index, templates, data_length, factors, data_splits, **kwargs
    ):
        self.calls.append(dict(data_splits=np.asarray(data_splits), n=len(group_index)))
        target = templates[0]
        for leaf in range(len(group_index)):
            config = int(group_index[leaf])
            start = config * self.block
            # linear in the coordinate, so that a residual rebuilt from the wrong
            # coordinates differs from one rebuilt from the right ones
            target[start : start + self.block] += float(factors[leaf]) * float(
                params[leaf, 0]
            )


class MockACS:
    """The bare surface of ``AnalysisContainerArray`` that the residual path touches."""

    def __init__(self):
        self.xp = np
        self.nchannels = NCHANNELS
        self.data_length = DATA_LENGTH
        self.end_shape = (DATA_LENGTH,)
        self.gpus = [DEVICE]
        self.linear_data_arr = [
            np.zeros(NWALKERS * NCHANNELS * DATA_LENGTH, dtype=np.complex128)
        ]

    def __len__(self):
        return NWALKERS


def build(rng=None):
    acs = MockACS()
    block = NCHANNELS * DATA_LENGTH
    wave_gen = MockWaveGen(block, acs.gpus)
    evaluator = TemperatureLikelihood(acs, wave_gen, waveform_kwargs={})
    return evaluator, acs, wave_gen


def coords_and_inds(seed: int):
    rng = np.random.default_rng(seed)
    coords = rng.normal(size=(NWALKERS, NLEAVES, NDIM))
    inds = rng.random(size=(NWALKERS, NLEAVES)) > 0.4
    inds[:, 0] = True  # never leave a walker with nothing alive
    return coords, inds


# ----------------------------------------------------------------------
# the device the templates are routed to
# ----------------------------------------------------------------------


def test_templates_are_routed_to_the_containers_device():
    """``data_splits`` carries the device id, not a zero.

    On a device other than 0 a zero here matches nothing, the generator keeps no
    template, and the injection is a silent no-op.
    """
    evaluator, _, wave_gen = build()
    coords, inds = coords_and_inds(0)

    evaluator.refresh_data(coords, inds)

    assert wave_gen.calls, "refresh_data did not inject anything"
    assert np.all(wave_gen.calls[-1]["data_splits"] == DEVICE)


def test_a_generator_on_another_device_is_refused():
    """Templates generated on one device cannot be written into a buffer on another."""
    acs = MockACS()
    wave_gen = MockWaveGen(NCHANNELS * DATA_LENGTH, gpus=[DEVICE + 1])
    with pytest.raises(ValueError, match="same single"):
        TemperatureLikelihood(acs, wave_gen, waveform_kwargs={})


# ----------------------------------------------------------------------
# the shared residual
# ----------------------------------------------------------------------


def test_restoring_the_same_state_returns_the_original_residual():
    """refresh_data then restore_residual is the identity when nothing was swapped."""
    evaluator, acs, _ = build()
    coords, inds = coords_and_inds(1)
    original = np.random.default_rng(2).normal(size=acs.linear_data_arr[0].shape)
    acs.linear_data_arr[0][:] = original

    evaluator.refresh_data(coords, inds)
    evaluator.restore_residual(coords, inds)

    assert np.allclose(acs.linear_data_arr[0], original)


def test_restoring_a_new_state_rebuilds_rather_than_undoing():
    """The residual is rebuilt from the recovered data, not patched incrementally.

    The distinction matters because a patch would have to know which templates were
    subtracted last, which is exactly the bookkeeping a swap has just invalidated.
    """
    evaluator, acs, _ = build()
    old_coords, old_inds = coords_and_inds(3)
    new_coords, new_inds = coords_and_inds(4)
    original = np.random.default_rng(5).normal(size=acs.linear_data_arr[0].shape)
    acs.linear_data_arr[0][:] = original

    evaluator.refresh_data(old_coords, old_inds)
    data = evaluator._base_data.copy()
    evaluator.restore_residual(new_coords, new_inds)

    # residual == data - templates(new state), independently of what was there before
    expected = data.copy()
    block = NCHANNELS * DATA_LENGTH
    for walker in range(NWALKERS):
        subtract = new_coords[walker][new_inds[walker]][:, 0].sum()
        expected[walker * block : (walker + 1) * block] -= subtract
    assert np.allclose(acs.linear_data_arr[0], expected)


def test_restoring_before_the_data_is_recovered_is_refused():
    """Without the data there is nothing to rebuild from, and a silent no-op would be
    the worst possible outcome."""
    evaluator, _, _ = build()
    coords, inds = coords_and_inds(6)
    with pytest.raises(RuntimeError, match="refresh_data"):
        evaluator.restore_residual(coords, inds)
