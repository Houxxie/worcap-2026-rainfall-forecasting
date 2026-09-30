"""Check pairing, reconstruction stage and complementary-error accounting."""
from pathlib import Path
import unittest
import numpy as np
import pandas as pd

from research.fixed_blend.evaluate import fixed_blend, error_products, summarize_products, paired
from research.workflow.config import normalize, WorkflowError
from research.workflow.test_workflow import example


class FixedBlendChecks(unittest.TestCase):
    def test_constant_reference_and_convex_bounds(self):
        h = np.array([0, 1, 10], dtype='float32')
        np.testing.assert_array_equal(fixed_blend(h, h), h)
        u = np.array([4, 0, 2], dtype='float32')
        blend = fixed_blend(h, u)
        self.assertEqual(blend.dtype, np.dtype('float64'))
        self.assertTrue(((blend >= np.minimum(h, u)) & (blend <= np.maximum(h, u))).all())
        with self.assertRaisesRegex(ValueError, 'nonnegative'):
            fixed_blend(h, [-1, 0, 2])
        with self.assertRaisesRegex(ValueError, 'different shapes'):
            fixed_blend(h, [[4, 0, 2]])
        with self.assertRaisesRegex(ValueError, 'nonfinite'):
            fixed_blend(h, [np.nan, 0, 2])

    def test_mse_identity_and_complementarity(self):
        y = np.full(4, 5.)
        h = y + np.array([1., -1., 2., -2.])
        u = y - 3 * (h - y)
        # Nonnegative construction with exactly cancelling weighted errors.
        y += 10; h += 10; u += 10
        result = summarize_products([error_products(y[:2], h[:2], u[:2]), error_products(y[2:], h[2:], u[2:])])
        np.testing.assert_array_equal(fixed_blend(h, u), y)
        self.assertAlmostEqual(result['centered_error_correlation'], -1)
        self.assertAlmostEqual(result['blend_sse_from_products'], 0)
        self.assertAlmostEqual(result['blend_sse_from_disagreement'], 0)

    def test_pairing_and_signed_vs_absolute_bias(self):
        frame = pd.DataFrame([dict(model='hybrid', year=1, n=4, rmse=2., mae=1., bias=-.2, sse=16.),
                              dict(model='blend', year=1, n=4, rmse=1., mae=.8, bias=-.1, sse=4.)])
        result = paired(frame, ['year']).iloc[0]
        self.assertAlmostEqual(result.delta_bias, .1)
        self.assertAlmostEqual(result.delta_absolute_bias, -.1)
        frame.loc[1, 'n'] = 3
        with self.assertRaisesRegex(ValueError, 'Unpaired'):
            paired(frame, ['year'])

    def test_config_rejects_training_and_weight_search(self):
        config = example(Path('/example'))
        config['experiment'] = 'fixed_blend_v1'
        self.assertEqual(normalize(config, Path.cwd())['experiment'], 'fixed_blend_v1')
        config['weights'] = [.5, .5]
        with self.assertRaisesRegex(WorkflowError, 'Unknown configuration'):
            normalize(config, Path.cwd())
        del config['weights']
        config['mode'] = 'train'
        with self.assertRaisesRegex(WorkflowError, 'saved maps only'):
            normalize(config, Path.cwd())


if __name__ == '__main__':
    unittest.main()
