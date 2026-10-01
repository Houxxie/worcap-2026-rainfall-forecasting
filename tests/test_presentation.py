import unittest
import pandas as pd
from pandas.testing import assert_frame_equal
from research.common.presentation import display_frame, model_label

class PresentationTests(unittest.TestCase):
    def test_display_copy_preserves_saved_keys_and_scores(self):
        frame = pd.DataFrame({'modelo': ['hybrid', 'hybrid_sst', 'unet', 'unknown'], 'rmse': [1.1, 1.2, 1.3, 1.4]})
        original = frame.copy(deep=True)
        shown = display_frame(frame)
        self.assertEqual(shown.modelo.tolist(), ['Reference model', 'Reference model + SST', 'U-Net', 'unknown'])
        self.assertEqual(shown.rmse.tolist(), original.rmse.tolist())
        shown.loc[0, 'rmse'] = 99
        assert_frame_equal(frame, original)

    def test_candidate_and_reference_remain_distinct(self):
        frame = pd.DataFrame({'model': ['blend_reference', 'blend_sst'], 'reference': ['hybrid', 'hybrid']})
        shown = display_frame(frame)
        self.assertEqual(shown.model.nunique(), 2)
        self.assertEqual(shown.reference.tolist(), ['Reference model', 'Reference model'])
        self.assertEqual(model_label('future_model'), 'future_model')

if __name__ == '__main__':
    unittest.main()
