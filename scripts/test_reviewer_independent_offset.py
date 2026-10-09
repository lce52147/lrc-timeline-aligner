import unittest
from unittest.mock import patch
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import numpy as np

import reviewer_independent_offset as c
import r2_pipeline as p

class IndependentReviewerOffsetTests(unittest.TestCase):
    def features(self):
        time=np.arange(0.,10.,.025,dtype=np.float32)
        db=np.full(time.shape,-20.,dtype=np.float32)
        strength=np.zeros(time.shape,dtype=np.float32)
        for s in range(1,9):
            strength[np.argmin(abs(time-float(s)))]=1.
        return SimpleNamespace(frame_times=time,rms_db=db,onset_strength=strength)
    def test_physical_onset_anchor_calibration(self):
        f=self.features()
        peaks=c.anchor_peaks_from_vocal_features(f.frame_times,f.rms_db,f.onset_strength)
        self.assertEqual(len(peaks),8)
        times=[float(i)+.12 for i in range(1,9)]
        x=c.calibrate_reviewer_vocal_offset(times,peaks)
        self.assertEqual(x.status,"CALIBRATED_VOCAL_ONSETS")
        self.assertEqual(x.sample_count,8)
        self.assertAlmostEqual(x.offset_seconds,.12,places=4)
    def test_no_central_baseline_influence(self):
        f=self.features()
        with TemporaryDirectory() as tmp:
            file=Path(tmp)/'vocals.wav';file.write_bytes(b'fixture')
            rows=[{'entry':i,'sources':{'CUR':{'time':1000+i*40},
               'WX':{'time':i+.12}, 'XLSR':{'time':i+.20}}} for i in range(1,9)]
            payload={'songs':[{'id':'song','vocal_audio_path':str(file),'rows':rows}]}
            with patch.object(p.auto_lrc,'probe_duration',return_value=10.), \
                patch.object(p.auto_lrc,'analyze_vocal_onsets',return_value=f):
                side=p.build_r2_sidecar(payload,{})
            self.assertAlmostEqual(side['offsets']['WX'],.12,places=4)
            self.assertAlmostEqual(side['offsets']['XLSR'],.20,places=4)
            self.assertNotIn('HUBP',side['offsets'])
            self.assertFalse(side['offset_calibration_uses_central'])
            for row in rows:row['sources']['CUR']['time']=-8000
            with patch.object(p.auto_lrc,'probe_duration',return_value=10.), \
                patch.object(p.auto_lrc,'analyze_vocal_onsets',return_value=f):
                second=p.build_r2_sidecar(payload,{})
            self.assertEqual(side['offsets'],second['offsets'])
    def test_not_enough_acoustic_witnesses_fail_closed(self):
        x=c.calibrate_reviewer_vocal_offset([.12,1.12],[0,1])
        self.assertIsNone(x.offset_seconds)
        self.assertIn('INSUFFICIENT',x.status)
    def test_ambiguous_onset_dispersions_fail_closed(self):
        peaks=[float(i) for i in range(1,12)]
        times=[i+(.02 if i%2 else .28) for i in range(1,12)]
        x=c.calibrate_reviewer_vocal_offset(times,peaks,min_samples=8)
        self.assertIsNone(x.offset_seconds)
        self.assertEqual(x.status,"AMBIGUOUS_VOCAL_CLOCK_DISPERSION")
    def test_no_independent_audio_preserves_raw_reviewer_rows(self):
        rows=[{'entry':i,'sources':{'CUR':{'time':i+1},'WX':{'time':i+.01}}}
               for i in range(1,10)]
        v=p.build_r2_sidecar({'songs':[{'id':'song','rows':rows}]},{})
        self.assertEqual(v['offsets'],{})
        self.assertEqual(len(v['rows']),9)
        self.assertTrue(all(row['reviewer_times']['WX'] is not None for row in v['rows']))
if __name__=='__main__':unittest.main()
