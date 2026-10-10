"""H2 synthetic (F1-F4) with pure fake final-state rows, no GPU.
"""
import math,tempfile,types,pathlib,unittest,sys
from unittest.mock import patch
import numpy as np
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parents[1]/'scripts'))
import auto_lrc as l
class H2Tests(unittest.TestCase):
    def make(self,times,words,source=None):
        decisions=[types.SimpleNamespace(written_time=types.SimpleNamespace(seconds=t),
            candidate=types.SimpleNamespace(source=source[i] if source else 'ctc-current')) for i,t in enumerate(times)]
        state=types.SimpleNamespace(decisions=decisions)
        entries=[l.LyricEntry([name]) for name in words]
        return state,entries
    def test_f1_vocal_absent_absolute_rms(self):
        state,entries=self.make([.2,1.25,2.2],['A','B','C'])
        x=np.zeros(4*l.SAMPLE_RATE,dtype=np.float32)
        x[0:l.SAMPLE_RATE]=.25
        x[2*l.SAMPLE_RATE:3*l.SAMPLE_RATE]=.25
        a={'ctc_audio_source':'vocal-stem','assignments':[{} for _ in entries]}
        with tempfile.TemporaryDirectory() as tmp:
            path=pathlib.Path(tmp)/'vocals.mp3';path.write_bytes(b'mock')
            a['ctc_audio_path']=str(path)
            features=types.SimpleNamespace(segments=[(0,1),(2,3)])
            with patch.object(l,'decode_audio',return_value=x),patch.object(l,'analyze_audio',return_value=features):
                g=l.compute_final_timing_self_check(a,state,entries)
        self.assertEqual(g['vocal_status'],'checked')
        self.assertIn('F1-vocal-absent',g['flags_by_entry']['2'])
        self.assertNotIn('1',g['flags_by_entry'])
    def test_f2_repeated_content_identity(self):
        state,entries=self.make([2,7,12],["I'm singing","I'm singing","different"],
            ['raw-vocal-independent-fusion','ctc-current','ctc-current'])
        a={'assignments':[{}, {}, {}],'ctc_audio_source':'mix'}
        g=l.compute_final_timing_self_check(a,state,entries)
        self.assertIn('F2-repeated-content-identity',g['flags_by_entry']['1'])
        self.assertNotIn('2',g['flags_by_entry'])
    def test_f3_compressed_run(self):
        times=[1,4,7,7.2,7.4,10,14,18]
        state,entries=self.make(times,[f'line_{i}' for i in range(len(times))])
        g=l.compute_final_timing_self_check({'assignments':[{} for _ in times]},state,entries)
        self.assertIn('F3-compressed-or-nonmonotonic',g['flags_by_entry']['3'])
        self.assertIn('F3-compressed-or-nonmonotonic',g['flags_by_entry']['5'])
    def test_f4_large_ctc_disagreement(self):
        state,entries=self.make([3,8,13],['a','b','c'])
        g=l.compute_final_timing_self_check({'assignments':[{}, {'ctc_first_token_start':4.0},{}]},state,entries)
        self.assertIn('F4-ctc-final-disagreement',g['flags_by_entry']['2'])
    def test_f4_preserved_global_ctc_source(self):
        state,entries=self.make([160.89],['dream'])
        report={'assignments':[{'ctc_first_token_start':160.89,
               'global_ctc_first_token_start':164.79}]}
        result=l.compute_final_timing_self_check(report,state,entries)
        self.assertIn('F4-ctc-final-disagreement',result['flags_by_entry']['1'])

    def test_f5_dream_row28_repeated_untrusted_flag_without_retiming(self):
        words=[f'line_{i}' for i in range(30)]
        words[12]=words[27]="I'm dreaming without you"
        times=[float(i)*6 for i in range(30)]
        times[27]=160.89
        state,entries=self.make(times,words)
        assignments=[{'ctc_first_token_start':t,'global_ctc_first_token_start':t}
                     for t in times]
        assignments[27]['global_ctc_first_token_start']=164.79
        assignments[28]['global_ctc_first_token_start']=times[28]+3.9
        report={'assignments':assignments,
                'segment_skeleton_status':{'status':'SEGMENT_CURRENT'}}
        result=l.compute_final_timing_self_check(report,state,entries)
        self.assertIn('skeleton_global_ctc_disagreement',result['flags_by_entry']['28'])
        self.assertNotIn('skeleton_global_ctc_disagreement',
                         result['flags_by_entry'].get('29',[]))
        self.assertEqual(state.decisions[27].written_time.seconds,160.89)
        report['segment_skeleton_status']={'status':'FALLBACK_CTC'}
        fallback=l.compute_final_timing_self_check(report,state,entries)
        self.assertNotIn('skeleton_global_ctc_disagreement',fallback['flags_by_entry']['28'])

    def test_f3_nonmonotonic_two_lines(self):
        state,entries=self.make([4,3],['a','b'])
        data=l.compute_final_timing_self_check({'assignments':[{},{}]},state,entries)
        self.assertIn('F3-nonmonotonic',data['flags_by_entry']['1'])
        self.assertIn('F3-nonmonotonic',data['flags_by_entry']['2'])
    def test_no_stem_skip_log(self):
        state,entries=self.make([2,4],['a','b'])
        x=l.compute_final_timing_self_check({'assignments':[{},{}]},state,entries)
        self.assertEqual(x['vocal_status'],'skipped-no-stem')
    def test_no_time_mutation(self):
        state,entries=self.make([2,4,6],['a','a','c'],['raw-vocal-independent-fusion']*3)
        before=[x.written_time.seconds for x in state.decisions]
        l.compute_final_timing_self_check({'assignments':[{}, {}, {}]},state,entries)
        self.assertEqual(before,[x.written_time.seconds for x in state.decisions])
if __name__=='__main__':unittest.main()
