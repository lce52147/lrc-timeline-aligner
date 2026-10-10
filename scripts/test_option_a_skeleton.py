"""Option A: monotone many-to-one repeat-agnostic DP and whole-song rollback."""
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent))
from segment_skeleton import plan_segment_skeleton,validate_local_ctc
from segment_skeleton_production import run_segment_skeleton


class OptionADPTests(unittest.TestCase):
    def test_p4_common_ctc_score_mapping(self):
        from ctc_confidence import ctc_alignment_confidence
        for score in (-.01,0.,.0299,.03,.0799,.08,.2,1.):
            expected=.90 if score>=.08 else .70 if score>=.03 else .35
            self.assertEqual(ctc_alignment_confidence(score),expected)
            self.assertEqual(ctc_alignment_confidence(score,has_start=False),0.)

    def test_p1_raw_asr_bigram_content_penalty(self):
        from segment_skeleton import _dice_bigram,_asr_for_passages,_text_for_match
        from segment_structure import VocalPassage
        self.assertAlmostEqual(_dice_bigram('dreaming','dreaming'),1.0)
        self.assertEqual(_dice_bigram('dreaming','zxy'),0.0)
        self.assertEqual(_text_for_match('ＡｌＰＨＡ!'), 'alpha')
        passages=(VocalPassage(1,3),VocalPassage(6,8))
        self.assertEqual(_asr_for_passages(passages,
            ((1.1,2.9,'first'),(6.1,7.9,'second'))),('first','second'))

    def test_p1_no_asr_must_preserve_dp(self):
        texts=['verse','repeat','repeat','outro']
        seg=[(1,4),(6,9),(11,14),(16,19)]
        baseline=plan_segment_skeleton(seg,duration=22.,lyric_texts=texts)
        none=plan_segment_skeleton(seg,duration=22.,lyric_texts=texts,raw_asr_events=())
        self.assertEqual(baseline,none)

    def test_p1_dp_content_changes_group_choice(self):
        seg=[(1,4),(6,9),(11,14)]
        words=['alpha','beta']
        blank=plan_segment_skeleton(seg,duration=22.,lyric_texts=words)
        events=[(1,4,'junk'),(6,9,'alpha'),(11,14,'beta')]
        content=plan_segment_skeleton(seg,duration=22.,lyric_texts=words,
                                      raw_asr_events=events)
        self.assertEqual(blank.status,'READY')
        self.assertEqual(content.status,'READY')
        self.assertEqual([x.passage_index for x in blank.assignments],[0,2])
        self.assertEqual([x.passage_index for x in content.assignments],[1,2])

    def test_p1_repetitions_in_order(self):
        seg=[(1,4),(6,9),(11,14),(16,19)]
        words=['repeat','repeat','repeat','end']
        raw=[(1,3,'repeat'),(6,8,'repeat'),(11,13,'repeat'),(16,18,'end')]
        result=plan_segment_skeleton(seg,duration=21.,lyric_texts=words,raw_asr_events=raw)
        self.assertEqual(result.status,'READY')
        ids=[x.passage_index for x in result.assignments]
        self.assertEqual(ids,sorted(ids))

    def test_single_lyric_three_times_requires_no_unique_repeat_anchor(self):
        texts=["hello refrain"]*3+["verse words"]*2
        audio=[(1,3),(6,8),(11,13),(16,20)]
        result=plan_segment_skeleton(audio,duration=22.,lyric_texts=texts)
        self.assertEqual(result.status,"READY")
        self.assertEqual(len(result.assignments),len(texts))
        self.assertTrue(all(b.initial_time_s>a.initial_time_s
                 for a,b in zip(result.assignments,result.assignments[1:])))
        self.assertTrue(all(x.confidence>=0 for x in result.assignments))
    def test_two_repeated_choruses_without_anchors(self):
        texts=["chorus a","chorus b","verse a","verse b",
               "chorus a","chorus b","outro"]
        seg=[(1,4),(6,9),(11,15),(17,20),(22,26)]
        result=plan_segment_skeleton(seg,duration=30.,lyric_texts=texts)
        self.assertEqual(result.status,"READY")
        self.assertEqual(len(result.assignments),7)
    def test_continuous_vocal_dominance_falls_back_whole_song(self):
        result=plan_segment_skeleton([(1,95)],duration=100,
            lyric_texts=["line"]*10)
        self.assertEqual(result.status,"FALLBACK_CTC")
    def test_too_few_passages_falls_back_whole_song(self):
        result=plan_segment_skeleton([(1,3),(5,7)],duration=20,
            lyric_texts=["a"]*20)
        self.assertEqual(result.reason,"TOO_FEW_PASSAGES")
    def test_nonmonotonic_local_gpu_failure(self):
        plan=plan_segment_skeleton([(1,3),(5,7)],duration=12,
            lyric_texts=["hello","world"])
        self.assertEqual(plan.status,"READY")
        rows=[
          {'start':5.2,'ctc_score':.5,'token_spans':[{'start':5.2,'end':5.3,'score':.5,'char':'h'}]},
          {'start':5.1,'ctc_score':.5,'token_spans':[{'start':5.1,'end':5.2,'score':.5,'char':'w'}]}
        ]
        ok,reason=validate_local_ctc(plan,rows,duration=12)
        self.assertFalse(ok)
        self.assertIn(reason,('LOCAL_NONMONOTONIC','LOCAL_OUTSIDE_PASSAGE'))


class OptionAProductionTests(unittest.TestCase):
    def case(self,folder,runner=None,fail=False,repeat_conflict=False):
        stem=folder/'vocals.mp3';stem.write_bytes(b'fake-stem')
        python=folder/'python.exe';python.write_bytes(b'exe')
        helper=folder/'ctc_align.py';helper.write_bytes(b'ctc')
        texts=["aaa","bbb","ccc","ddd"]
        times=[1.2,2.2,8.2,8.8] if repeat_conflict else [1.2,2.2,4.2,5.2]
        if repeat_conflict:
            texts=["aaa","bbb","aaa","ddd"]
        report={'backend':'ctc','ctc_audio_source':'vocal-stem',
                'assignments':[{'entry':i+1,'timestamp':t,'ctc_score':.03,
                   'ctc_first_token_start':t,
                   'ctc_token_spans':[{'start':t,'end':t+.1,'score':.5,'char':'a'}]}
                   for i,t in enumerate(times)],'ctc_acoustic_backtracks':[{'row':1}]}
        def fake_runner(cmd,timeout):
            start=float(cmd[cmd.index('--start')+1]);out=Path(cmd[cmd.index('--output')+1])
            lyrics=json.loads(Path(cmd[cmd.index('--transcript')+1]).read_text(encoding='utf8'))
            content=[]
            for j,word in enumerate(lyrics):
                t=round(start+.45+.6*j,3)
                if fail and len(content)==1:t=start-.1
                content.append({'start':t,'ctc_score':(.10 if '_neighbor' in out.name else .5),'token_spans':[{'start':t,'end':t+.1,'score':.8,'char':'a'}],
                'romaji':word})
            out.write_text(json.dumps({'entries':content}),encoding='utf8')
            return SimpleNamespace(returncode=0,stderr='')
        fake=SimpleNamespace(
            normalize_match_text=lambda t:str(t).strip().casefold(),
            analyze_audio=lambda stem,duration: SimpleNamespace(
                segments=[(1,3),(4,6)]),
            annotate_ctc_boundary_evidence=lambda assignments:None,
            update_report_confidence_metrics=lambda report:None)
        return run_segment_skeleton(report=report,timestamps=times,
            lyric_texts=texts,vocal_stem=stem,duration=9.,
            audio_api=fake,python_exe=python,ctc_helper=helper,
            helper_runner=runner or fake_runner,device='cuda')
    def test_all_song_segment_current_atomic_success(self):
        with tempfile.TemporaryDirectory() as d:
            t,r,x=self.case(Path(d))
            self.assertEqual(x['status'],'SEGMENT_CURRENT')
            self.assertEqual(len(t),4)
            self.assertTrue(all(b>a for a,b in zip(t,t[1:])))
            self.assertNotIn('ctc_acoustic_backtracks',r)
            self.assertEqual(r['global_ctc_diagnostics']['ctc_acoustic_backtracks'],[{'row':1}])
            self.assertEqual([z['global_ctc_first_token_start'] for z in r['assignments']],
                             [1.2,2.2,4.2,5.2])
            self.assertEqual([z['global_ctc_score'] for z in r['assignments']],[.03]*4)
            self.assertTrue(all(z['global_ctc_token_spans'][0]['char']=='a'
                             for z in r['assignments']))
            self.assertTrue(all(z['timing_repair']=='segment-skeleton-local-ctc'
                               for z in r['assignments']))
    def test_repeated_global_ctc_conflict_rolls_back_atomically(self):
        with tempfile.TemporaryDirectory() as d:
            times=[1.2,2.2,8.2,8.8]
            result,report,status=self.case(Path(d),repeat_conflict=True)
            self.assertEqual(status['status'],'FALLBACK_CTC')
            self.assertEqual(status['reason'],'REPEATED_SKELETON_GLOBAL_CTC_CONFLICT')
            self.assertEqual(result,times)
            self.assertEqual([x['timestamp'] for x in report['assignments']],times)

    def test_any_helper_failure_retains_complete_global_original(self):
        with tempfile.TemporaryDirectory() as d:
            def fail(cmd,timeout):
                return SimpleNamespace(returncode=1,stderr='model-failed')
            t,r,x=self.case(Path(d),runner=fail)
            self.assertEqual(x['status'],'FALLBACK_CTC')
            self.assertEqual(t,[1.2,2.2,4.2,5.2])
            self.assertEqual([z['timestamp'] for z in r['assignments']],t)
    def test_flag_explicit_ctc_mode_default_is_segment(self):
        import auto_lrc as a
        args=a.build_parser().parse_args(['file.flac'])
        self.assertEqual(args.skeleton,'segment')
        args=a.build_parser().parse_args(['file.flac','--skeleton','ctc'])
        self.assertEqual(args.skeleton,'ctc')


if __name__=='__main__':unittest.main()