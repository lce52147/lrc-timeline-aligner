"""S5 performance guard may only defer NEW evidence, never forge or edit it."""
from dataclasses import dataclass
from pathlib import Path
import sys,unittest
sys.path.insert(0,str(Path(__file__).resolve().parent))
from local_whisper_incremental_budget import budget_incremental_whisper_requests


@dataclass(frozen=True)
class Req:
    entry_index:int
    context_entry_indexes:tuple[int,...]
    window_start:float
    window_end:float
    request_revision:str


def key(x):
    return 'local-whisper:'+x.request_revision


class EvidenceBudgetTests(unittest.TestCase):
    def items(self,n=30,contexts=3):
        return tuple(
            Req(i,tuple(range(max(0,i-context),i+1)),float(i),
                float(i)+2+context, f'r{i}-{context}')
            for i in range(n) for context in range(contexts)
        )
    def test_first_pass_is_unmodified(self):
        req=self.items()
        self.assertEqual(budget_incremental_whisper_requests(req,{'ctc:a':1},key,maximum_new=10),
                         (req,0))
    def test_later_pass_preserves_all_cached_requests(self):
        req=self.items()
        cache={key(x):'done' for x in req if x.entry_index%4==0}
        kept,removed=budget_incremental_whisper_requests(req,cache,key,maximum_new=9)
        self.assertEqual(len(kept),len([r for r in req if key(r) in cache])+9)
        self.assertEqual(removed,len(req)-len(kept))
        self.assertTrue(all(x in kept for x in req if key(x) in cache))
    def test_representative_new_probes_spread_to_late_rows(self):
        req=self.items(52)
        cache={'local-whisper:old':'done'}
        kept,deferred=budget_incremental_whisper_requests(req,cache,key,maximum_new=16)
        self.assertEqual(len(kept),16)
        rows={x.entry_index for x in kept}
        self.assertTrue(any(x<12 for x in rows))
        self.assertTrue(any(x>40 for x in rows))
        self.assertEqual(deferred,len(req)-16)
    def test_no_phantom_or_modified_requests(self):
        req=self.items(8)
        kept,_=budget_incremental_whisper_requests(req,{'local-whisper:old':'done'},key,maximum_new=3)
        self.assertTrue(all(k in req for k in kept))
        self.assertEqual(kept,budget_incremental_whisper_requests(
            req,{'local-whisper:old':'done'},key,maximum_new=3)[0])
    def test_no_new_work_does_not_defer(self):
        req=self.items(5)
        kept,removed=budget_incremental_whisper_requests(
            req,{key(x):'done' for x in req},key,maximum_new=1)
        self.assertEqual((kept,removed),(req,0))
    def test_invalid_budget(self):
        with self.assertRaises(ValueError):
            budget_incremental_whisper_requests(self.items(2),{},key,maximum_new=0)


if __name__=='__main__':unittest.main()
