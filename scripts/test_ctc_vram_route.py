import unittest
from types import SimpleNamespace
try:
    import torch
    import ctc_align as c
except ImportError:
    torch = None
    c = None

class FakeExtractor(torch.nn.Module if torch is not None else object):
 def __init__(self):
  super().__init__();self.calls=0
 def forward(self,waveform,length):
  self.calls+=1
  return waveform+1,length

@unittest.skipUnless(torch is not None and c is not None, "requires torchaudio CTC runtime")
class CTCVramRouteTests(unittest.TestCase):
 def test_full_length_forward_and_idempotent_patch(self):
  extractor=FakeExtractor()
  encoder=torch.nn.Linear(1,1)
  model=SimpleNamespace(model=SimpleNamespace(feature_extractor=extractor,encoder=encoder))
  c.configure_cuda_cpu_feature_extractor(model)
  wave=torch.arange(0,400,dtype=torch.float32)[None,:]
  length=torch.tensor([400])
  out,n=c.model_forward_placeholder(model,wave,length) if hasattr(c,'model_forward_placeholder') else extractor(wave,length)
  self.assertTrue(torch.equal(out,wave+1))
  self.assertTrue(torch.equal(n,length))
  original=extractor._lrc_ctc_original_forward
  c.configure_cuda_cpu_feature_extractor(model)
  self.assertIs(extractor._lrc_ctc_original_forward,original)
  out2,n2=extractor(wave,length)
  self.assertTrue(torch.equal(out2,out))
  self.assertEqual(extractor.calls,2)
 def test_precision_stays_fp32(self):
  obj=FakeExtractor()
  encoder=torch.nn.Linear(1,1)
  c.configure_cuda_cpu_feature_extractor(SimpleNamespace(model=SimpleNamespace(feature_extractor=obj,encoder=encoder)))
  t=torch.ones((1,100),dtype=torch.float32)
  out,_=obj(t,torch.tensor([100]))
  self.assertEqual(out.dtype,torch.float32)
if __name__=='__main__':unittest.main()
