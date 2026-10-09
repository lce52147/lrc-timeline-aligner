import io
import unittest
from unittest.mock import Mock
import auto_lrc as a

class PersistentHelperPipeLifecycleTests(unittest.TestCase):
 def test_already_exited_worker_closes_all_pipes(self):
  p=Mock()
  p.poll.return_value=0
  p.stdin=io.StringIO();p.stdout=io.StringIO();p.stderr=io.StringIO()
  worker=object.__new__(a._PersistentHelperClient)
  worker.process=p
  worker.close()
  for f in (p.stdin,p.stdout,p.stderr):self.assertTrue(f.closed)
 def test_graceful_shutdown_closes_all_pipes(self):
  p=Mock()
  p.poll.return_value=None
  p.wait.return_value=0
  p.stdin=io.StringIO();p.stdout=io.StringIO();p.stderr=io.StringIO()
  worker=object.__new__(a._PersistentHelperClient)
  worker.process=p
  worker.close()
  p.wait.assert_called_once()
  for f in (p.stdin,p.stdout,p.stderr):self.assertTrue(f.closed)
if __name__=="__main__":unittest.main()
