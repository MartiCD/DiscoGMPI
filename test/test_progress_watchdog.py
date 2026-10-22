"""Exercise real child processes: timeout, live heartbeat, and nonzero exit."""
import importlib.util
from pathlib import Path
import sys
import tempfile
import time
import unittest

module_path=Path(__file__).resolve().parents[1]/'examples/run_with_progress_watchdog.py'
spec=importlib.util.spec_from_file_location('watchdog',module_path)
watchdog=importlib.util.module_from_spec(spec)
spec.loader.exec_module(watchdog)


class ProgressWatchdog(unittest.TestCase):
    def run_child(self,root,code,stall='.2',startup='2'):
        return watchdog.main(['--heartbeat',str(root/'heartbeat'),
            '--stall-seconds',stall,'--startup-seconds',startup,
            '--poll-seconds','.02','--terminate-grace','.1','--',sys.executable,'-c',code])

    def test_stall_and_startup_timeout(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            code=f'from pathlib import Path; import time; Path({str(root/"heartbeat")!r}).touch(); time.sleep(30)'
            started=time.monotonic()
            self.assertEqual(self.run_child(root,code),124)
            self.assertLess(time.monotonic()-started,5)
        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(self.run_child(Path(tmp),'import time; time.sleep(30)',startup='.2'),124)

    def test_progress_and_exit_status(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            code=('from pathlib import Path; import time\n'
                  f'p=Path({str(root/"heartbeat")!r})\n'
                  'for _ in range(8):\n p.touch(); time.sleep(.06)\n')
            self.assertEqual(self.run_child(root,code),0)
        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(self.run_child(Path(tmp),'raise SystemExit(17)'),17)


if __name__=='__main__':
    unittest.main()
