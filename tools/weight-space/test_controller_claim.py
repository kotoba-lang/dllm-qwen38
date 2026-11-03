"""Real competing processes must not share a controller/status destination."""
import json
import multiprocessing
from pathlib import Path
import tempfile
import unittest
from controller_claim import claim


def compete(folder,barrier,queue):
    barrier.wait(timeout=5)
    try:
        owner=claim(folder,'a'*64)
    except FileExistsError:
        queue.put('rejected');return
    (Path(folder)/'status.json').write_text(json.dumps({'owner_pid':owner['pid']}))
    queue.put('claimed')


class ControllerClaimTest(unittest.TestCase):
    def test_competing_processes_have_one_owner(self):
        with tempfile.TemporaryDirectory() as folder:
            ctx=multiprocessing.get_context('spawn');barrier=ctx.Barrier(2);queue=ctx.Queue()
            children=[ctx.Process(target=compete,args=(folder,barrier,queue)) for _ in range(2)]
            try:
                for child in children:child.start()
                for child in children:child.join(timeout=10)
                self.assertTrue(all(not c.is_alive() and c.exitcode==0 for c in children))
                self.assertEqual(sorted(queue.get(timeout=5) for _ in children),['claimed','rejected'])
                owner=json.loads((Path(folder)/'controller-owner.json').read_text())
                status=json.loads((Path(folder)/'status.json').read_text())
                self.assertEqual(owner['pid'],status['owner_pid'])
            finally:
                for child in children:
                    if child.is_alive():child.terminate();child.join()
                queue.close();queue.join_thread()

    def test_completed_destination_cannot_be_reused(self):
        with tempfile.TemporaryDirectory() as folder:
            first=claim(folder,'a'*64)
            marker=Path(folder)/'status.json';marker.write_text('completed')
            with self.assertRaises(FileExistsError):claim(folder,'b'*64)
            self.assertEqual(marker.read_text(),'completed')
            self.assertEqual(json.loads((Path(folder)/'controller-owner.json').read_text())['pid'],first['pid'])


if __name__=='__main__':unittest.main()
