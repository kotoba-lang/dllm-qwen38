"""Integrity boundaries for completed-run collection; fixtures are not model evidence."""
import json
import pathlib
import tempfile
import unittest

import bench
import collect


class CollectTest(unittest.TestCase):
    def test_completed_pairs_and_corrupted_or_live_runs(self):
        with tempfile.TemporaryDirectory() as td:
            root = pathlib.Path(td)
            datafile=root/'data.json';datafile.write_text(json.dumps({'cases':[{'id':'a'},{'id':'b'}]}))
            for name in ['original','candidate']:
                d=root/name;d.mkdir()
                identity=dict(case_ids=['a','b'],data_sha256=bench.digest(datafile),split='dev',settings={},thinking='off',backend_commit='fixture',server_context=2048,bench_sha256='frozen-scorer',model_sha256=name)
                rows=[dict(id='a',correct=True,predicted='2',expected='2',wall_seconds=1.0,response=dict(content='FINAL: 2',tokens_predicted=5)),dict(id='b',correct=name=='candidate',predicted='3' if name=='candidate' else '4',expected='3',wall_seconds=2.0,response=dict(content='FINAL: 3' if name=='candidate' else 'FINAL: 4',tokens_predicted=7))]
                (d/'manifest.json').write_text(json.dumps({'identity':identity}))
                (d/'results.jsonl').write_text('\n'.join(json.dumps(x) for x in rows)+'\n')
                (d/'summary.json').write_text(json.dumps(bench.summary(rows)))
                (d/'launch.json').write_text(json.dumps({'config':dict(engine_sha256='same-engine',engine_args=[],backend_commit='fixture',model_sha256=name,data=str(datafile))}))
                (d/'process-status.json').write_text(json.dumps(dict(state='completed',engine_stopped=True,engine_exit_code=0)))
            plan=root/'plan.json';plan.write_text(json.dumps(dict(conditions=[dict(name=n,run=n) for n in ['original','candidate']],paired=[['original','candidate']],qualification='fixture')))
            result=collect.collect(plan)
            self.assertEqual(result['paired'][0]['result']['wins'],1)
            self.assertEqual(result['runs']['candidate']['summary']['correct'],2)
            for filename,mutate,reason in [
                ('process-status.json',lambda x:x.update(engine_stopped=False),'not completed and stopped'),
                ('summary.json',lambda x:x.update(correct=99),'summary mismatch'),
                ('manifest.json',lambda x:x['identity'].update(bench_sha256='different-scorer'),'bench_sha256 differs'),
                ('launch.json',lambda x:x['config'].update(model_sha256='wrong-model'),'provenance mismatch'),
                ('launch.json',lambda x:x['config'].update(engine_args=['--changed-profile']),'engine_args differs')]:
                p=root/'candidate'/filename;original=p.read_text();data=json.loads(original);mutate(data);p.write_text(json.dumps(data))
                with self.subTest(reason=reason),self.assertRaisesRegex(ValueError,reason):collect.collect(plan)
                p.write_text(original)
            original_data=datafile.read_text();datafile.write_text('changed dataset')
            with self.assertRaisesRegex(ValueError,'dataset checksum'):collect.collect(plan)
            datafile.write_text(original_data)
            rows=root/'candidate/results.jsonl';rows.write_text(rows.read_text().splitlines()[0]+'\n')
            with self.assertRaisesRegex(ValueError,'incomplete or reordered'):collect.collect(plan)


if __name__=='__main__':unittest.main()
