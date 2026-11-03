"""Synthetic leakage controls, no benchmark question/reference data."""
import json
from pathlib import Path
import tempfile
import unittest
from split_audit import compare,audit


class SplitAuditTest(unittest.TestCase):
    def test_template_and_unrelated_controls(self):
        a='Alice owns 12 red pencils and buys 3 more. How many now?'
        b='ALICE owns 92 red pencils and buys 7 more. How many now?'
        flags,similarity=compare(a,b)
        self.assertIn('number_masked_exact',flags)
        self.assertNotIn('normalized_exact',flags)
        self.assertEqual(similarity,1)
        self.assertIn('normalized_exact',compare('Ａ + B','a  + b')[0])
        self.assertEqual(compare(a,'Compute the area of a circle of radius five.')[0],[])

    def test_report_omits_text_and_rejects_wrong_role(self):
        with tempfile.TemporaryDirectory() as folder:
            paths=[Path(folder)/name for name in ['dev.json','validation.json']]
            for path,split in zip(paths,['dev','validation']):
                path.write_text(json.dumps(dict(split=split,cases=[dict(id=split,question='Synthetic PRIVATE_SENTINEL + 12',answer='secret') ])))
            result=audit([paths[0]],[paths[1]])
            self.assertEqual(result['cross_split_pair_count'],1)
            self.assertEqual(result['flagged_pair_count'],1)
            self.assertNotIn('PRIVATE_SENTINEL',json.dumps(result))
            self.assertNotIn('secret',json.dumps(result))
            with self.assertRaises(ValueError):audit([paths[1]],[paths[0]])


if __name__=='__main__':unittest.main()
