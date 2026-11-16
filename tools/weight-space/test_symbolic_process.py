"""Exercise real child success, timeout and invalid-reference separation."""
import time
import unittest
from symbolic_process import score_isolated


class IsolatedScoreTest(unittest.TestCase):
    def test_child_scoring(self):
        result=score_isolated(r'\frac{1}{2}',r'\boxed{0.5}')
        self.assertEqual(result['status'],'scored')
        self.assertIs(result['correct'],True)
        self.assertNotIn('0.5',str(result))

    def test_deadline_is_not_wrong_answer(self):
        start=time.monotonic()
        result=score_isolated('1',r'\boxed{1}',timeout_seconds=0.000001)
        self.assertEqual(result['reason'],'process_timeout')
        self.assertIsNone(result['correct'])
        self.assertLess(time.monotonic()-start,5)

    def test_invalid_inputs(self):
        for timeout in (0,61,float('nan'),float('inf')):
            with self.assertRaises(ValueError):
                score_isolated('1',r'\boxed{1}',timeout_seconds=timeout)
        with self.assertRaises(ValueError):
            score_isolated('',r'\boxed{1}')


if __name__=='__main__':unittest.main()
