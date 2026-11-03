"""Synthetic expressions only; no MATH-500 or other final examples are loaded."""
import unittest
from symbolic_score import score,final_box


class SymbolicScoreTest(unittest.TestCase):
    def test_equivalence_and_disagreement(self):
        cases=[
            (r'\frac{1}{2}',r'\boxed{0.5}',True),
            (r'3\sqrt{2}',r'\boxed{\sqrt{18}}',True),
            (r'\{1,2\}',r'\boxed{\{2,1\}}',True),
            ('x+1',r'\boxed{1+x}',True),
            ('1',r'\boxed{2}',False),
            ('x+1',r'\boxed{y+1}',False),
            ('1',r'\boxed{0.9999999}',False),
            # Pinned parser/comparator rejects this near-integer value. The
            # rounding argument alone does not define a uniform tolerance.
            ('1',r'\boxed{0.9999999999999}',False),
        ]
        for gold,prediction,expected in cases:
            with self.subTest(gold=gold,prediction=prediction):
                result=score(gold,prediction)
                self.assertEqual(result['status'],'scored')
                self.assertIs(result['correct'],expected)

    def test_box_and_thinking_boundaries(self):
        self.assertEqual(final_box(r'\boxed{1} then \boxed{\frac{1}{\boxed{2}}}'),r'\boxed{\frac{1}{\boxed{2}}}')
        self.assertTrue(score('2',r'\boxed{1} then \boxed{2}')['correct'])
        self.assertTrue(score('1',r'<think>\boxed{9}</think>\boxed{1}')['correct'])
        self.assertTrue(score('1',r'\boxed{9}</think>\boxed{1}',thinking_open=True)['correct'])
        for content,opened,status in [
            (r'\boxed{1}',True,'unfinished_thinking'),
            (r'<think>\boxed{1}',False,'unfinished_thinking'),
            (r'\boxed{1} then \boxed{',False,'unfinished_box'),
            ('The result is1',False,'missing_box'),
            (r'</think>\boxed{1}',False,'unmatched_thinking_close')]:
            result=score('1',content,thinking_open=opened)
            self.assertEqual(result['status'],status);self.assertFalse(result['correct'])
        with self.assertRaises(ValueError):score('',r'\boxed{1}')


if __name__=='__main__':unittest.main()
