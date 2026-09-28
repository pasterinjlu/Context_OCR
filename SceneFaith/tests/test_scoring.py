import sys
import unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from common import classify
from score import score

ROW = {'id':'sample', 'observed_text':'dolphiin', 'canonical_text':'dolphin'}

class ScoringTests(unittest.TestCase):
    def test_categories(self):
        for raw, expected in [('DOLPHIIN','L'), ('dolphin','C'), ('fish','O'), ('text: "dolphiin"','L'), ('```text\ndolphiin\n```','L')]:
            self.assertEqual(classify(raw, ROW)[1], expected)
    def test_literal_canonical_collision(self):
        with self.assertRaises(ValueError):
            classify('abc', dict(ROW, observed_text='ABC', canonical_text='abc'))
    def test_errors_are_not_other(self):
        rows=[ROW, dict(ROW,id='second')]
        result=score(rows,[{'id':'sample','raw_response':'dolphin'},{'id':'second','error':'timeout'}])[0]
        self.assertEqual((result['n'],result['errors'],result['counts']['O'],result['rewriting_rate_percent']),(1,1,0,100))
    def test_duplicate_rejected(self):
        r={'id':'sample','raw_response':'dolphiin'}
        with self.assertRaises(ValueError):score([ROW],[r,r])
    def test_unknown_id_rejected(self):
        with self.assertRaises(ValueError):score([ROW],[{'id':'unknown','raw_response':'dolphiin'}])
    def test_empty_rejected(self):
        with self.assertRaises(ValueError):score([ROW],[{'id':'sample','raw_response':''}])

if __name__=='__main__':unittest.main()
