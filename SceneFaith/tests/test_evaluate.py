import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
import evaluate

ROOT=Path(__file__).resolve().parents[1]

class Response:
    status_code=200
    def json(self):
        return {'model':'mock','choices':[{'finish_reason':'stop','message':{'content':'dolphiin'}}]}

class Session:
    def __enter__(self):return self
    def __exit__(self,*args):pass
    def post(self,*args,**kwargs):return Response()

class EvaluateTests(unittest.TestCase):
    def run_mock(self, output):
        argv=['evaluate.py','--annotations',str(ROOT/'examples/annotations.json'),'--model','mock','--limit','1','--output',str(output)]
        with patch.object(sys,'argv',argv), patch.dict(os.environ,{'CONTEXT_OCR_API_KEY':'test-key','CONTEXT_OCR_BASE_URL':'https://mock.invalid/v1'}), patch('requests.Session',Session):
            evaluate.main()
    def test_success_and_no_overwrite(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'predictions.jsonl';self.run_mock(p)
            self.assertEqual(json.loads(p.read_text())['category'],'L')
            with self.assertRaises(FileExistsError):self.run_mock(p)
    def test_truncated_response_is_error(self):
        with tempfile.TemporaryDirectory() as d, patch.object(Response,'json',return_value={'choices':[{'finish_reason':'length','message':{'content':'dolphiin'}}]}):
            p=Path(d)/'predictions.jsonl';self.run_mock(p)
            self.assertTrue(json.loads(p.read_text())['error'])
    def test_unauthorized_stops(self):
        with tempfile.TemporaryDirectory() as d, patch.object(Response,'status_code',401):
            p=Path(d)/'predictions.jsonl'
            with self.assertRaises(SystemExit):self.run_mock(p)
            self.assertEqual(json.loads(p.read_text())['error'],'HTTP 401')

if __name__=='__main__':unittest.main()
