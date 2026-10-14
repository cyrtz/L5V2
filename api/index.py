from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Any
from analysis_core import run_analysis, run_prediction

app=FastAPI(title='Regression Lab API',version='1.0.0')
class AnalyzeRequest(BaseModel):
    rows: list[dict[str,Any]] = Field(min_length=16,max_length=2000)
    target: str
    method: str = 'all'
    feature_count: int = 2
class PredictRequest(AnalyzeRequest):
    values: dict[str,Any]

@app.get('/api/health')
def health(): return {'status':'ok'}
@app.post('/api/analyze')
def analyze(request:AnalyzeRequest):
    try: return run_analysis(request.rows,request.target,request.method,request.feature_count)
    except (ValueError,TypeError,KeyError) as e: raise HTTPException(status_code=422,detail=str(e))
@app.post('/api/predict')
def predict(request:PredictRequest):
    try: return run_prediction(request.rows,request.target,request.method,request.feature_count,request.values)
    except (ValueError,TypeError,KeyError) as e: raise HTTPException(status_code=422,detail=str(e))

from analysis_core import compare_methods

@app.post('/api/compare')
def compare(request:AnalyzeRequest):
    try: return compare_methods(request.rows,request.target,request.feature_count)
    except (ValueError,TypeError,KeyError) as e: raise HTTPException(status_code=422,detail=str(e))

from analysis_core import feature_count_trends

@app.post('/api/feature-trends')
def trends(request: AnalyzeRequest):
    try: return feature_count_trends(request.rows, request.target)
    except (ValueError, TypeError, KeyError) as e: raise HTTPException(status_code=422, detail=str(e))
