from analysis_core import run_analysis,run_prediction

def fake_rows():
    return [{'spend':i*13+5,'staff':i%5,'state':['A','B','C'][i%3],
             'profit':i*25+6+(i%4)*4} for i in range(35)]

def test_analysis():
    for method in ('all','f_test','mutual_info','forward','backward','rfe','rfecv','lasso'):
        v=run_analysis(fake_rows(),'profit',method,2)
        assert len(v['points'])==7
        assert 'r2' in v['metrics']

def test_prediction():
    v=run_prediction(fake_rows(),'profit','all',2,{'spend':100,'staff':2,'state':'B'})
    assert isinstance(v['prediction'],float)

def test_compare():
    from analysis_core import compare_methods
    result=compare_methods(fake_rows(),'profit',2)
    assert len(result['results'])==8
    assert result['winner_cv'] in ['all','f_test','mutual_info','forward','backward','rfe','rfecv','lasso']

def test_feature_count_trends():
    import csv
    from analysis_core import feature_count_trends
    from pathlib import Path
    csv_path = Path(__file__).parents[1] / 'public' / '50_Startups.csv'
    with open(csv_path, newline='', encoding='utf-8') as f:
        rows = list(csv.DictReader(f))
    result = feature_count_trends(rows, 'Profit')
    assert result['max_features'] == 5
    assert len(result['curves']) == 5
    assert all(len(c['points']) == 5 for c in result['curves'])
    assert result['cv_best']['k'] in range(1, 6)
