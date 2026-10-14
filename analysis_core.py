"""Stateless sklearn engine for CRISP-DM regression web lab."""
import math
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression, RFE, SequentialFeatureSelector, RFECV
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LassoCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ALGORITHMS = {'all','f_test','mutual_info','forward','backward','rfe','rfecv','lasso'}

def safe_number(x):
    try:
        v = float(x)
        return round(v, 6) if math.isfinite(v) else None
    except (TypeError, ValueError):
        return None

def _to_float(x):
    if x is None:
        return None
    try:
        v = float(x)
        return v if math.isfinite(v) else None
    except (TypeError, ValueError):
        return None

def prepare(rows, target):
    if not isinstance(rows, list) or len(rows) < 16 or len(rows) > 2000:
        raise ValueError('資料筆數需介於 16 至 2,000 筆。')
    if not isinstance(rows[0], dict) or len(rows[0]) > 30:
        raise ValueError('CSV 最多支援 30 個欄位。')
    rows = [{str(k).strip(): v for k, v in row.items()} for row in rows]
    columns = list(rows[0].keys())
    if target not in columns:
        raise ValueError('找不到指定的目標欄位。')
    y_raw = [_to_float(row.get(target)) for row in rows]
    valid_mask = [v is not None for v in y_raw]
    orig_indices = np.array([i for i, ok in enumerate(valid_mask) if ok])
    rows_clean = [row for row, ok in zip(rows, valid_mask) if ok]
    y = np.array([v for v, ok in zip(y_raw, valid_mask) if ok], dtype=float)
    dropped = sum(1 for ok in valid_mask if not ok)
    if len(y) < 16: raise ValueError('有效目標數值少於 16 筆。')
    if len(set(y)) < 2: raise ValueError('目標欄位至少需要兩種不同數值。')
    feature_cols = [c for c in columns if c != target]
    if not feature_cols: raise ValueError('至少需要一個自變數欄位。')
    numeric = []; categoric = []
    for c in feature_cols:
        vals = [row.get(c) for row in rows_clean]
        non_empty = [v for v in vals if v is not None and str(v).strip() != '']
        if not non_empty: continue
        num_count = sum(1 for v in non_empty if _to_float(v) is not None)
        if num_count / len(non_empty) >= 0.95:
            numeric.append(c)
        else:
            unique_cats = {str(v) for v in non_empty}
            if len(unique_cats) > 16:
                raise ValueError(f'類別欄位「{c}」有超過 16 個類別，請先合併分類或刪除 ID 類欄位。')
            categoric.append(c)
    if not numeric and not categoric: raise ValueError('找不到可用特徵。')
    all_cols = numeric + categoric
    n_num = len(numeric)
    X = np.empty((len(rows_clean), len(all_cols)), dtype=object)
    for j in range(n_num):
        c = numeric[j]
        for i, row in enumerate(rows_clean):
            fv = _to_float(row.get(c))
            X[i, j] = fv if fv is not None else np.nan
    for j, c in enumerate(categoric):
        col_j = n_num + j
        for i, row in enumerate(rows_clean):
            v = row.get(c)
            s = str(v).strip() if v is not None else ''
            X[i, col_j] = s if s != '' else None
    return X, y, numeric, categoric, dropped, orig_indices

def _get_encoded_names(preprocessor, numeric, categoric):
    names = list(numeric)
    if categoric and 'cat' in preprocessor.named_transformers_:
        cat_pipe = preprocessor.named_transformers_['cat']
        ohe = cat_pipe.named_steps.get('onehot') if hasattr(cat_pipe, 'named_steps') else cat_pipe
        if ohe is not None and hasattr(ohe, 'categories_'):
            for col, cats in zip(categoric, ohe.categories_):
                for cat in cats[1:]:
                    names.append(f"{col}_{cat}")
    return names

def build(numeric, categoric, method, k, folds):
    n_num = len(numeric)
    n_cat = len(categoric)
    numeric_idx = list(range(n_num))
    categoric_idx = list(range(n_num, n_num + n_cat))
    trans = []
    if numeric_idx:
        trans.append(('num', Pipeline([('impute', SimpleImputer(strategy='median'))]), numeric_idx))
    if categoric_idx:
        trans.append(('cat', Pipeline([
            ('impute', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore', drop='first', sparse_output=False))
        ]), categoric_idx))
    pre = ColumnTransformer(trans, verbose_feature_names_out=False)
    if method == 'f_test': selector = SelectKBest(score_func=f_regression, k=k)
    elif method == 'mutual_info': selector = SelectKBest(score_func=lambda a, b: mutual_info_regression(a, b, random_state=42), k=k)
    elif method in ('forward', 'backward'):
        selector = SequentialFeatureSelector(LinearRegression(), n_features_to_select=k,
            direction=method, scoring='neg_mean_squared_error', cv=folds)
    elif method == 'rfe': selector = RFE(LinearRegression(), n_features_to_select=k)
    elif method == 'rfecv': selector = RFECV(LinearRegression(), cv=folds, scoring='neg_mean_squared_error', min_features_to_select=1)
    else: selector = 'passthrough'
    steps = [('preprocessor', pre)]
    if method == 'lasso':
        steps += [('scale', StandardScaler()), ('regressor', LassoCV(cv=folds, random_state=42, max_iter=20000))]
    else:
        steps += [('selector', selector), ('regressor', LinearRegression())]
    return Pipeline(steps)

def run_analysis(rows, target, method='all', feature_count=2):
    if method not in ALGORITHMS: raise ValueError('不支援的演算法。')
    X, y, numeric, categoric, dropped, orig_indices = prepare(rows, target)
    if len(X) > 400 and method in ('forward', 'backward', 'rfecv'):
        raise ValueError('逐步或遞迴選擇在雲端限制 400 筆；請改用 F-Test、RFE 或 Lasso。')
    X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
        X, y, orig_indices, test_size=.2, random_state=42)
    probe = build(numeric, categoric, 'all', 1, 3).named_steps['preprocessor']
    probe.fit(X_train)
    encoded = _get_encoded_names(probe, numeric, categoric)
    n = len(encoded)
    if n == 0: raise ValueError('沒有可用的編碼後特徵。')
    if n > 50: raise ValueError('編碼後特徵超過 50 個。')
    k = int(feature_count)
    if method in ('f_test', 'mutual_info', 'forward', 'backward', 'rfe') and not 1 <= k <= n:
        raise ValueError(f'特徵數量應在 1 至 {n} 之間。')
    folds = KFold(n_splits=3 if len(X_train) < 30 else 5, shuffle=True, random_state=42)
    if method in ('forward', 'backward') and k == n: method_effective = 'all'
    else: method_effective = method
    pipeline = build(numeric, categoric, method_effective, k, folds)
    pipeline.fit(X_train, y_train)
    pred = pipeline.predict(X_test)
    if method_effective in ('all', 'lasso'):
        chosen = list(encoded)
    else:
        chosen = [name for name, keep in zip(encoded, pipeline.named_steps['selector'].get_support()) if keep]
    coef = pipeline.named_steps['regressor'].coef_
    if method == 'lasso':
        chosen = [name for name, c in zip(encoded, coef) if abs(c) > 1e-9]
        coeff = [{'feature': name, 'coefficient': safe_number(c)} for name, c in zip(encoded, coef) if abs(c) > 1e-9]
    else: coeff = [{'feature': name, 'coefficient': safe_number(c)} for name, c in zip(chosen, coef)]
    mae = mean_absolute_error(y_test, pred); mse = mean_squared_error(y_test, pred)
    r2_val = r2_score(y_test, pred)
    n_test = len(y_test); p_feat = len(chosen)
    adj_r2 = safe_number(1 - (1 - r2_val) * (n_test - 1) / (n_test - p_feat - 1)) if n_test > p_feat + 1 else None
    f_stat_val = f_pvalue = None
    if p_feat > 0 and n_test > p_feat + 1 and r2_val < 1.0:
        raw_f = (r2_val / p_feat) / ((1 - r2_val) / (n_test - p_feat - 1))
        f_stat_val = safe_number(raw_f)
        try:
            from scipy.stats import f as _f_dist
            f_pvalue = safe_number(float(1 - _f_dist.cdf(raw_f, p_feat, n_test - p_feat - 1)))
        except Exception:
            pass
    cv_error = None; cv_scores = []
    try:
        cv_scores = cross_val_score(pipeline, X_train, y_train, cv=folds, scoring='neg_mean_absolute_error').tolist()
        cv_scores = [-v for v in cv_scores]
    except ValueError as e: cv_error = str(e)[:160]
    residual = y_test - pred
    sd = float(np.std(residual, ddof=1)) if len(residual) > 1 else 0
    points = []
    for ix, a, p, r in zip(idx_test, y_test, pred, residual):
        z = float((r - np.mean(residual)) / sd) if sd > 0 else 0.
        points.append({'row': int(ix) + 2, 'actual': safe_number(float(a)), 'predicted': safe_number(float(p)),
            'residual': safe_number(float(r)), 'z': safe_number(z), 'outlier': abs(z) > 2})
    corr = []
    for j, col in enumerate(numeric):
        col_vals = X[:, j].astype(float)
        valid = ~np.isnan(col_vals)
        if valid.sum() > 1:
            x_v = col_vals[valid]; y_v = y[valid]
            r_val = float(np.corrcoef(x_v, y_v)[0, 1])
            corr.append({'feature': col, 'correlation': safe_number(r_val)})
    model_info = {'intercept': safe_number(float(pipeline.named_steps['regressor'].intercept_)),
        'coefficients': coeff, 'standardized': method == 'lasso',
        'selected': chosen, 'total_encoded': n}
    return {'rows': len(X), 'dropped': dropped, 'target': target, 'numeric': numeric, 'categorical': categoric,
        'methods': method, 'feature_count': len(chosen),
        'metrics': {'mae': safe_number(mae), 'mse': safe_number(mse),
        'rmse': safe_number(np.sqrt(mse)), 'r2': safe_number(r2_val),
        'adj_r2': adj_r2, 'f_stat': f_stat_val, 'f_pvalue': f_pvalue,
        'cv_mae': safe_number(np.mean(cv_scores)) if cv_scores else None,
        'cv_std': safe_number(np.std(cv_scores)) if cv_scores else None},
        'cross_validation_error': cv_error, 'model': model_info, 'points': points,
        'correlations': corr, 'outlier_count': sum(p['outlier'] for p in points)}

def run_prediction(rows, target, method, feature_count, values):
    X, y, numeric, categoric, _, _ = prepare(rows, target)
    all_feature_cols = numeric + categoric
    missing = [col for col in all_feature_cols if col not in values]
    if missing: raise ValueError('缺少欄位：' + ', '.join(missing))
    X_train, _, y_train, _ = train_test_split(X, y, test_size=.2, random_state=42)
    probe = build(numeric, categoric, 'all', 1, 3).named_steps['preprocessor']
    probe.fit(X_train)
    n = len(_get_encoded_names(probe, numeric, categoric))
    k = int(feature_count)
    if method in ('f_test', 'mutual_info', 'forward', 'backward', 'rfe') and not 1 <= k <= n:
        raise ValueError('特徵數量超出範圍。')
    folds = KFold(n_splits=3 if len(X_train) < 30 else 5, shuffle=True, random_state=42)
    effective = 'all' if method in ('forward', 'backward') and k == n else method
    model = build(numeric, categoric, effective, k, folds)
    model.fit(X_train, y_train)
    n_num = len(numeric)
    X_pred = np.empty((1, n_num + len(categoric)), dtype=object)
    for j, c in enumerate(numeric):
        fv = _to_float(values.get(c))
        X_pred[0, j] = fv if fv is not None else np.nan
    for j, c in enumerate(categoric):
        v = values.get(c)
        s = str(v).strip() if v is not None else ''
        X_pred[0, n_num + j] = s if s != '' else None
    return {'prediction': safe_number(float(model.predict(X_pred)[0])),
        'training_rows': len(X_train),
        'message': '此預測使用 80% 訓練資料擬合的模型；測試集不參與模型擬合。'}

METHOD_LABELS = {
    'all':'全部特徵', 'f_test':'F-Test', 'mutual_info':'Mutual Information',
    'forward':'Forward Selection', 'backward':'Backward Selection',
    'rfe':'RFE', 'rfecv':'RFECV', 'lasso':'LassoCV'
}

def compare_methods(rows, target, feature_count=2):
    scores = []
    for method, label in METHOD_LABELS.items():
        try:
            result = run_analysis(rows, target, method, feature_count)
            scores.append({'method': method, 'label': label, 'metrics': result['metrics'],
                           'selected': result['model']['selected'],
                           'feature_count': result['feature_count'],
                           'error': result['cross_validation_error']})
        except (ValueError, TypeError) as exc:
            scores.append({'method': method, 'label': label, 'error': str(exc), 'metrics': {}, 'selected': []})
    valid = [r for r in scores if r['metrics'].get('cv_mae') is not None]
    if not valid: raise ValueError('沒有成功完成交叉驗證的模型可供比較。')
    ranked = sorted(valid, key=lambda r: r['metrics']['cv_mae'])
    return {'results': scores, 'winner_cv': ranked[0]['method'],
            'winner_holdout': max((r for r in valid if r['metrics'].get('r2') is not None),
                                  key=lambda r: r['metrics']['r2'])['method'],
            'selection_note': '以訓練集 5-Fold CV MAE 選擇模型；測試集只應用於最終評估，不能用其反覆挑選最佳模型。',
            'holdout_note': '測試集 R² 最高的方法僅供事後比較，不應據此調參或宣稱泛化最佳。'}


TREND_METHODS = ['f_test', 'mutual_info', 'forward', 'backward', 'rfe']

def feature_count_trends(rows, target, max_features=8):
    X, y, numeric, categoric, _, _ = prepare(rows, target)
    if len(X) > 400:
        raise ValueError('特徵數量趨勢分析最多支援 400 筆資料。')
    X_train, _, _, _ = train_test_split(X, y, test_size=.2, random_state=42)
    probe = build(numeric, categoric, 'all', 1, 3).named_steps['preprocessor']
    probe.fit(X_train)
    n = len(_get_encoded_names(probe, numeric, categoric))
    if n < 2: raise ValueError('至少需要 2 個編碼後特徵才能比較特徵數量。')
    if n > 50: raise ValueError('編碼後的特徵數量超過 50 個。')
    max_k = min(n, max_features)
    curves = []
    for method in TREND_METHODS:
        points = []
        for k in range(1, max_k + 1):
            try:
                result = run_analysis(rows, target, method, k)
                points.append({'k': k, 'rmse': result['metrics']['rmse'],
                               'r2': result['metrics']['r2'],
                               'cv_mae': result['metrics']['cv_mae'],
                               'selected': result['model']['selected']})
            except (ValueError, TypeError) as error:
                points.append({'k': k, 'rmse': None, 'r2': None,
                               'cv_mae': None, 'selected': [], 'error': str(error)[:150]})
        curves.append({'method': method, 'label': METHOD_LABELS[method], 'points': points})
    viable = [(p['cv_mae'], c['method'], p['k']) for c in curves for p in c['points']
              if p['cv_mae'] is not None]
    best = min(viable) if viable else None
    return {'curves': curves, 'max_features': max_k, 'total_features': n,
            'cv_best': {'method': best[1], 'k': best[2], 'cv_mae': best[0]} if best else None,
            'note': '以訓練集交叉驗證 CV MAE 評選方法與 k；測試集 RMSE/R² 趨勢僅作診斷，不應反覆使用測試集選模。'}
