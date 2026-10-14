"""50 Startups 多元線性迴歸分析 — CRISP-DM 六階段。

可作為 Flask、FastAPI 或其他 Web 後端的模型服務基底。

安裝：
    pip install pandas numpy scikit-learn matplotlib joblib

訓練：
    python startup_analysis.py train --csv 50_Startups.csv

預測：
    python startup_analysis.py predict --rd 120000 --administration 130000 \
        --marketing 250000 --state California

Web 後端可直接 import train_model、predict_profit、load_model。
模型檔案只能從可信來源載入。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import joblib
import matplotlib
matplotlib.use("Agg")  # 適合無桌面的伺服器環境
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

NUMERIC_FEATURES = ["R&D Spend", "Administration", "Marketing Spend"]
CATEGORICAL_FEATURES = ["State"]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES
TARGET = "Profit"
DEFAULT_MODEL = Path("outputs/startup_profit_model.pkl")


def build_pipeline() -> Pipeline:
    """CRISP-DM 3-4：資料前處理與線性迴歸封裝，避免資料洩漏。"""
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", SimpleImputer(strategy="median"), NUMERIC_FEATURES),
            (
                "cat",
                Pipeline([
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("encoder", OneHotEncoder(drop="first", handle_unknown="ignore")),
                ]),
                CATEGORICAL_FEATURES,
            ),
        ]
    )
    return Pipeline([("preprocessor", preprocessor), ("regressor", LinearRegression())])


def load_dataset(csv_path: str | Path) -> pd.DataFrame:
    """CRISP-DM 2：讀取並確認資料欄位及基本合法性。"""
    path = Path(csv_path)
    if not path.is_file():
        raise FileNotFoundError(f"找不到資料集：{path}，請先下載 50_Startups.csv")
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    missing = [name for name in FEATURES + [TARGET] if name not in df.columns]
    if missing:
        raise ValueError(f"CSV 缺少必要欄位：{missing}")
    for name in NUMERIC_FEATURES + [TARGET]:
        df[name] = pd.to_numeric(df[name], errors="coerce")
    df["State"] = df["State"].replace(r"^\s*$", np.nan, regex=True)
    # 目標值無法用回歸模型訓練；特徵缺失值由 Pipeline 處理。
    df = df.dropna(subset=[TARGET]).drop_duplicates().reset_index(drop=True)
    if len(df) < 15:
        raise ValueError("有效資料不足 15 筆，無法進行合理的訓練與評估。")
    if df["State"].notna().sum() == 0:
        raise ValueError("State 全部缺失，無法進行類別編碼。")
    return df


def _as_number(value: Any) -> float:
    return float(value)


def train_model(
    csv_path: str | Path = "50_Startups.csv",
    output_dir: str | Path = "outputs",
    test_size: float = 0.2,
    random_state: int = 42,
) -> dict[str, Any]:
    """執行 CRISP-DM 2-6；輸出 dict 方便 Web API 回傳 JSON。"""
    df = load_dataset(csv_path)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    # 2. 資料理解
    descriptive_stats = df[NUMERIC_FEATURES + [TARGET]].describe().round(3)
    correlation = df[NUMERIC_FEATURES + [TARGET]].corr().round(4)
    descriptive_stats.to_csv(output / "descriptive_statistics.csv", encoding="utf-8-sig")
    correlation.to_csv(output / "correlation_matrix.csv", encoding="utf-8-sig")

    # 3. 資料準備：先切分，Pipeline 只在訓練集 fit
    X = df[FEATURES]
    y = df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    # 4. 模型建立
    model = build_pipeline()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    feature_names = model.named_steps["preprocessor"].get_feature_names_out()
    coefficients = model.named_steps["regressor"].coef_
    coefficient_table = pd.DataFrame({
        "feature": feature_names,
        "coefficient": coefficients,
    })
    coefficient_table.to_csv(output / "coefficients.csv", index=False, encoding="utf-8-sig")

    # 5. 模型評估（獨立測試集）
    predictions = pd.DataFrame({
        "actual": y_test.to_numpy(),
        "predicted": y_pred,
    })
    predictions["error"] = predictions["actual"] - predictions["predicted"]
    predictions["absolute_error"] = predictions["error"].abs()
    predictions.to_csv(output / "test_predictions.csv", index=False, encoding="utf-8-sig")

    metrics = {
        "mae": _as_number(mean_absolute_error(y_test, y_pred)),
        "mse": _as_number(mean_squared_error(y_test, y_pred)),
        "rmse": _as_number(np.sqrt(mean_squared_error(y_test, y_pred))),
        "r2": _as_number(r2_score(y_test, y_pred)),
    }

    # 僅用訓練資料做交叉驗證，避免測試集參與模型選擇。
    kfold = KFold(n_splits=5, shuffle=True, random_state=random_state)
    cv_r2 = cross_val_score(build_pipeline(), X_train, y_train, cv=kfold, scoring="r2")
    cv_mae = -cross_val_score(
        build_pipeline(), X_train, y_train, cv=kfold, scoring="neg_mean_absolute_error"
    )
    simple_r2 = cross_val_score(
        LinearRegression(), X_train[["R&D Spend"]], y_train, cv=kfold, scoring="r2"
    )

    # 6. 部署：模型保存 + 分析報告 + 圖片
    model_path = output / "startup_profit_model.pkl"
    joblib.dump(model, model_path)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(y_test, y_pred, alpha=0.8)
    min_value = min(float(y_test.min()), float(y_pred.min()))
    max_value = max(float(y_test.max()), float(y_pred.max()))
    ax.plot([min_value, max_value], [min_value, max_value], "r--", label="Perfect prediction")
    ax.set(xlabel="Actual Profit", ylabel="Predicted Profit", title="Actual vs Predicted Profit")
    ax.legend()
    fig.tight_layout()
    chart_path = output / "actual_vs_predicted.png"
    fig.savefig(chart_path, dpi=160)
    plt.close(fig)

    result = {
        "project": "50 Startups — CRISP-DM Multiple Linear Regression",
        "business_understanding": {
            "goal": "利用研發、行政、行銷支出與所在州預測公司利潤",
            "target": TARGET,
            "features": FEATURES,
        },
        "data_understanding": {
            "rows": len(df),
            "columns": len(df.columns),
            "missing_values": {k: int(v) for k, v in df[FEATURES + [TARGET]].isna().sum().items()},
            "profit_correlations": {
                k: (None if pd.isna(v) else float(v))
                for k, v in correlation[TARGET].items()
            },
        },
        "data_preparation": {"train_rows": len(X_train), "test_rows": len(X_test)},
        "modeling": {
            "algorithm": "sklearn.linear_model.LinearRegression",
            "intercept": float(model.named_steps["regressor"].intercept_),
            "coefficients": {str(k): float(v) for k, v in zip(feature_names, coefficients)},
        },
        "evaluation": {
            "test_metrics": metrics,
            "cross_validation": {
                "folds": 5,
                "r2_each_fold": cv_r2.tolist(),
                "mean_r2": float(cv_r2.mean()),
                "std_r2": float(cv_r2.std()),
                "mean_mae": float(cv_mae.mean()),
                "rd_only_mean_r2": float(simple_r2.mean()),
            },
        },
        "deployment": {
            "model_path": str(model_path),
            "predictions_csv": str(output / "test_predictions.csv"),
            "chart_path": str(chart_path),
        },
    }
    report_path = output / "report.json"
    report_path.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    return result


def load_model(model_path: str | Path = DEFAULT_MODEL) -> Pipeline:
    """只載入可信任的本機模型檔。"""
    return joblib.load(model_path)


def predict_profit(
    rd_spend: float,
    administration: float,
    marketing_spend: float,
    state: str,
    model: Pipeline | None = None,
    model_path: str | Path = DEFAULT_MODEL,
) -> float:
    """供 Web API 呼叫的預測函式。"""
    if model is None:
        model = load_model(model_path)
    new_company = pd.DataFrame([{
        "R&D Spend": float(rd_spend),
        "Administration": float(administration),
        "Marketing Spend": float(marketing_spend),
        "State": state,
    }])
    return float(model.predict(new_company)[0])


def main() -> None:
    parser = argparse.ArgumentParser(description="50 Startups CRISP-DM 多元線性迴歸")
    subs = parser.add_subparsers(dest="command", required=True)

    train = subs.add_parser("train", help="訓練模型並輸出評估結果")
    train.add_argument("--csv", default="50_Startups.csv", help="CSV 資料集路徑")
    train.add_argument("--output", default="outputs", help="結果輸出資料夾")

    predict = subs.add_parser("predict", help="使用已訓練的模型預測利潤")
    predict.add_argument("--rd", type=float, required=True, help="研發支出")
    predict.add_argument("--administration", type=float, required=True, help="行政支出")
    predict.add_argument("--marketing", type=float, required=True, help="行銷支出")
    predict.add_argument("--state", required=True, help="公司所在州")
    predict.add_argument("--model", default=str(DEFAULT_MODEL), help="模型檔路徑")

    args = parser.parse_args()
    if args.command == "train":
        result = train_model(csv_path=args.csv, output_dir=args.output)
        print(json.dumps(result, ensure_ascii=False, indent=2))
    elif args.command == "predict":
        profit = predict_profit(
            args.rd, args.administration, args.marketing, args.state,
            model_path=args.model,
        )
        print(json.dumps({"predicted_profit": profit}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
