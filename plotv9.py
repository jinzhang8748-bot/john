# ==============================================================================
# --- 1. 全局配置区 ---
# ==============================================================================
# 设置目标变量 (例如 'Solar', 'Geothermal', 'Wind')
TARGET_VARIABLE_NAME = 'Solar'
DEVICE = 'cpu'

COUNTRIES_TO_RUN = [
    'Italy', 'Germany', 'China', 'Japan', 'Australia',
    'India', 'United States', 'United Kingdom'
]

# --- 自动生成路径变量 (无需修改) ---
TARGET_COLUMN_NAME = f'{TARGET_VARIABLE_NAME}_Deployment'
INPUT_FILE_PATH = f'Processed_{TARGET_VARIABLE_NAME}_Data.csv'
OUTPUT_DIR_BASELINE = f"{TARGET_VARIABLE_NAME}Part_1_Baseline_Results"
OUTPUT_DIR_SHAP_TOP15 = f"{TARGET_VARIABLE_NAME}Part_2_SHAP_Top15_Results"
# ==============================================================================


# ==============================================================================
# --- 2. 导入库 ---
# ==============================================================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os, warnings, joblib, logging
from tqdm import tqdm
from scipy.optimize import curve_fit
import shap

from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error

warnings.filterwarnings('ignore')

# ==============================================================================
# --- 3. 工具函数与模型定义 ---
# ==============================================================================
def setup_logging(log_path):
    for handler in logging.root.handlers[:]: logging.root.removeHandler(handler)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s',
                        handlers=[logging.FileHandler(log_path, mode='w'), logging.StreamHandler()])

def logistic_model(t, L, k, t0): return L / (1 + np.exp(-k * (t - t0)))
def exponential_model(t, a, b): return a * np.exp(b * t)
TREND_MODELS = {'Logistic': logistic_model, 'Exponential': exponential_model}

def calculate_metrics(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    valid = ~np.isnan(y_pred); y_true, y_pred = y_true[valid], y_pred[valid]
    if len(y_true) == 0: return {'RMSE': np.inf, 'MAE': np.inf, 'MAPE': np.inf}
    rmse = np.sqrt(mean_squared_error(y_true, y_pred)); mae = mean_absolute_error(y_true, y_pred)
    non_zero = y_true != 0
    mape = np.mean(np.abs((y_true[non_zero] - y_pred[non_zero]) / y_true[non_zero])) * 100 if np.any(non_zero) else 0
    return {'RMSE': rmse, 'MAE': mae, 'MAPE': mape}

def plot_unified_comparison(country, results_df, out_dir, target, suffix="", split_year=None):
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(18, 10))
    ax.plot(results_df['Year'], results_df['Actual'], 'o-', color='black', label='Actual Data', zorder=10)
    models = [c for c in results_df.columns if c.find('_residual')==-1 and c not in ['Year', 'Country', 'Actual']]
    if models:
        cmap = plt.cm.turbo(np.linspace(0, 1, len(models)))
        for i, model in enumerate(models): ax.plot(results_df['Year'], results_df[model], '--', color=cmap[i], label=model)
    if split_year: ax.axvline(x=split_year - 0.5, color='brown', ls=':', label='Train/Test Split')
    ax.set_title(f'Model Comparison for {country} ({suffix})', fontsize=20, fontweight='bold')
    ax.set_xlabel('Year', fontsize=14); ax.set_ylabel(f'{target} (MW)', fontsize=14)
    ax.legend(loc='upper left', fontsize=10); plt.tight_layout()
    os.makedirs(out_dir, exist_ok=True)
    plt.savefig(os.path.join(out_dir, f'{country}_preds_{suffix}.png'), dpi=300); plt.close()

def ensure_dir(path):
    os.makedirs(path, exist_ok=True); return path

# ==============================================================================
# --- 4. 核心类定义 ---
# ==============================================================================
class ModelRunner:
    """用于基线模型训练的核心类"""
    def __init__(self, data_path, out_dir, target, suffix=""):
        self.data_path, self.out_dir, self.target, self.suffix = data_path, out_dir, target, suffix
        self.models_dir = ensure_dir(os.path.join(out_dir, 'saved_models'))
        self.plots_dir = ensure_dir(os.path.join(out_dir, 'comparison_plots'))
        self.preds_dir = ensure_dir(os.path.join(out_dir, 'predictions'))
        setup_logging(os.path.join(out_dir, 'run_log.log'))
        self.df = pd.read_csv(data_path)
        self.metrics = []

    def run(self, countries_to_run=None):
        logging.info("--- Starting Data Preprocessing ---")
        ratio = self.df.isnull().sum() / len(self.df)
        to_drop = ratio[ratio > 0.1].index
        self.df.drop(columns=[c for c in to_drop if c not in ['Country', 'Year', self.target]], inplace=True)
        self.df = self.df.groupby('Country', group_keys=False).apply(lambda g: g.sort_values('Year').ffill().bfill())
        logging.info("--- Data Preprocessing Complete ---")
        if countries_to_run: self.df = self.df[self.df['Country'].isin(countries_to_run)].copy()
        for country in tqdm(self.df['Country'].unique(), desc=f"Processing Countries ({self.suffix})"):
            self._process_country(country)
        pd.DataFrame(self.metrics).to_csv(os.path.join(self.out_dir, 'full_metrics_summary.csv'), index=False)
        logging.info("Saved metrics summary: full_metrics_summary.csv")

    def _process_country(self, country):
        c_data = self.df[self.df['Country'] == country].sort_values('Year')
        c_data = c_data[c_data[self.target] > 0].reset_index(drop=True)
        if len(c_data) < 10: logging.warning(f"Skipping {country}: data too short."); return
        split_idx = int(len(c_data) * 0.7)
        train, test = c_data.iloc[:split_idx], c_data.iloc[split_idx:]
        if len(test) == 0: logging.warning(f"Skipping {country}: no test data."); return
        logging.info(f"--- Processing {country} ({self.suffix}) ---")
        feats = [c for c in self.df.columns if c not in ['Country', 'Year', self.target]]
        if not feats: logging.warning(f"Skipping {country}: no feature columns."); return
        X_train, y_train, X_test, y_test = train[feats], train[self.target], test[feats], test[self.target]
        t_train, t_test = train['Year'] - train['Year'].min(), test['Year'] - train['Year'].min()
        preds_map = {}
        ml_models = {'LinearRegression': LinearRegression(), 'Lasso': Lasso(), 'Ridge': Ridge(),
                     'RandomForest': RandomForestRegressor(random_state=42),
                     'XGBoost': XGBRegressor(random_state=42, **({'device': DEVICE} if 'device' in XGBRegressor().get_params() else {})),
                     'CatBoost': CatBoostRegressor(random_state=42, verbose=0, allow_writing_files=False)}
        for name, model in ml_models.items():
            met, pred = self._train_eval_save_ml(model, name, country, X_train, y_train, X_test, y_test)
            if met: self.metrics.append(met); preds_map[name] = pred
        for name, func in TREND_MODELS.items():
            met, pred = self._train_eval_save_trend(func, name, country, t_train, y_train, t_test, y_test)
            if met: self.metrics.append(met); preds_map[name] = pred
        results = pd.DataFrame({'Year': c_data['Year'], 'Actual': c_data[self.target]})
        for name, pred in preds_map.items():
            p_series = pd.Series(np.nan, index=results.index); p_series.iloc[split_idx:] = pred
            results[name] = p_series; results[f'{name}_residual'] = results['Actual'] - results[name]
        out_csv = os.path.join(self.preds_dir, f"{country}_predictions.csv")
        results.to_csv(out_csv, index=False); logging.info(f"Saved predictions: {out_csv}")
        plot_unified_comparison(country, results, self.plots_dir, self.target, self.suffix, c_data.iloc[split_idx]['Year'])

    def _train_eval_save_ml(self, model, name, country, X_train, y_train, X_test, y_test):
        try:
            model.fit(X_train, y_train); preds = model.predict(X_test)
            metrics = calculate_metrics(y_test, preds); metrics.update({'Country': country, 'Model': name})
            logging.info(f"    - {country} | {name:<16} | Test MAPE: {metrics['MAPE']:.2f}%")
            joblib.dump(model, os.path.join(ensure_dir(os.path.join(self.models_dir, country)), f'{name}.joblib'))
            return metrics, preds
        except Exception as e:
            logging.error(f"Failed ML model {name} for {country}: {e}"); return None, None

    def _train_eval_save_trend(self, func, name, country, t_train, y_train, t_test, y_test):
        try:
            p0, bounds = ([max(y_train.max(), 1e-6), 0.5, len(t_train)/2], (0, [np.inf, 5.0, np.inf])) if name == 'Logistic' else ([y_train.iloc[0] if y_train.iloc[0] > 0 else max(y_train.median(), 1.0), 0.1], (0, [np.inf, 2.0]))
            params, _ = curve_fit(func, t_train, y_train, p0=p0, maxfev=10000, bounds=bounds)
            preds = func(t_test.to_numpy(), *params)
            metrics = calculate_metrics(y_test, preds); metrics.update({'Country': country, 'Model': name})
            logging.info(f"    - {country} | {name:<16} | Test MAPE: {metrics['MAPE']:.2f}%")
            joblib.dump({'func_name': name, 'params': params}, os.path.join(ensure_dir(os.path.join(self.models_dir, country)), f'{name}.joblib'))
            return metrics, preds
        except Exception as e:
            logging.error(f"Failed Trend model {name} for {country}: {e}"); return None, None

class ShapModelRunner(ModelRunner):
    """用于 SHAP Top 15 特征再训练的核心类 (已修正)"""
    def __init__(self, shap_df, original_data_path, out_dir, target, suffix=""):
        super().__init__(original_data_path, out_dir, target, suffix)
        self.shap_df = shap_df
        self.df = pd.read_csv(original_data_path)

    def _process_country(self, country):
        top_15_features = self.shap_df[self.shap_df['Country'] == country].head(15)['Feature'].tolist()
        if not top_15_features: logging.warning(f"No SHAP features for {country}. Skipping."); return
        logging.info(f"--- Processing {country} with Top 15 Features: {top_15_features} ---")
        
        c_data_full = self.df[self.df['Country'] == country]
        c_data = c_data_full[['Year', self.target] + top_15_features].sort_values('Year')
        c_data = c_data.ffill().bfill()
        c_data = c_data[c_data[self.target] > 0].reset_index(drop=True)
        if len(c_data) < 10: logging.warning(f"Skipping {country}: data too short."); return
        
        split_idx = int(len(c_data) * 0.7)
        train, test = c_data.iloc[:split_idx], c_data.iloc[split_idx:]
        if len(test) == 0: logging.warning(f"Skipping {country}: no test data."); return

        X_train, y_train = train[top_15_features], train[self.target]
        X_test,  y_test  = test[top_15_features],  test[self.target]
        t_train, t_test = train['Year'] - train['Year'].min(), test['Year'] - train['Year'].min()
        
        preds_map = {}
        ml_models = {
            'LinearRegression': LinearRegression(), 'Lasso': Lasso(), 'Ridge': Ridge(),
            'RandomForest': RandomForestRegressor(random_state=42),
            'XGBoost': XGBRegressor(random_state=42, **({'device': DEVICE} if 'device' in XGBRegressor().get_params() else {})),
            'CatBoost': CatBoostRegressor(random_state=42, verbose=0, allow_writing_files=False)
        }
        for name, model in ml_models.items():
            met, pred = self._train_eval_save_ml(model, name, country, X_train, y_train, X_test, y_test)
            if met: self.metrics.append(met); preds_map[name] = pred
        
        # --- 关键修正：将趋势模型添加回SHAP流程 ---
        for name, func in TREND_MODELS.items():
            met, pred = self._train_eval_save_trend(func, name, country, t_train, y_train, t_test, y_test)
            if met: self.metrics.append(met); preds_map[name] = pred

        results = pd.DataFrame({'Year': c_data['Year'], 'Actual': c_data[self.target]})
        for name, pred in preds_map.items():
            p_series = pd.Series(np.nan, index=results.index)
            p_series.iloc[split_idx:] = pred
            results[name] = p_series; results[f'{name}_residual'] = results['Actual'] - results[name]
        out_csv = os.path.join(self.preds_dir, f"{country}_predictions.csv")
        results.to_csv(out_csv, index=False); logging.info(f"Saved SHAP Top 15 predictions: {out_csv}")
        plot_unified_comparison(country, results, self.plots_dir, self.target, self.suffix, c_data.iloc[split_idx]['Year'])

def perform_shap_analysis(models_base_dir, data_path, countries, target_column):
    logging.info("--- Starting SHAP Analysis for XGBoost Models ---")
    df_full = pd.read_csv(data_path)
    ratio = df_full.isnull().sum() / len(df_full)
    to_drop = ratio[ratio > 0.1].index
    df_full.drop(columns=[c for c in to_drop if c not in ['Country', 'Year', target_column]], inplace=True)
    df_full = df_full.groupby('Country', group_keys=False).apply(lambda g: g.sort_values('Year').ffill().bfill())
    all_shap_values = []
    for country in tqdm(countries, desc="Running SHAP Analysis"):
        model_path = os.path.join(models_base_dir, 'saved_models', country, 'XGBoost.joblib')
        if not os.path.exists(model_path): logging.warning(f"XGBoost model for {country} not found."); continue
        c_data = df_full[df_full['Country'] == country].sort_values('Year')
        c_data = c_data[c_data[target_column] > 0].reset_index(drop=True)
        if len(c_data) < 10: continue
        features = [c for c in c_data.columns if c not in ['Country', 'Year', target_column]]
        X = c_data[features]
        model = joblib.load(model_path)
        try:
            explainer = shap.TreeExplainer(model); shap_values = explainer.shap_values(X)
            mean_abs_shap = np.abs(shap_values).mean(axis=0)
            shap_df = pd.DataFrame({'Feature': X.columns, 'Mean_ABS_SHAP': mean_abs_shap, 'Country': country}).sort_values('Mean_ABS_SHAP', ascending=False)
            all_shap_values.append(shap_df)
        except Exception as e:
            logging.error(f"SHAP analysis failed for {country}: {e}")
    if not all_shap_values:
        logging.error("SHAP analysis failed for all countries."); return None
    full_shap_df = pd.concat(all_shap_values, ignore_index=True)
    shap_csv_path = os.path.join(models_base_dir, "shap_feature_importance.csv")
    full_shap_df.to_csv(shap_csv_path, index=False)
    logging.info(f"✅ SHAP analysis complete. Importance scores saved to {shap_csv_path}")
    return shap_csv_path

# ==============================================================================
# --- 5. 主流程 ---
# ==============================================================================
if __name__ == "__main__":
    if not os.path.exists(INPUT_FILE_PATH):
        print(f"Error: Input data file not found at '{INPUT_FILE_PATH}'")
    else:
        print("===== STARTING: Part 1 - Baseline Model Training =====")
        baseline_runner = ModelRunner(INPUT_FILE_PATH, OUTPUT_DIR_BASELINE, TARGET_COLUMN_NAME, "All_Features")
        baseline_runner.run(countries_to_run=COUNTRIES_TO_RUN)
        print(f"\n--- Part 1 complete. Baseline results saved in '{OUTPUT_DIR_BASELINE}' ---")

        print("\n===== STARTING: Part 2 - SHAP Analysis =====")
        shap_csv_path = perform_shap_analysis(OUTPUT_DIR_BASELINE, INPUT_FILE_PATH, COUNTRIES_TO_RUN, TARGET_COLUMN_NAME)
        
        if shap_csv_path and os.path.exists(shap_csv_path):
            print("\n--- Part 2 complete. SHAP importance file saved. ---")
            
            print("\n===== STARTING: Part 3 - Retraining on SHAP Top 15 Features =====")
            shap_df = pd.read_csv(shap_csv_path)
            
            shap_runner = ShapModelRunner(
                shap_df=shap_df,
                original_data_path=INPUT_FILE_PATH,
                out_dir=OUTPUT_DIR_SHAP_TOP15,
                target=TARGET_COLUMN_NAME,
                suffix="SHAP_Top15"
            )
            shap_runner.run(countries_to_run=COUNTRIES_TO_RUN)
            print(f"\n--- Part 3 complete. SHAP Top 15 results in '{OUTPUT_DIR_SHAP_TOP15}' ---")
        else:
            print("\n--- Pipeline stopped. SHAP analysis failed, cannot proceed to Part 3. ---")
        
        print("\n✅✅✅ Entire pipeline finished!")