# ==============================================================================
# --- 1. 全局配置区 ---
# ==============================================================================
# 设置目标变量 (确保与已完成的基线运行一致)
TARGET_VARIABLE_NAME = 'Geothermal'
DEVICE = 'cpu'

COUNTRIES_TO_RUN = [
    'Italy', 'Germany', 'China', 'Japan', 'Australia',
    'India', 'United States', 'United Kingdom'
]

# --- 自动生成路径变量 (无需修改) ---
TARGET_COLUMN_NAME = f'{TARGET_VARIABLE_NAME}_Deployment'
INPUT_FILE_PATH = f'Processed_{TARGET_VARIABLE_NAME}_Data.csv'
OUTPUT_DIR_BASELINE = f"{TARGET_VARIABLE_NAME}Part_1_Baseline_Results"
OUTPUT_DIR_BAYES_OPT = f"{TARGET_VARIABLE_NAME}Part_4_Bayes_Opt_Results"
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
from functools import partial

try:
    from skopt import gp_minimize
    from skopt.space import Real, Integer
    SKOPT_AVAILABLE = True
except ImportError:
    SKOPT_AVAILABLE = False
    print("Fatal Error: scikit-optimize is not installed and is required for this script.")
    print("Please run 'pip install scikit-optimize'")
    exit()

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
# --- 4. 核心类定义 (保留了 ModelRunner 作为父类) ---
# ==============================================================================
class ModelRunner:
    """父类，提供基础功能和预处理逻辑。"""
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

    def _train_eval_save_ml(self, model, name, country, X_train, y_train, X_test, y_test):
        try:
            model.fit(X_train, y_train); preds = model.predict(X_test)
            metrics = calculate_metrics(y_test, preds); metrics.update({'Country': country, 'Model': name})
            logging.info(f"    - {country} | {name:<16} | Test MAPE: {metrics['MAPE']:.2f}%")
            joblib.dump(model, os.path.join(ensure_dir(os.path.join(self.models_dir, country)), f'{name}.joblib'))
            return metrics, preds
        except Exception as e:
            logging.error(f"Failed ML model {name} for {country}: {e}"); return None, None

    def _process_country(self, country):
        # This method will be overridden by the child class
        raise NotImplementedError

class BayesianOptimizationRunner(ModelRunner):
    """对每个国家动态选择其最佳基线模型进行贝叶斯优化。"""
    def __init__(self, baseline_metrics_path, data_path, out_dir, target, suffix="BayesOpt"):
        super().__init__(data_path, out_dir, target, suffix)
        if not os.path.exists(baseline_metrics_path):
            raise FileNotFoundError(f"Baseline metrics file not found at: {baseline_metrics_path}")
        self.baseline_metrics_df = pd.read_csv(baseline_metrics_path)
        self.search_spaces = {
            'XGBoost': [Integer(100, 1000, name='n_estimators'), Real(0.01, 0.3, name='learning_rate'),
                        Integer(3, 10, name='max_depth'), Real(0.5, 1.0, name='subsample'), Real(0.5, 1.0, name='colsample_bytree')],
            'RandomForest': [Integer(100, 1000, name='n_estimators'), Integer(5, 50, name='max_depth'),
                             Integer(2, 20, name='min_samples_split'), Integer(1, 10, name='min_samples_leaf')],
            'CatBoost': [Integer(100, 1000, name='iterations'), Real(0.01, 0.3, name='learning_rate'),
                         Integer(3, 10, name='depth'), Real(0.5, 1.0, name='subsample')]
        }

    def _objective(self, params, param_names, model_name, X_train, y_train, X_test, y_test):
        params_dict = dict(zip(param_names, params))
        if model_name == 'XGBoost': model = XGBRegressor(random_state=42, **params_dict, **({'device': DEVICE} if 'device' in XGBRegressor().get_params() else {}))
        elif model_name == 'RandomForest': model = RandomForestRegressor(random_state=42, **params_dict)
        elif model_name == 'CatBoost': model = CatBoostRegressor(random_state=42, verbose=0, allow_writing_files=False, **params_dict)
        model.fit(X_train, y_train)
        return calculate_metrics(y_test, model.predict(X_test))['MAPE']

    def _process_country(self, country):
        country_metrics = self.baseline_metrics_df[self.baseline_metrics_df['Country'] == country]
        tunable_models = ['XGBoost', 'RandomForest', 'CatBoost']
        country_metrics = country_metrics[country_metrics['Model'].isin(tunable_models)]
        if country_metrics.empty:
            logging.warning(f"No tunable baseline models found for {country}. Skipping optimization."); return
        best_model_name = country_metrics.sort_values('MAPE').iloc[0]['Model']
        logging.info(f"--- Best model for {country} is {best_model_name}. Starting optimization... ---")
        
        c_data = self.df[self.df['Country'] == country].sort_values('Year')
        c_data = c_data[c_data[self.target] > 0].reset_index(drop=True)
        if len(c_data) < 10: return
        split_idx = int(len(c_data) * 0.7)
        train, test = c_data.iloc[:split_idx], c_data.iloc[split_idx:]
        if len(test) == 0: return
        
        feats = [c for c in self.df.columns if c not in ['Country', 'Year', self.target]]
        X_train, y_train, X_test, y_test = train[feats], train[self.target], test[feats], test[self.target]
        X_full, y_full = c_data[feats], c_data[self.target]
        
        search_space = self.search_spaces.get(best_model_name)
        if not search_space: logging.error(f"No search space for {best_model_name}."); return
        param_names = [p.name for p in search_space]
        
        objective_with_data = partial(self._objective, param_names=param_names, model_name=best_model_name, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test)
        
        result = gp_minimize(func=objective_with_data, dimensions=search_space, n_calls=25, random_state=42)
        best_params = dict(zip(param_names, result.x))
        logging.info(f"Best params for {country} ({best_model_name}): {best_params}")
        
        if best_model_name == 'XGBoost': final_model = XGBRegressor(random_state=42, **best_params, **({'device': DEVICE} if 'device' in XGBRegressor().get_params() else {}))
        elif best_model_name == 'RandomForest': final_model = RandomForestRegressor(random_state=42, **best_params)
        elif best_model_name == 'CatBoost': final_model = CatBoostRegressor(random_state=42, verbose=0, allow_writing_files=False, **best_params)
        
        met, pred = self._train_eval_save_ml(final_model, f"{best_model_name}_Optimized", country, X_train, y_train, X_test, y_test)
        if met: self.metrics.append(met)
        
        results = pd.DataFrame({'Year': c_data['Year'], 'Actual': y_full})
        p_series = pd.Series(np.nan, index=results.index); p_series.iloc[split_idx:] = pred
        results[f"{best_model_name}_Optimized"] = p_series
        results[f"{best_model_name}_Optimized_residual"] = results['Actual'] - results[f"{best_model_name}_Optimized"]
        out_csv = os.path.join(self.preds_dir, f"{country}_predictions.csv")
        results.to_csv(out_csv, index=False); logging.info(f"Saved BayesOpt predictions: {out_csv}")
        plot_unified_comparison(country, results, self.plots_dir, self.target, self.suffix, c_data.iloc[split_idx]['Year'])

# ==============================================================================
# --- 5. 主执行流程 ---
# ==============================================================================
if __name__ == "__main__":
    if not os.path.exists(INPUT_FILE_PATH):
        print(f"Error: Input data file not found at '{INPUT_FILE_PATH}'")
    elif not SKOPT_AVAILABLE:
        print("\nPipeline stopped. Please install scikit-optimize by running 'pip install scikit-optimize'")
    else:
        print("\n===== STARTING: Part 4 - Bayesian Optimization of Best Model for Each Country =====")
        baseline_metrics_path = os.path.join(OUTPUT_DIR_BASELINE, 'full_metrics_summary.csv')
        
        if os.path.exists(baseline_metrics_path):
            bayes_runner = BayesianOptimizationRunner(
                baseline_metrics_path=baseline_metrics_path,
                data_path=INPUT_FILE_PATH,
                out_dir=OUTPUT_DIR_BAYES_OPT,
                target=TARGET_COLUMN_NAME
            )
            bayes_runner.run(countries_to_run=COUNTRIES_TO_RUN)
            print(f"\n--- Part 4 complete. Bayesian Optimization results in '{OUTPUT_DIR_BAYES_OPT}' ---")
        else:
            print(f"\n--- Baseline metrics not found at '{baseline_metrics_path}'. Cannot run Part 4. ---")
            print("Please run the full pipeline first to generate baseline results.")
        
        print("\n✅✅✅ Bayesian Optimization pipeline finished!")