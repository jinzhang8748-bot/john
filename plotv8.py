# ==============================================================================
# --- 1. 全局配置区 ---
# ==============================================================================
TARGET_VARIABLE_NAME = 'Wind'
EXECUTION_MODE = 'PLOT_ONLY'   # FULL_PIPELINE / PLOT_ONLY
DEVICE = 'cpu'  # xgboost>=2.0 支持 device='cpu'/'cuda'；老版本可忽略

COUNTRIES_TO_RUN = [
    'Italy', 'Germany', 'China', 'Japan', 'Australia',
    'India', 'United States', 'United Kingdom'
]

TARGET_COLUMN_NAME = f'{TARGET_VARIABLE_NAME}_Deployment'
INPUT_FILE_PATH = f'Processed_{TARGET_VARIABLE_NAME}_Data.csv'
OUTPUT_DIR_BASELINE = f"{TARGET_VARIABLE_NAME}Part_1_Baseline_Results"

# ==============================================================================
# --- 2. 导入库 ---
# ==============================================================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os, warnings, joblib, logging, glob
from tqdm import tqdm
from scipy.optimize import curve_fit

from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error

warnings.filterwarnings('ignore')

# ==============================================================================
# --- 3. 工具函数 ---
# ==============================================================================
def setup_logging(log_path):
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s',
                        handlers=[logging.FileHandler(log_path, mode='w'), logging.StreamHandler()])

def logistic_model(t, L, k, t0):
    return L / (1 + np.exp(-k * (t - t0)))

def exponential_model(t, a, b):
    return a * np.exp(b * t)

TREND_MODELS = {'Logistic': logistic_model, 'Exponential': exponential_model}

def calculate_metrics(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    valid = ~np.isnan(y_pred)
    y_true, y_pred = y_true[valid], y_pred[valid]
    if len(y_true) == 0:
        return {'RMSE': np.inf, 'MAE': np.inf, 'MAPE': np.inf}
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    non_zero = y_true != 0
    mape = np.mean(np.abs((y_true[non_zero] - y_pred[non_zero]) / y_true[non_zero])) * 100 if np.any(non_zero) else 0
    return {'RMSE': rmse, 'MAE': mae, 'MAPE': mape}

def plot_unified_comparison(country, results_df, out_dir, target, suffix="", split_year=None):
    """绘制预测对比图（不显示 MAPE）。"""
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(18, 10))
    ax.plot(results_df['Year'], results_df['Actual'], 'o-', color='black', label='Actual Data', zorder=10)

    models = [c for c in results_df.columns if c not in ['Year', 'Country', 'Actual']]
    if len(models) > 0:
        cmap = plt.cm.turbo(np.linspace(0, 1, len(models)))
        for i, model in enumerate(models):
            ax.plot(results_df['Year'], results_df[model], '--', color=cmap[i], label=model)

    if split_year:
        ax.axvline(x=split_year - 0.5, color='brown', ls=':', label='Train/Test Split')

    ax.set_title(f'Model Comparison for {country} ({suffix})', fontsize=20, fontweight='bold')
    ax.set_xlabel('Year', fontsize=14)
    ax.set_ylabel(f'{target} (MW)', fontsize=14)
    ax.legend(loc='upper left', fontsize=10)
    plt.tight_layout()
    os.makedirs(out_dir, exist_ok=True)
    plt.savefig(os.path.join(out_dir, f'{country}_preds_{suffix}.png'), dpi=300)
    plt.close()

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)
    return path

# ==============================================================================
# --- 4. FULL_PIPELINE ---
# ==============================================================================
class ModelRunner:
    def __init__(self, data_path, out_dir, target, suffix=""):
        self.data_path = data_path
        self.out_dir = out_dir
        self.target = target
        self.suffix = suffix

        self.models_dir = ensure_dir(os.path.join(out_dir, 'saved_models'))
        self.plots_dir = ensure_dir(os.path.join(out_dir, 'comparison_plots'))
        self.preds_dir = ensure_dir(os.path.join(out_dir, 'predictions'))

        setup_logging(os.path.join(out_dir, 'run_log.log'))
        self.df = pd.read_csv(data_path)
        self.metrics = []

    def run(self, countries=None):
        logging.info("--- Starting Data Preprocessing ---")
        # 删高缺失列（保留关键列）
        ratio = self.df.isnull().sum() / len(self.df)
        to_drop = ratio[ratio > 0.1].index
        self.df.drop(columns=[c for c in to_drop if c not in ['Country', 'Year', self.target]], inplace=True)
        # 分国排序并前后向填充
        self.df = self.df.groupby('Country', group_keys=False).apply(lambda g: g.sort_values('Year').ffill().bfill())
        logging.info("--- Data Preprocessing Complete ---")

        if countries:
            self.df = self.df[self.df['Country'].isin(countries)].copy()

        for country in tqdm(self.df['Country'].unique(), desc=f"Processing Countries ({self.suffix})"):
            self._process_country(country)

        # 保存全局指标汇总
        pd.DataFrame(self.metrics).to_csv(os.path.join(self.out_dir, 'full_metrics_summary.csv'), index=False)
        logging.info("Saved metrics summary: full_metrics_summary.csv")

    def _process_country(self, country):
        c_data = self.df[self.df['Country'] == country].sort_values('Year')
        c_data = c_data[c_data[self.target] > 0].reset_index(drop=True)
        if len(c_data) < 10:
            logging.warning(f"Skipping {country}: data too short.")
            return

        split_idx = int(len(c_data) * 0.7)
        train, test = c_data.iloc[:split_idx], c_data.iloc[split_idx:]
        if len(test) == 0:
            logging.warning(f"Skipping {country}: no test data.")
            return

        logging.info(f"--- Processing {country} ({self.suffix}) ---")
        feats = [c for c in self.df.columns if c not in ['Country', 'Year', self.target]]
        if len(feats) == 0:
            logging.warning(f"Skipping {country}: no feature columns.")
            return

        X_train, y_train = train[feats], train[self.target]
        X_test,  y_test  = test[feats],  test[self.target]
        t_train, t_test  = train['Year'] - train['Year'].min(), test['Year'] - train['Year'].min()

        preds_map = {}

        # --------- 机器学习模型 ----------
        ml_models = {
            'LinearRegression': LinearRegression(),
            'Lasso': Lasso(),
            'Ridge': Ridge(),
            'RandomForest': RandomForestRegressor(random_state=42),
            # 对于 xgboost<2.0 没有 device 参数，可自动忽略；为了兼容，指定 tree_method
            'XGBoost': XGBRegressor(
                random_state=42,
                tree_method='hist',
                # 如果你的 xgboost>=2.0，下面这一行生效；否则不传也可以
                **({'device': DEVICE} if 'device' in XGBRegressor().get_params().keys() else {})
            ),
            'CatBoost': CatBoostRegressor(random_state=42, verbose=0, allow_writing_files=False)
        }

        for name, model in ml_models.items():
            met, pred = self._train_eval_save_ml(model, name, country, X_train, y_train, X_test, y_test)
            if met:
                self.metrics.append(met)
                preds_map[name] = pred

        # --------- 趋势模型（Logistic / Exponential） ----------
        for name, func in TREND_MODELS.items():
            met, pred = self._train_eval_save_trend(func, name, country, t_train, y_train, t_test, y_test)
            if met:
                self.metrics.append(met)
                preds_map[name] = pred

        # --------- 组装并保存预测 CSV + 绘图 ----------
        results = pd.DataFrame({'Year': c_data['Year'], 'Actual': c_data[self.target]})
        for name, pred in preds_map.items():
            p_series = pd.Series(np.nan, index=results.index)
            p_series.iloc[split_idx:] = pred
            results[name] = p_series
            results[f'{name}_residual'] = results['Actual'] - results[name]

        # 保存到 predictions/{country}_predictions.csv
        out_csv = os.path.join(self.preds_dir, f"{country}_predictions.csv")
        results.to_csv(out_csv, index=False)
        logging.info(f"Saved predictions: {out_csv}")

        # 绘图
        plot_unified_comparison(
            country, results, self.plots_dir, self.target, self.suffix, c_data.iloc[split_idx]['Year']
        )

    def _train_eval_save_ml(self, model, name, country, X_train, y_train, X_test, y_test):
        try:
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            metrics = calculate_metrics(y_test, preds)
            metrics.update({'Country': country, 'Model': name})
            logging.info(f"    - {country} | {name:<16} | Test MAPE: {metrics['MAPE']:.2f}%")

            country_model_dir = ensure_dir(os.path.join(self.models_dir, country))
            joblib.dump(model, os.path.join(country_model_dir, f'{name}.joblib'))

            return metrics, preds
        except Exception as e:
            logging.error(f"Failed ML model {name} for {country}: {e}")
            return None, None

    def _train_eval_save_trend(self, func, name, country, t_train, y_train, t_test, y_test):
        try:
            # 初始参数
            if name == 'Logistic':
                p0 = [max(y_train.max(), 1e-6), 0.5, len(t_train)/2]
                bounds = (0, [np.inf, 5.0, np.inf])  # 合理约束
                params, _ = curve_fit(func, t_train, y_train, p0=p0, maxfev=10000, bounds=bounds)
            else:
                a0 = y_train.iloc[0] if y_train.iloc[0] > 0 else max(y_train.median(), 1.0)
                p0 = [a0, 0.1]
                bounds = (0, [np.inf, 2.0])
                params, _ = curve_fit(func, t_train, y_train, p0=p0, maxfev=10000, bounds=bounds)

            preds = func(t_test.to_numpy(), *params)
            metrics = calculate_metrics(y_test, preds)
            metrics.update({'Country': country, 'Model': name})
            logging.info(f"    - {country} | {name:<16} | Test MAPE: {metrics['MAPE']:.2f}%")

            country_model_dir = ensure_dir(os.path.join(self.models_dir, country))
            joblib.dump({'func_name': name, 'params': params}, os.path.join(country_model_dir, f'{name}.joblib'))

            return metrics, preds
        except Exception as e:
            logging.error(f"Failed Trend model {name} for {country}: {e}")
            return None, None

# ==============================================================================
# --- 5. PLOT_ONLY ---
# ==============================================================================
def run_plot_only_mode(data_path, models_base_dir, countries, target, suffix):
    setup_logging(os.path.join(models_base_dir, 'plot_only_log.log'))
    logging.info(f"--- Starting PLOT_ONLY mode for {suffix} ---")

    df_full = pd.read_csv(data_path)
    output_plot_dir = ensure_dir(os.path.join(models_base_dir, 'plots_from_plot_only'))
    output_pred_dir = ensure_dir(os.path.join(models_base_dir, 'predictions_plot_only'))

    logging.info("--- Applying preprocessing steps ---")
    ratio = df_full.isnull().sum() / len(df_full)
    to_drop = ratio[ratio > 0.1].index
    df_full.drop(columns=[c for c in to_drop if c not in ['Country', 'Year', target]], inplace=True)
    df_full = df_full.groupby('Country', group_keys=False).apply(lambda g: g.sort_values('Year').ffill().bfill())
    logging.info("--- Preprocessing complete ---")

    for country in tqdm(countries, desc=f"Plotting for {suffix}"):
        country_data = df_full[df_full['Country'] == country].sort_values('Year')
        country_data = country_data[country_data[target] > 0].reset_index(drop=True)
        if len(country_data) < 10:
            logging.warning(f"Skipping {country}: data too short.")
            continue

        split_idx = int(len(country_data) * 0.7)
        train_data, test_data = country_data.iloc[:split_idx], country_data.iloc[split_idx:]
        if len(test_data) == 0:
            logging.warning(f"Skipping {country}: no test data.")
            continue

        features = [c for c in country_data.columns if c not in ['Country', 'Year', target]]
        if len(features) == 0:
            logging.warning(f"Skipping {country}: no feature columns.")
            continue

        X_test, y_test = test_data[features], test_data[target]
        t_test = test_data['Year'] - train_data['Year'].min()

        results_df = pd.DataFrame({'Year': country_data['Year'], 'Actual': country_data[target]})

        any_model_loaded = False
        for model_path in glob.glob(os.path.join(models_base_dir, 'saved_models', country, '*.joblib')):
            name = os.path.basename(model_path).replace('.joblib', '')
            try:
                model_obj = joblib.load(model_path)

                if isinstance(model_obj, dict) and 'func_name' in model_obj:
                    # 趋势模型
                    params = model_obj['params']
                    func = TREND_MODELS[model_obj['func_name']]
                    preds = func(t_test.to_numpy(), *params)
                else:
                    # ML 模型
                    if hasattr(model_obj, 'feature_names_in_'):
                        X_pred = X_test[model_obj.feature_names_in_]
                    else:
                        X_pred = X_test
                    preds = model_obj.predict(X_pred)

                metrics = calculate_metrics(y_test, preds)
                logging.info(f"    - {country} | {name:<16} | Test MAPE: {metrics['MAPE']:.2f}%")

                p_series = pd.Series(np.nan, index=results_df.index)
                p_series.iloc[split_idx:] = preds
                results_df[name] = p_series
                results_df[f'{name}_residual'] = results_df['Actual'] - results_df[name]
                any_model_loaded = True
            except Exception as e:
                logging.error(f"Failed to plot model {name} for {country}: {e}")

        if not any_model_loaded:
            logging.warning(f"No models found for {country}. Skipped plotting and CSV saving.")
            continue

        # 保存预测结果 CSV
        out_csv = os.path.join(output_pred_dir, f"{country}_predictions.csv")
        results_df.to_csv(out_csv, index=False)
        logging.info(f"Saved PLOT_ONLY predictions: {out_csv}")

        # 绘图
        plot_unified_comparison(
            country, results_df, output_plot_dir, target, suffix, country_data.iloc[split_idx]['Year']
        )

# ==============================================================================
# --- 6. 主流程 ---
# ==============================================================================
if __name__ == "__main__":
    if EXECUTION_MODE == 'FULL_PIPELINE':
        print("===== STARTING: Full Pipeline Run =====")
        runner = ModelRunner(INPUT_FILE_PATH, OUTPUT_DIR_BASELINE, TARGET_COLUMN_NAME, "All_Features")
        runner.run(COUNTRIES_TO_RUN)
        print(f"\n--- Pipeline finished. Results in '{OUTPUT_DIR_BASELINE}' ---")

    elif EXECUTION_MODE == 'PLOT_ONLY':
        print("===== RUNNING IN PLOT_ONLY MODE =====")
        run_plot_only_mode(INPUT_FILE_PATH, OUTPUT_DIR_BASELINE, COUNTRIES_TO_RUN, TARGET_COLUMN_NAME, "All_Features")
        print(f"\n✅✅✅ PLOT_ONLY mode finished!")

    else:
        print(f"Error: Unknown EXECUTION_MODE '{EXECUTION_MODE}'.")
