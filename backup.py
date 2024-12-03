# %%
from sklearn.model_selection import cross_validate, train_test_split
from sklearn.metrics import make_scorer, mean_squared_error, mean_absolute_error, r2_score
import time
import pandas as pd
import lightgbm as lgb
import xgboost as xgb
import numpy as np
import matplotlib.pyplot as plt


# %%
df = pd.read_csv('SAdata_allMeasures.csv')

print(df.isnull().sum())

y = df['Y']
X = df.drop(columns=['Y'])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(X_train.shape)
df.head()

# %% [markdown]
# ## Preprocessing
# 
# Since many features have different 

# %%
from sklearn.preprocessing import StandardScaler

#from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)



# %% [markdown]
# ## Baseline LR

# %%
from sklearn.linear_model import LinearRegression

linear_model = LinearRegression()

linear_model.fit(X_train, y_train)

linear_predictions = linear_model.predict(X_test)

linear_mse = mean_squared_error(y_test, linear_predictions)
print("Linear Regression Mean Squared Error:", linear_mse)

linear_r2 = r2_score(y_test, linear_predictions)
print("Linear Regression R-squared:", linear_r2)

# %% [markdown]
# ## XGB

# %%
#https://stackoverflow.com/questions/68766331/how-to-apply-predict-to-xgboost-cross-validation
#https://xgboost.readthedocs.io/en/latest/python/examples/cross_validation.html

dtrain = xgb.DMatrix(X, label=y) 
#https://stackoverflow.com/questions/70127049/what-is-the-use-of-dmatrix

params = {
    "max_depth": 2, 
    "eta": 0.1,
    "objective": "reg:squarederror",
    "verbosity": 0
}

start_time_gbm = time.time()

results = xgb.cv(
    params,
    dtrain,
    num_boost_round=100,
    nfold=5,
    metrics=['rmse', 'mae'],
    early_stopping_rounds=10,
    callbacks=[xgb.callback.EvaluationMonitor()], #https://xgboosting.com/how-to-use-xgboost-evaluationmonitor-callback/
    seed=42,
    verbose_eval=False
)

stop_time_gbm = time.time()

gbm_train_time = stop_time_gbm - start_time_gbm

gbm_rsme = results['test-rmse-mean']
gbm_mae = results['test-mae-mean']

gbm_rsme_train = results['train-rmse-mean']
gbm_mae_train = results['train-mae-mean']




# %% [markdown]
# # LGB

# %%
params = {
    'objective': 'regression',
    'num_leaves': 20,              # Smaller for small dataset
    'learning_rate': 0.1,          # Slightly higher
    'min_data_in_leaf': 5,         # Smaller for small dataset
    'bagging_freq': 5,
    'lambda_l1': 0.1,              # Some regularization
    'metric': ['rmse', 'mae'],
    'verbose': -1,
}

train_data = lgb.Dataset(X_train, label=y_train)

#https://lightgbm.readthedocs.io/en/latest/pythonapi/lightgbm.early_stopping.htmlxw
#https://stackoverflow.com/questions/49774825/how-to-use-lightgbm-cv-for-regression

# Store evaluation history
evals_result = {}
record_eval = lgb.record_evaluation(evals_result)
early_stopping_callback = lgb.early_stopping(5, first_metric_only=False, verbose=False)

start_time_lgb = time.time()
cv_results = lgb.cv(
    params,
    train_data,
    num_boost_round=100,
    nfold=10,
    callbacks=[record_eval, early_stopping_callback],
    stratified=False, 
    eval_train_metric=True, 
    seed=42
)
stop_time_lgb = time.time()


############    Results    ############
lgb_train_time = stop_time_lgb - start_time_lgb

lgb_train_rmse = evals_result['train']['rmse-mean']
lgb_train_mae = evals_result['train']['l1-mean']

lgb_valid_rmse = evals_result['valid']['rmse-mean']
lgb_valid_mae = evals_result['valid']['l1-mean']

# %% [markdown]
# ## LGB Distributed

# %%
params = {
        'objective': 'regression',
        'metric': ['rmse', 'mae'],
        'num_leaves': 20,           
        'learning_rate': 0.1,     
        'min_data_in_leaf': 5,    
        'bagging_freq': 5,
        'verbose': -1,       
        # Distributed settings
        'num_threads': 8,
        'tree_learner': 'feature', 
        'force_row_wise': True
    }



# Store evaluation history
evals_result_distributed = {}
record_eval = lgb.record_evaluation(evals_result)
early_stopping_callback = lgb.early_stopping(5, first_metric_only=False, verbose=False)



start_time_lgb = time.time()

cv_results = lgb.cv(
    params,
    train_data,
    num_boost_round=100,
    nfold=10,
    callbacks=[record_eval, early_stopping_callback],
    stratified=False, 
    eval_train_metric=True, 
    seed=42
)

stop_time_lgb = time.time()

print(evals_result)

############    Results    ############
lgb_train_time = stop_time_lgb - start_time_lgb

lgb_train_rmse_disributed = evals_result['train']['rmse-mean']
lgb_train_mae_disributed = evals_result['train']['l1-mean']

lgb_valid_rmse_disributed = evals_result['valid']['rmse-mean']
lgb_valid_mae_disributed = evals_result['valid']['l1-mean']


# %% [markdown]
# ## REPORT

# %%
#AI GENERATED CODE
def plot_split_metrics(metrics_dict):
    """
    Plot training and validation metrics in separate subplots
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 12))
    
    # Color palette - same base color with different shades for each model
    colors = {
        'LightGBM': {
            'rmse': '#1a53ff',  # Darker blue
            'mae': '#668cff'    # Lighter blue
        },
        'XGBoost': {
            'rmse': '#ff1a1a',  # Darker red
            'mae': '#ff6666'    # Lighter red
        },
        'Distributed LightGBM': {
            'rmse': '#00cc00',  # Darker green
            'mae': '#66ff66'    # Lighter green
        }
    }
    
    # Training metrics plot
    for model_name, metrics in metrics_dict.items():
        color_dict = colors[model_name]
        
        # Plot training metrics
        ax1.plot(metrics['train_rmse'], label=f'{model_name} RMSE', 
                color=color_dict['rmse'], linewidth=2)
        ax1.plot(metrics['train_mae'], label=f'{model_name} MAE', 
                color=color_dict['mae'], linewidth=2)
    
    ax1.grid(True, linestyle='--', alpha=0.7)
    ax1.set_xlabel('Iterations', fontsize=12)
    ax1.set_ylabel('Error', fontsize=12)
    ax1.set_title('Training Metrics', fontsize=14, pad=15)
    ax1.legend(frameon=True, fancybox=True, shadow=True, fontsize=10)
    ax1.set_facecolor('#f8f9fa')
    
    # Validation metrics plot
    for model_name, metrics in metrics_dict.items():
        color_dict = colors[model_name]
        
        # Plot validation metrics
        ax2.plot(metrics['valid_rmse'], label=f'{model_name} RMSE', 
                color=color_dict['rmse'], linewidth=2)
        ax2.plot(metrics['valid_mae'], label=f'{model_name} MAE', 
                color=color_dict['mae'], linewidth=2)
    
    ax2.grid(True, linestyle='--', alpha=0.7)
    ax2.set_xlabel('Iterations', fontsize=12)
    ax2.set_ylabel('Error', fontsize=12)
    ax2.set_title('Validation Metrics', fontsize=14, pad=15)
    ax2.legend(frameon=True, fancybox=True, shadow=True, fontsize=10)
    ax2.set_facecolor('#f8f9fa')
    
    plt.tight_layout()
    return fig

# %%
#show table of final results

results = pd.DataFrame({
    'Model': ['Linear Regression', 'XGBoost', 'LightGBM', 'Distributed LightGBM'],
    'Train Time': [0, gbm_train_time, lgb_train_time, lgb_train_time],
})

metrics_dict = {
    'XGBoost': {
        'train_rmse': gbm_rsme_train,
        'train_mae': gbm_mae_train,
        'valid_rmse': gbm_rsme,
        'valid_mae': gbm_mae
    },
    'LightGBM': {
        'train_rmse': lgb_train_rmse,
        'train_mae': lgb_train_mae,
        'valid_rmse': lgb_valid_rmse,
        'valid_mae': lgb_valid_mae
    },
    'Distributed LightGBM': {
        'train_rmse': lgb_train_rmse_disributed,
        'train_mae': lgb_train_mae_disributed,
        'valid_rmse': lgb_valid_rmse_disributed,
        'valid_mae': lgb_valid_mae_disributed
    }
}

plot_split_metrics(metrics_dict)

# %% [markdown]
# ## SHAP

# %%
#SHAP Analysis
import shap

shap.initjs()

explainer = shap.TreeExplainer(model_lgb)
shap_values = explainer.shap_values(X_test)
X_test_df = pd.DataFrame(X_test, columns=df.drop(columns=['Y']).columns) # Convert to DataFrame to display because scalar turns it into a numpy array
shap.waterfall_plot(shap.Explanation(values=shap_values[0], base_values=explainer.expected_value, data=X_test_df.iloc[0]), max_display=30)
shap.plots.force(explainer.expected_value, shap_values[0], X_test_df.iloc[0])
shap.plots.force(explainer.expected_value, shap_values[:100], X_test_df.iloc[:100])

shap.summary_plot(shap_values, X_test_df, plot_type='bar')
shap.summary_plot(shap_values, X_test_df)


# %%
X_test_df = pd.DataFrame(X_test, columns=df.drop(columns=['Y']).columns) # Convert to DataFrame to use iloc
shap.plots.force(explainer.expected_value, shap_values[:100], X_test_df.iloc[:100])
#export the plot
shap.save_html('shap_plot.html', shap.force_plot(explainer.expected_value, shap_values[:100], X_test_df.iloc[:100]))

# %%
shap.force_plot(explainer.expected_value, shap_values[:100], X_test_df.iloc[:100], matplotlib=True, show=False)
plt.savefig('shap_force_plot_multi.png', bbox_inches='tight', dpi=300)
plt.close()



