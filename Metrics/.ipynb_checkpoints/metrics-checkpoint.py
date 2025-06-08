import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, mean_squared_error
from scipy.spatial import distance

class SyntheticDataEvaluator:
    def __init__(self, original_data, synthetic_data):
        """
        Initialize with original and synthetic data.
        
        Parameters
        original_data (Dataframe): The original data with the target variable.
        synthetic_data (Dataframe): The synthetic data with the target variable.
        target_column (str): The name of the target column.
        task_type (str): 'classifiaction' or 'regression'
        """
        
        self.original_data = original_data
        self.synthetic_data = synthetic_data
        
    # Machine Learning Efficiency
    def mle(self, target_column: list, task_type: str = 'classification'):
        
        X_orig = self.original_data.drop(columns=[target_column])
        y_orig = self.original_data[target_column]
        X_train_orig, X_test_orig, y_train_orig, y_test_orig = train_test_split(X_orig, y_orig, test_size=0.2, random_state=42)
        
        X_synth = self.synthetic_data.drop(columns=[target_column])
        y_synth = self.synthetic_data[target_column]
        
        if task_type == "classification":
            models = {
            "Logistic Regression": LogisticRegression(),
            "Decision Tree": DecisionTreeClassifier(),
            "Random Forest": RandomForestClassifier()
            }
        elif task_type == "regression":
            models = {
            "Linear Regression": LinearRegression(),
            "Decision Tree": DecisionTreeRegressor(),
            "Random Forest": RandomForestRegressor()
            }
        else:
            raise ValueError("task_type must be 'classification' or 'regression'")
            
        results = {'Model': [], 'Original Data': [], 'Synthetic Data': []}
        
        for model_name, model in models.items():
            model.fit(X_train_orig, y_train_orig)
            y_pred_orig = model.predict(X_test_orig)
            if task_type == "classification":
                score = accuracy_score(y_test_orig, y_pred_orig)
            elif task_type == "regression":
                score = mean_squared_error(y_test_orig, y_pred_orig)
            
            results['Model'].append(model_name)
            results['Original Data'].append(score)
            
        for model_name, model in models.items():
            model.fit(X_synth, y_synth)
            y_pred_synth = model.predict(X_test_orig)
            if task_type == "classification":
                score = accuracy_score(y_test_orig, y_pred_synth)
            elif task_type == "regression":
                score = mean_squared_error(y_test_orig, y_pred_synth)
                
            results['Synthetic Data'].append(score)
            
        results_df = pd.DataFrame(results)
        return results_df
    
    def dcr(self):
        X_orig = self.original_data.values
        X_synth = self.synthetic_data.values
        
        distances = []
        for synth_record in X_synth:
            closest_dist = np.min([distance.euclidean(synth_record, orig_record) for orig_record in X_orig])
            distances.append(closest_dist)
            
        avg_distance = np.mean(distances)
        print(f"Average distance to closest original record: {avg_distance:.4f}")
        return avg_distance
    
    def dm(self):

        X_orig = self.original_data
        X_synth = self.synthetic_data
        
        X_orig['label'] = 1
        X_synth['label'] = 0
        
        combined_data = pd.concat([X_orig, X_synth], axis=0)
        
        X_combined = combined_data.drop(columns = ['label'])
        y_combined = combined_data['label']
        
        X_train, X_test, y_train, y_test = train_test_split(X_combined, y_combined, test_size=0.2, random_state=42)
        
        rf = RandomForestClassifier()
        param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [5, 10, 20],
        'min_samples_split': [2, 5, 10]
        }
        
        grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=3, scoring='accuracy', n_jobs=-1)
        grid_search.fit(X_train, y_train)
        
        best_rf = grid_search.best_estimator_
        
        y_pred = best_rf.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"Accuracy: {accuracy:.4f}")
        return accuracy
    
    
    def pc(self, columns: list = None):
        """
        - columns: List
            List of columns to compute correlation for. If None, all columns will be used 
            
        Returns:
            None (Displays correlation matrices and heatmaps for comparison).
        """
        
        # Use all columns if none are provided
        if columns is None:
            columns = self.original_data.columns.intersection(self.synthetic_data.columns)
            
        common_columns = list(set(columns).intersection(self.original_data.columns, self.synthetic_data.columns))
        
        if len(common_columns) == 0:
            raise ValueError("No common columns found between original and synthetic datasets")
            
        # Pearson correlation for original and synthetic dataset
        original_corr = self.original_data[common_columns].corr(method='pearson')
        synthetic_corr = self.synthetic_data[common_columns].corr(method='pearson')
        
        # Pearson correlation for combined original + synthetic dataset
        combined_data = pd.concat([self.original_data[common_columns], self.synthetic_data[common_columns]], axis=0, ignore_index=True)
        combined_corr = combined_data.corr(method='pearson')
        
        # Display correlation matrices
        #print("Original Data Correlation Matrix:")
        #print(original_corr)
        
        #print("\nSynthetic Data Correlation Matrix")
        #print(combined_corr)
        
        #print("\nCombined Original + Synthetic Data Correlation Matrix")
        #print(combined_corr)
        
        # Plot heatmaps for both correlation matrices
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))
        
        sns.heatmap(original_corr, annot=True, fmt='.2f', cmap='coolwarm', ax=axes[0, 0])
        axes[0, 0].set_title('Original Data Correlation')
        
        sns.heatmap(synthetic_corr, annot=True, fmt='.2f', cmap='coolwarm', ax=axes[0, 1])
        axes[0, 1].set_title('Synthetic Data Correlation')
        
        sns.heatmap(combined_corr, annot=True, fmt='.2f', cmap='coolwarm', ax=axes[1, 0])
        axes[1, 0].set_title('Original + Synthetic Data Correlation')
        
        # Calculate the similarity score
        similarity_matrix = 1 - (np.abs(synthetic_corr - original_corr)/2)
        
        sns.heatmap(similarity_matrix, annot=True, fmt='.2f', cmap='coolwarm', ax=axes[1, 1])
        axes[1, 1].set_title('Similarity Matrix')
        
        #print("\nSimilarity Matrix")
        #print(similarity_matrix)
        
        plt.tight_layout()
        plt.show()
        
        
    # For continuous variables    
    def kst(self, continuous_columns: list = None):
        
        if continuous_columns is None:
            continuous_columns = self.original_data.select_dtypes(include=['float64', 'int64']).columns
            
        common_columns = list(set(continuous_columns).intersection(self.original_data.columns, self.synthetic_data.columns))
        
        if len(common_columns) == 0:
            raise ValueError("No common continuous columns found between original and synthetic datasets.")
        
        # Initalize the dictionary to hold KS test results
        ks_results = {'Columns': [], 'KS Statistics': [], 'p-value': []}
        
        # Perform the KS test for each continous column
        for column in common_columns:
            original_values = self.original_data[column].dropna()
            synthetic_values = self.synthetic_data[column].dropna()
            ks_stat, p_value = stats.ks_2samp(original_values, synthetic_values)
            ks_results['Columns'].append(column)
            ks_results['KS Statistics'].append(ks_stat)
            ks_results['p-value'].append(p_value)
        
        ks_results_df = pd.DataFrame(ks_results)
        
        table = ks_results_df.style.background_gradient(subset=['KS Statistics'], cmap='coolwarm', vmin=0, vmax=1).format({'KS Statistics': '{:.4f}', 'p-value': '{:.4f}'})
            
        return table
        
    # For categorical variables    
    def tvd(self, categorical_columns: list = None):
        
        if categorical_columns is None:
            categorical_columns = self.original_data.select_dtypes(include=['object', 'category']).columns
            
        common_columns = list(set(categorical_columns).intersection(self.original_data.columns, self.synthetic_data.columns))
        
        if len(common_columns) == 0:
            raise ValueError("No common categorical columns found between original and synthetic datasets.")
        
        # Initalize the dictionary to hold KS test results
        tvd_results = {'Columns': [], 'TVD': []}
        
        for column in common_columns:
            original_counts = self.original_data[column].value_counts(normalize=True)
            synthetic_counts = self.synthetic_data[column].value_counts(normalize=True)
            
            combined_categories = pd.concat([original_counts, synthetic_counts], axis=1).fillna(0)
            combined_categories.columns = ['R', 'S']
            
            tvd = 0.5 * np.sum(np.abs(combined_categories['R'] - combined_categories['S']))
            
            tvd_results['Columns'].append(column)
            tvd_results['TVD'].append(tvd)
            
        tvd_results_df = pd.DataFrame(tvd_results)
        
        table = tvd_results_df.style.background_gradient(subset=['TVD'], cmap='coolwarm', vmin=0, vmax=1).format({'TVD': '{:.4f}'})
        
        return table
        
        