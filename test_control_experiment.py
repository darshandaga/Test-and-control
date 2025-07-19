import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import pairwise_distances
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Tuple, Dict
import warnings
warnings.filterwarnings('ignore')

class TestControlExperiment:
    """
    A comprehensive class for designing and analyzing test vs control campaigns.
    
    Test: Call users using state phone code numbers
    Control: Call users using a toll-free number
    Goal: See if local numbers increase pickup rates
    """
    
    def __init__(self, data_path: str):
        """Initialize with dataset"""
        self.data = pd.read_excel(data_path)
        self.original_data = self.data.copy()
        self.test_group = None
        self.control_group = None
        self.propensity_scores = None
        self.matched_pairs = None
        
    def explore_data(self):
        """Explore the dataset structure and distributions"""
        print("=== DATASET EXPLORATION ===")
        print(f"Dataset shape: {self.data.shape}")
        print(f"\nColumns: {list(self.data.columns)}")
        print(f"\nData types:\n{self.data.dtypes}")
        
        print(f"\n=== MISSING VALUES ===")
        print(self.data.isnull().sum())
        
        print(f"\n=== CATEGORICAL VARIABLES DISTRIBUTION ===")
        categorical_cols = ['state', 'age_group', 'gender', 'insurance_type', 'purpose_of_call']
        for col in categorical_cols:
            print(f"\n{col}:")
            print(self.data[col].value_counts())
            
        print(f"\n=== NUMERICAL VARIABLES SUMMARY ===")
        numerical_cols = ['age', 'prior_exposure_count', 'prior_engagement_score']
        print(self.data[numerical_cols].describe())
        
        return self.data.describe()
    
    def create_stratification(self, strata_vars: list = None):
        """
        Step 1-2: Create stratification based on key variables
        """
        if strata_vars is None:
            strata_vars = ['state', 'age_group', 'insurance_type']
            
        print(f"\n=== CREATING STRATIFICATION ===")
        print(f"Stratification variables: {strata_vars}")
        
        # Create engagement score quartiles for stratification
        self.data['engagement_quartile'] = pd.qcut(
            self.data['prior_engagement_score'], 
            q=4, 
            labels=['Q1', 'Q2', 'Q3', 'Q4']
        )
        
        # Add engagement quartile to stratification if not already included
        if 'engagement_quartile' not in strata_vars:
            strata_vars.append('engagement_quartile')
        
        # Create strata
        self.data['stratum'] = self.data[strata_vars].apply(
            lambda x: '_'.join(x.astype(str)), axis=1
        )
        
        strata_counts = self.data['stratum'].value_counts()
        print(f"\nNumber of strata created: {len(strata_counts)}")
        print(f"Strata sizes (top 10):")
        print(strata_counts.head(10))
        
        # Filter out strata with less than 4 observations (need at least 2 for each group)
        valid_strata = strata_counts[strata_counts >= 4].index
        self.data = self.data[self.data['stratum'].isin(valid_strata)]
        
        print(f"\nAfter filtering small strata:")
        print(f"Remaining observations: {len(self.data)}")
        print(f"Remaining strata: {len(self.data['stratum'].unique())}")
        
        return self.data['stratum'].value_counts()
    
    def calculate_propensity_scores(self):
        """
        Step 3 (Corrected): Calculate propensity scores using covariates to predict treatment assignment
        """
        print(f"\n=== CALCULATING PROPENSITY SCORES ===")
        
        # Prepare features for propensity score model
        # One-hot encode categorical variables
        categorical_features = pd.get_dummies(
            self.data[['state', 'age_group', 'gender', 'insurance_type', 'purpose_of_call']], 
            drop_first=True
        )
        
        # Combine with numerical features
        numerical_features = self.data[['age', 'prior_exposure_count', 'prior_engagement_score']]
        
        X = pd.concat([numerical_features, categorical_features], axis=1)
        
        # Standardize features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # For propensity score calculation, we need to simulate treatment assignment
        # In practice, this would be based on actual treatment assignment mechanism
        # Here we'll create a balanced assignment within strata
        np.random.seed(42)  # For reproducibility
        
        treatment_assignment = []
        for stratum in self.data['stratum'].unique():
            stratum_indices = self.data[self.data['stratum'] == stratum].index
            n_stratum = len(stratum_indices)
            n_treatment = n_stratum // 2
            
            # Randomly assign half to treatment
            treatment_indices = np.random.choice(stratum_indices, n_treatment, replace=False)
            stratum_treatment = [1 if idx in treatment_indices else 0 for idx in stratum_indices]
            treatment_assignment.extend(stratum_treatment)
        
        # Fit propensity score model
        propensity_model = LogisticRegression(random_state=42, max_iter=1000)
        propensity_model.fit(X_scaled, treatment_assignment)
        
        # Calculate propensity scores
        self.propensity_scores = propensity_model.predict_proba(X_scaled)[:, 1]
        self.data['propensity_score'] = self.propensity_scores
        self.data['treatment_assignment'] = treatment_assignment
        
        print(f"Propensity scores calculated.")
        print(f"Propensity score range: {self.propensity_scores.min():.3f} - {self.propensity_scores.max():.3f}")
        print(f"Mean propensity score: {self.propensity_scores.mean():.3f}")
        
        return self.propensity_scores
    
    def perform_matching(self, method='nearest_neighbor', caliper=0.1):
        """
        Step 4: Perform matching based on propensity scores within strata
        """
        print(f"\n=== PERFORMING MATCHING ===")
        print(f"Matching method: {method}")
        
        matched_pairs = []
        
        for stratum in self.data['stratum'].unique():
            stratum_data = self.data[self.data['stratum'] == stratum].copy()
            
            treatment_group = stratum_data[stratum_data['treatment_assignment'] == 1]
            control_group = stratum_data[stratum_data['treatment_assignment'] == 0]
            
            if len(treatment_group) == 0 or len(control_group) == 0:
                continue
                
            if method == 'nearest_neighbor':
                # Find nearest neighbor matches based on propensity scores
                for _, treated_unit in treatment_group.iterrows():
                    if len(control_group) == 0:
                        break  # No more control units available
                        
                    # Calculate distances to all control units in the same stratum
                    distances = np.abs(
                        control_group['propensity_score'] - treated_unit['propensity_score']
                    )
                    
                    if len(distances) == 0:
                        continue  # Skip if no control units
                    
                    # Find closest match within caliper
                    min_distance_idx = distances.idxmin()
                    min_distance = distances.min()
                    
                    if min_distance <= caliper:
                        matched_pairs.append({
                            'stratum': stratum,
                            'treated_id': treated_unit['patient_id'],
                            'control_id': control_group.loc[min_distance_idx, 'patient_id'],
                            'treated_ps': treated_unit['propensity_score'],
                            'control_ps': control_group.loc[min_distance_idx, 'propensity_score'],
                            'ps_difference': min_distance
                        })
                        
                        # Remove matched control unit to avoid reuse
                        control_group = control_group.drop(min_distance_idx)
        
        self.matched_pairs = pd.DataFrame(matched_pairs)
        
        print(f"Matching completed.")
        print(f"Number of matched pairs: {len(self.matched_pairs)}")
        print(f"Mean propensity score difference: {self.matched_pairs['ps_difference'].mean():.4f}")
        print(f"Max propensity score difference: {self.matched_pairs['ps_difference'].max():.4f}")
        
        return self.matched_pairs
    
    def assign_treatment_groups(self):
        """
        Step 5: Assign final treatment groups based on matching
        """
        print(f"\n=== ASSIGNING TREATMENT GROUPS ===")
        
        if self.matched_pairs is None:
            print("No matched pairs found. Using stratified randomization.")
            # Fallback to stratified randomization
            self.test_group = self.data[self.data['treatment_assignment'] == 1].copy()
            self.control_group = self.data[self.data['treatment_assignment'] == 0].copy()
        else:
            # Use matched pairs
            test_ids = self.matched_pairs['treated_id'].tolist()
            control_ids = self.matched_pairs['control_id'].tolist()
            
            self.test_group = self.data[self.data['patient_id'].isin(test_ids)].copy()
            self.control_group = self.data[self.data['patient_id'].isin(control_ids)].copy()
        
        # Add treatment labels
        self.test_group['treatment'] = 'Local Phone Numbers'
        self.control_group['treatment'] = 'Toll-Free Numbers'
        
        print(f"Test group size: {len(self.test_group)}")
        print(f"Control group size: {len(self.control_group)}")
        
        # Check balance
        self.check_balance()
        
        return self.test_group, self.control_group
    
    def check_balance(self):
        """Check covariate balance between treatment groups"""
        print(f"\n=== CHECKING COVARIATE BALANCE ===")
        
        # Combine groups for comparison
        combined = pd.concat([self.test_group, self.control_group])
        
        balance_results = []
        
        # Check categorical variables
        categorical_vars = ['state', 'age_group', 'gender', 'insurance_type', 'purpose_of_call']
        for var in categorical_vars:
            test_dist = self.test_group[var].value_counts(normalize=True)
            control_dist = self.control_group[var].value_counts(normalize=True)
            
            # Chi-square test
            contingency_table = pd.crosstab(combined[var], combined['treatment'])
            chi2, p_value = stats.chi2_contingency(contingency_table)[:2]
            
            balance_results.append({
                'variable': var,
                'test_type': 'Chi-square',
                'statistic': chi2,
                'p_value': p_value,
                'balanced': p_value > 0.05
            })
        
        # Check numerical variables
        numerical_vars = ['age', 'prior_exposure_count', 'prior_engagement_score']
        for var in numerical_vars:
            test_mean = self.test_group[var].mean()
            control_mean = self.control_group[var].mean()
            
            # T-test
            t_stat, p_value = stats.ttest_ind(
                self.test_group[var], 
                self.control_group[var]
            )
            
            balance_results.append({
                'variable': var,
                'test_type': 'T-test',
                'test_mean': test_mean,
                'control_mean': control_mean,
                'statistic': t_stat,
                'p_value': p_value,
                'balanced': p_value > 0.05
            })
        
        balance_df = pd.DataFrame(balance_results)
        print("\nBalance Check Results:")
        print(balance_df)
        
        balanced_vars = balance_df['balanced'].sum()
        total_vars = len(balance_df)
        print(f"\nBalanced variables: {balanced_vars}/{total_vars}")
        
        return balance_df
    
    def simulate_campaign_results(self, base_pickup_rate=0.3, treatment_effect=0.05):
        """
        Step 6: Simulate campaign results (pickup rates)
        In practice, this would be actual campaign data
        """
        print(f"\n=== SIMULATING CAMPAIGN RESULTS ===")
        print(f"Base pickup rate: {base_pickup_rate}")
        print(f"Treatment effect: +{treatment_effect}")
        
        np.random.seed(42)
        
        # Simulate pickup rates with some realistic variation based on covariates
        def simulate_pickup(group_data, is_treatment=False):
            # Base rate varies by engagement score and age
            base_rates = (
                base_pickup_rate + 
                0.2 * group_data['prior_engagement_score'] +  # Higher engagement = higher pickup
                0.1 * (group_data['age'] < 40).astype(int) -  # Younger people more likely to pick up
                0.05 * (group_data['prior_exposure_count'] > 3).astype(int)  # Fatigue effect
            )
            
            # Add treatment effect
            if is_treatment:
                base_rates += treatment_effect
            
            # Generate binary outcomes
            pickup_rates = np.random.binomial(1, np.clip(base_rates, 0, 1))
            return pickup_rates
        
        # Simulate outcomes
        self.test_group['picked_up'] = simulate_pickup(self.test_group, is_treatment=True)
        self.control_group['picked_up'] = simulate_pickup(self.control_group, is_treatment=False)
        
        test_pickup_rate = self.test_group['picked_up'].mean()
        control_pickup_rate = self.control_group['picked_up'].mean()
        
        print(f"\nSimulated Results:")
        print(f"Test group pickup rate: {test_pickup_rate:.3f}")
        print(f"Control group pickup rate: {control_pickup_rate:.3f}")
        print(f"Observed treatment effect: {test_pickup_rate - control_pickup_rate:.3f}")
        
        return test_pickup_rate, control_pickup_rate
    
    def analyze_results(self):
        """
        Step 7: Comprehensive analysis of campaign results
        """
        print(f"\n=== ANALYZING CAMPAIGN RESULTS ===")
        
        # Basic statistics
        test_pickup_rate = self.test_group['picked_up'].mean()
        control_pickup_rate = self.control_group['picked_up'].mean()
        treatment_effect = test_pickup_rate - control_pickup_rate
        
        # Statistical tests
        # 1. Two-sample t-test
        t_stat, t_p_value = stats.ttest_ind(
            self.test_group['picked_up'], 
            self.control_group['picked_up']
        )
        
        # 2. Chi-square test
        contingency = pd.crosstab(
            pd.concat([self.test_group, self.control_group])['treatment'],
            pd.concat([self.test_group, self.control_group])['picked_up']
        )
        chi2, chi2_p_value = stats.chi2_contingency(contingency)[:2]
        
        # 3. Effect size (Cohen's d)
        pooled_std = np.sqrt(
            ((len(self.test_group) - 1) * self.test_group['picked_up'].var() + 
             (len(self.control_group) - 1) * self.control_group['picked_up'].var()) /
            (len(self.test_group) + len(self.control_group) - 2)
        )
        cohens_d = treatment_effect / pooled_std
        
        # 4. Confidence interval for treatment effect
        se_diff = np.sqrt(
            test_pickup_rate * (1 - test_pickup_rate) / len(self.test_group) +
            control_pickup_rate * (1 - control_pickup_rate) / len(self.control_group)
        )
        ci_lower = treatment_effect - 1.96 * se_diff
        ci_upper = treatment_effect + 1.96 * se_diff
        
        # Results summary
        results = {
            'test_group_size': len(self.test_group),
            'control_group_size': len(self.control_group),
            'test_pickup_rate': test_pickup_rate,
            'control_pickup_rate': control_pickup_rate,
            'treatment_effect': treatment_effect,
            'treatment_effect_ci_lower': ci_lower,
            'treatment_effect_ci_upper': ci_upper,
            't_statistic': t_stat,
            't_p_value': t_p_value,
            'chi2_statistic': chi2,
            'chi2_p_value': chi2_p_value,
            'cohens_d': cohens_d,
            'statistically_significant': t_p_value < 0.05
        }
        
        print(f"\n=== FINAL RESULTS ===")
        print(f"Test Group (Local Numbers): {test_pickup_rate:.1%} pickup rate (n={len(self.test_group)})")
        print(f"Control Group (Toll-Free): {control_pickup_rate:.1%} pickup rate (n={len(self.control_group)})")
        print(f"Treatment Effect: {treatment_effect:.1%} (95% CI: {ci_lower:.1%} to {ci_upper:.1%})")
        print(f"Statistical Significance: {'Yes' if results['statistically_significant'] else 'No'} (p={t_p_value:.4f})")
        print(f"Effect Size (Cohen's d): {cohens_d:.3f}")
        
        # Interpretation
        if results['statistically_significant']:
            if treatment_effect > 0:
                print(f"\n✅ CONCLUSION: Local phone numbers significantly INCREASE pickup rates")
            else:
                print(f"\n❌ CONCLUSION: Local phone numbers significantly DECREASE pickup rates")
        else:
            print(f"\n➖ CONCLUSION: No significant difference between local and toll-free numbers")
        
        return results
    
    def create_visualizations(self):
        """Create visualizations for the experiment results"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Pickup rates by treatment group
        combined_data = pd.concat([self.test_group, self.control_group])
        pickup_by_treatment = combined_data.groupby('treatment')['picked_up'].agg(['mean', 'count'])
        
        axes[0, 0].bar(pickup_by_treatment.index, pickup_by_treatment['mean'])
        axes[0, 0].set_title('Pickup Rates by Treatment Group')
        axes[0, 0].set_ylabel('Pickup Rate')
        axes[0, 0].set_ylim(0, 1)
        
        # Add value labels on bars
        for i, v in enumerate(pickup_by_treatment['mean']):
            axes[0, 0].text(i, v + 0.01, f'{v:.1%}', ha='center')
        
        # 2. Propensity score distribution
        axes[0, 1].hist(self.test_group['propensity_score'], alpha=0.7, label='Test Group', bins=20)
        axes[0, 1].hist(self.control_group['propensity_score'], alpha=0.7, label='Control Group', bins=20)
        axes[0, 1].set_title('Propensity Score Distribution')
        axes[0, 1].set_xlabel('Propensity Score')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].legend()
        
        # 3. Pickup rates by state
        pickup_by_state = combined_data.groupby(['state', 'treatment'])['picked_up'].mean().unstack()
        pickup_by_state.plot(kind='bar', ax=axes[1, 0])
        axes[1, 0].set_title('Pickup Rates by State and Treatment')
        axes[1, 0].set_ylabel('Pickup Rate')
        axes[1, 0].legend(title='Treatment')
        axes[1, 0].tick_params(axis='x', rotation=45)
        
        # 4. Treatment effect by engagement score quartile
        combined_data['engagement_quartile'] = pd.qcut(
            combined_data['prior_engagement_score'], 
            q=4, 
            labels=['Q1', 'Q2', 'Q3', 'Q4']
        )
        pickup_by_engagement = combined_data.groupby(['engagement_quartile', 'treatment'])['picked_up'].mean().unstack()
        pickup_by_engagement.plot(kind='bar', ax=axes[1, 1])
        axes[1, 1].set_title('Pickup Rates by Engagement Quartile')
        axes[1, 1].set_ylabel('Pickup Rate')
        axes[1, 1].legend(title='Treatment')
        axes[1, 1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig('campaign_analysis_results.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return fig
    
    def export_results(self, filename='campaign_experiment_results.xlsx'):
        """Export all results to Excel file"""
        with pd.ExcelWriter(filename, engine='openpyxl') as writer:
            # Original data with assignments
            self.data.to_excel(writer, sheet_name='Full_Dataset', index=False)
            
            # Test and control groups
            self.test_group.to_excel(writer, sheet_name='Test_Group', index=False)
            self.control_group.to_excel(writer, sheet_name='Control_Group', index=False)
            
            # Matched pairs (if available)
            if self.matched_pairs is not None:
                self.matched_pairs.to_excel(writer, sheet_name='Matched_Pairs', index=False)
        
        print(f"\nResults exported to {filename}")

def main():
    """Main execution function"""
    print("=== TEST vs CONTROL CAMPAIGN EXPERIMENT ===")
    print("Objective: Test if local phone numbers increase pickup rates vs toll-free numbers")
    
    # Initialize experiment
    experiment = TestControlExperiment('data/test_learn_campaign_dataset.xlsx')
    
    # Step 1: Explore data
    experiment.explore_data()
    
    # Step 2: Create stratification
    experiment.create_stratification()
    
    # Step 3: Calculate propensity scores
    experiment.calculate_propensity_scores()
    
    # Step 4: Perform matching
    experiment.perform_matching()
    
    # Step 5: Assign treatment groups
    experiment.assign_treatment_groups()
    
    # Step 6: Simulate campaign results
    experiment.simulate_campaign_results()
    
    # Step 7: Analyze results
    results = experiment.analyze_results()
    
    # Create visualizations
    experiment.create_visualizations()
    
    # Export results
    experiment.export_results()
    
    return experiment, results

if __name__ == "__main__":
    experiment, results = main()
