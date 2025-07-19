#!/usr/bin/env python3
"""
Demo script showing how to use the TestControlExperiment class
with different parameters and scenarios.
"""

from test_control_experiment import TestControlExperiment
import pandas as pd

def run_demo():
    """Run demonstration of the experiment with different scenarios"""
    
    print("=" * 60)
    print("TEST & CONTROL CAMPAIGN EXPERIMENT - DEMO")
    print("=" * 60)
    
    # Initialize experiment
    experiment = TestControlExperiment('data/test_learn_campaign_dataset.xlsx')
    
    print("\n🔍 SCENARIO 1: Standard Experiment")
    print("-" * 40)
    
    # Run standard experiment
    experiment.explore_data()
    experiment.create_stratification()
    experiment.calculate_propensity_scores()
    experiment.perform_matching()
    experiment.assign_treatment_groups()
    
    # Test different treatment effects
    scenarios = [
        {"name": "Conservative Effect", "base_rate": 0.25, "treatment_effect": 0.02},
        {"name": "Moderate Effect", "base_rate": 0.30, "treatment_effect": 0.05},
        {"name": "Strong Effect", "base_rate": 0.35, "treatment_effect": 0.10}
    ]
    
    results_summary = []
    
    for i, scenario in enumerate(scenarios, 1):
        print(f"\n📊 SCENARIO {i}: {scenario['name']}")
        print("-" * 40)
        
        # Simulate with different parameters
        test_rate, control_rate = experiment.simulate_campaign_results(
            base_pickup_rate=scenario['base_rate'],
            treatment_effect=scenario['treatment_effect']
        )
        
        # Analyze results
        results = experiment.analyze_results()
        
        # Store summary
        results_summary.append({
            'Scenario': scenario['name'],
            'Base_Rate': scenario['base_rate'],
            'Expected_Effect': scenario['treatment_effect'],
            'Observed_Effect': results['treatment_effect'],
            'P_Value': results['t_p_value'],
            'Significant': results['statistically_significant'],
            'Effect_Size': results['cohens_d']
        })
    
    # Create summary table
    print("\n" + "=" * 80)
    print("SUMMARY OF ALL SCENARIOS")
    print("=" * 80)
    
    summary_df = pd.DataFrame(results_summary)
    print(summary_df.to_string(index=False, float_format='%.4f'))
    
    print(f"\n📈 KEY INSIGHTS:")
    print(f"• Larger treatment effects are more likely to be statistically significant")
    print(f"• Effect size (Cohen's d) increases with treatment effect magnitude")
    print(f"• Sample size of {len(experiment.test_group)} per group provides reasonable power for moderate effects")
    
    return summary_df

def demonstrate_custom_stratification():
    """Demonstrate custom stratification options"""
    
    print("\n" + "=" * 60)
    print("CUSTOM STRATIFICATION DEMO")
    print("=" * 60)
    
    experiment = TestControlExperiment('data/test_learn_campaign_dataset.xlsx')
    
    # Test different stratification approaches
    stratification_options = [
        ['state', 'age_group'],
        ['state', 'insurance_type'],
        ['age_group', 'insurance_type', 'gender'],
        ['state', 'age_group', 'insurance_type', 'purpose_of_call']
    ]
    
    for i, strata_vars in enumerate(stratification_options, 1):
        print(f"\n🔧 Stratification Option {i}: {strata_vars}")
        print("-" * 50)
        
        # Reset data for each test
        experiment.data = experiment.original_data.copy()
        
        # Create stratification
        strata_counts = experiment.create_stratification(strata_vars)
        print(f"Total strata: {len(strata_counts)}")
        print(f"Remaining observations: {len(experiment.data)}")
        print(f"Average stratum size: {len(experiment.data) / len(strata_counts):.1f}")

def show_balance_analysis():
    """Demonstrate detailed balance analysis"""
    
    print("\n" + "=" * 60)
    print("COVARIATE BALANCE ANALYSIS")
    print("=" * 60)
    
    experiment = TestControlExperiment('data/test_learn_campaign_dataset.xlsx')
    
    # Run experiment setup
    experiment.create_stratification()
    experiment.calculate_propensity_scores()
    experiment.perform_matching()
    experiment.assign_treatment_groups()
    
    # Get detailed balance results
    balance_df = experiment.check_balance()
    
    print("\n📊 DETAILED BALANCE ANALYSIS:")
    print("-" * 40)
    
    categorical_vars = balance_df[balance_df['test_type'] == 'Chi-square']
    numerical_vars = balance_df[balance_df['test_type'] == 'T-test']
    
    print(f"\nCategorical Variables Balance:")
    for _, row in categorical_vars.iterrows():
        status = "✅ Balanced" if row['balanced'] else "❌ Imbalanced"
        print(f"  {row['variable']}: {status} (p={row['p_value']:.4f})")
    
    print(f"\nNumerical Variables Balance:")
    for _, row in numerical_vars.iterrows():
        status = "✅ Balanced" if row['balanced'] else "❌ Imbalanced"
        diff = abs(row['test_mean'] - row['control_mean'])
        print(f"  {row['variable']}: {status} (p={row['p_value']:.4f}, diff={diff:.3f})")

if __name__ == "__main__":
    # Run all demonstrations
    summary_results = run_demo()
    demonstrate_custom_stratification()
    show_balance_analysis()
    
    print("\n" + "=" * 60)
    print("DEMO COMPLETED SUCCESSFULLY! 🎉")
    print("=" * 60)
    print("\nFiles created:")
    print("• test_control_experiment.py - Main experiment class")
    print("• Test_Control_Campaign_Summary.md - Detailed methodology")
    print("• campaign_analysis_results.png - Visualization")
    print("• demo_experiment.py - This demo script")
    
    print(f"\nTo run individual experiments:")
    print(f"  python test_control_experiment.py")
    print(f"  python demo_experiment.py")
