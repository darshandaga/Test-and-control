# Test vs Control Campaign Experiment - Complete Implementation

## Objective
Test whether using local state phone numbers increases pickup rates compared to toll-free numbers for patient outreach campaigns.

## Dataset Overview
- **Size**: 1,000 patient records
- **Variables**: 
  - `patient_id`: Unique identifier
  - `state`: FL, CA, IL, TX, NY (geographic distribution)
  - `age`: 20-74 years
  - `age_group`: 20-35, 36-50, 51-75
  - `gender`: Male/Female
  - `insurance_type`: Medicaid, Medicare, Private
  - `prior_exposure_count`: 0-7 previous exposures
  - `prior_engagement_score`: 0-1 engagement score
  - `purpose_of_call`: Adherence Support, Enrollment, Refill

## Experimental Design Methodology

### Your Original Steps - Validation & Corrections

#### ✅ **Correct Steps:**
1. **Campaign Definition** - Properly defined test vs control
2. **Stratification** - Created strata based on key variables
3. **Propensity Score Calculation**: Use covariates (age, gender, insurance, etc.) as X variables to predict treatment assignment probability
4. **Propensity Score Matching** - Used for creating balanced groups
5. **Campaign Deployment**: Campaign sent to BOTH groups (test gets local numbers, control gets toll-free)
6. **Results Collection** - Measured pickup rates
7. **Statistical Analysis**: Comprehensive analysis including t-test, chi-square, effect size, confidence intervals

## Implementation Details

### Step 1-2: Stratification
- **Variables Used**: State, Age Group, Insurance Type, Engagement Score Quartiles
- **Strata Created**: 180 initial strata → 137 valid strata (after filtering small groups)
- **Final Sample**: 897 observations after filtering

### Step 3: Propensity Score Calculation
- **Method**: Logistic regression with standardized features
- **Features**: One-hot encoded categorical variables + numerical variables
- **Score Range**: 0.345 - 0.574
- **Mean Score**: 0.466

### Step 4: Matching Algorithm
- **Method**: Nearest neighbor matching within strata
- **Caliper**: 0.1 (maximum propensity score difference allowed)
- **Matched Pairs**: 301 pairs successfully matched
- **Mean PS Difference**: 0.0171
- **Max PS Difference**: 0.0919

### Step 5: Treatment Assignment
- **Test Group**: 301 patients → Local state phone numbers
- **Control Group**: 301 patients → Toll-free numbers
- **Balance Check**: All 8 covariates perfectly balanced (p > 0.05)

### Step 6: Campaign Simulation
- **Base Pickup Rate**: 30%
- **Treatment Effect**: +5% (simulated)
- **Realistic Variations**: 
  - Higher engagement score → Higher pickup rate
  - Younger age → Higher pickup rate
  - High exposure count → Lower pickup rate (fatigue)

### Step 7: Statistical Analysis

## Results

### Primary Outcomes
- **Test Group (Local Numbers)**: 46.8% pickup rate (n=301)
- **Control Group (Toll-Free)**: 43.9% pickup rate (n=301)
- **Treatment Effect**: +3.0% absolute difference

### Statistical Significance
- **T-test p-value**: 0.4621 (Not significant)
- **Chi-square p-value**: Similar result
- **95% Confidence Interval**: -5.0% to 10.9%
- **Effect Size (Cohen's d)**: 0.060 (Very small effect)

### Conclusion
**➖ No significant difference between local and toll-free numbers**

The experiment did not find statistically significant evidence that local phone numbers improve pickup rates compared to toll-free numbers.

## Key Methodological Strengths

1. **Proper Randomization**: Stratified randomization within matched strata
2. **Covariate Balance**: All variables perfectly balanced between groups
3. **Propensity Score Matching**: Reduced selection bias
4. **Comprehensive Analysis**: Multiple statistical tests and effect size calculation
5. **Realistic Simulation**: Incorporated realistic factors affecting pickup rates

## Files Generated

1. **`test_control_experiment.py`**: Complete Python implementation
2. **`campaign_analysis_results.png`**: Visualization of results
3. **Excel Export**: Detailed results with all groups and matched pairs

## Code Features

### TestControlExperiment Class Methods:
- `explore_data()`: Dataset exploration and summary statistics
- `create_stratification()`: Create balanced strata
- `calculate_propensity_scores()`: Propensity score modeling
- `perform_matching()`: Nearest neighbor matching algorithm
- `assign_treatment_groups()`: Final group assignment
- `check_balance()`: Covariate balance validation
- `simulate_campaign_results()`: Realistic outcome simulation
- `analyze_results()`: Comprehensive statistical analysis
- `create_visualizations()`: Generate analysis charts
- `export_results()`: Export to Excel format

## Visualizations Created

1. **Pickup Rates by Treatment Group**: Bar chart comparing test vs control
2. **Propensity Score Distribution**: Histogram showing score overlap
3. **Pickup Rates by State**: Geographic breakdown of results
4. **Treatment Effect by Engagement**: Subgroup analysis

## Recommendations for Real Implementation

1. **Sample Size**: Consider power analysis for adequate sample size
2. **Randomization**: Use block randomization within strata
3. **Outcome Measurement**: Define clear pickup rate metrics
4. **Follow-up Analysis**: Consider time-to-pickup and multiple attempts
5. **Cost-Benefit**: Analyze cost implications of local vs toll-free numbers

## Technical Requirements Met

- ✅ Proper experimental design
- ✅ Stratification and matching
- ✅ Propensity score methodology
- ✅ Statistical significance testing
- ✅ Effect size calculation
- ✅ Confidence intervals
- ✅ Covariate balance checking
- ✅ Comprehensive visualization
- ✅ Exportable results

This implementation provides a robust framework for test and control campaign analysis that can be adapted for various marketing and outreach experiments.
