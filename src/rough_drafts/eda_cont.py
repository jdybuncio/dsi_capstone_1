import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from patsy import dmatrices
import statsmodels.api as sm
%matplotlib inline




port_predictors = ['age', 'absences', 'school', 'sex', 'address',
       'famsize', 'Pstatus', 'romantic', 'schoolsup', 'famsup', 'paid', 'activities',
       'nursery', 'higher', 'internet', 'Medu_1', 'Medu_2',
       'Medu_3', 'Medu_4', 'Fedu_1', 'Fedu_2', 'Fedu_3', 'Fedu_4',
       'Mjob_health', 'Mjob_other', 'Mjob_services', 'Mjob_teacher',
       'Fjob_health', 'Fjob_other', 'Fjob_services', 'Fjob_teacher',
       'reason_home', 'reason_other', 'reason_reputation', 'guardian_mother',
       'guardian_other', 'traveltime_2', 'traveltime_3', 'traveltime_4',
       'studytime_2', 'studytime_3', 'studytime_4', 'failures_1', 'failures_2',
       'failures_3', 'famrel_2', 'famrel_3', 'famrel_4', 'famrel_5',
       'freetime_2', 'freetime_3', 'freetime_4', 'freetime_5', 'goout_2',
       'goout_3', 'goout_4', 'goout_5', 'Dalc_2', 'Dalc_3', 'Dalc_4', 'Dalc_5',
       'Walc_2', 'Walc_3', 'Walc_4', 'Walc_5', 'health_2', 'health_3',
       'health_4', 'health_5']

def run_multiple_regr(df, target_lst, predictors_lst):
    all_sign_coeffs = dict()

    for target in target_lst:
        X = df[predictors_lst]
        y = df[target]

        X = sm.add_constant(X)
        model = sm.OLS(y, X)
        results = model.fit()

        fitted_vals = results.predict(X)
        stu_resid = results.resid_pearson
        residuals = results.resid
        y_vals = pd.DataFrame({'residuals':residuals, 'fitted_vals':fitted_vals, 'stu_resid': stu_resid})
        print(results.summary())

        significant_vars = results.pvalues[results.pvalues <= 0.05]
        signif_coeffs = significant_vars.index

        for coeff in signif_coeffs:
            if coeff in all_sign_coeffs.keys():
                all_sign_coeffs[coeff] += target
            else:
                all_sign_coeffs[coeff] = target

    return all_sign_coeffs




    print(g1_on_all_significant_coeffs.sort_values())
    print(g2_on_all_significant_coeffs.sort_values())
    print(g3_on_all_significant_coeffs.sort_values())

    all_sig_coeffs = pd.Series(all_sig_coeffs)
    all_sig_coeffs.value_counts()
