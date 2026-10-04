"""Leakage-controlled churn ranking with validation-selected model and budget policy."""
from reporting import export_sql_results
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss, precision_recall_curve
from sklearn.calibration import calibration_curve
from sklearn.inspection import permutation_importance
from reporting import ROOT, fetch, manifest, save_json, report, preview

NUMERIC=['tenure','MonthlyCharges','TotalCharges']
# Exclude IDs and sensitive demographic attributes from the intervention model.
CATEGORICAL=['Contract','PaymentMethod','InternetService','OnlineSecurity','OnlineBackup','DeviceProtection','TechSupport','StreamingTV','StreamingMovies','PaperlessBilling','PhoneService','MultipleLines']

def preprocess():
    return ColumnTransformer([
        ('numeric',Pipeline([('impute',SimpleImputer(strategy='median')),('scale',StandardScaler())]),NUMERIC),
        ('category',Pipeline([('impute',SimpleImputer(strategy='most_frequent')),('encode',OneHotEncoder(handle_unknown='ignore',sparse_output=False))]),CATEGORICAL)])

def top_indices(probabilities, fraction):
    if not 0 < fraction <= 1: raise ValueError('fraction must be in (0, 1]')
    k=max(1,int(np.ceil(len(probabilities)*fraction)))
    return np.argsort(-np.asarray(probabilities),kind='stable')[:k]

def evaluate(y,p):
    return {'roc_auc':float(roc_auc_score(y,p)),'average_precision':float(average_precision_score(y,p)),'brier':float(brier_score_loss(y,p))}

def main():
    source=fetch('https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv','telco.csv')
    raw=pd.read_csv(source)
    assert raw.customerID.is_unique, 'Duplicate customer IDs'
    assert raw.Churn.isin(['Yes','No']).all()
    raw['TotalCharges']=pd.to_numeric(raw.TotalCharges,errors='coerce')
    x=raw[NUMERIC+CATEGORICAL]; y=raw.Churn.eq('Yes').astype(int)
    train,remaining=train_test_split(np.arange(len(raw)),test_size=.4,stratify=y,random_state=42)
    validation,test=train_test_split(remaining,test_size=.5,stratify=y.iloc[remaining],random_state=42)
    assert not (set(train)&set(validation) or set(train)&set(test) or set(validation)&set(test))
    candidates={'logistic':LogisticRegression(C=1,max_iter=1500,random_state=42),'random_forest':RandomForestClassifier(n_estimators=220,min_samples_leaf=15,max_features=.7,n_jobs=-1,random_state=42)}
    fitted={}; results=[]
    for name,estimator in candidates.items():
        model=Pipeline([('prepare',preprocess()),('model',estimator)])
        model.fit(x.iloc[train],y.iloc[train])
        p=model.predict_proba(x.iloc[validation])[:,1]
        results.append({'model':name,**evaluate(y.iloc[validation],p)})
        fitted[name]=model
    validation_scores=pd.DataFrame(results)
    selected=validation_scores.sort_values('average_precision',ascending=False).iloc[0].model
    model=fitted[selected]
    # Test is touched only after selecting on validation AP. Do not tune to this result.
    truth=y.iloc[test].to_numpy(); p=model.predict_proba(x.iloc[test])[:,1]
    metrics=evaluate(truth,p)
    rank=top_indices(p,.2)
    metrics.update({'top_20pct_precision':float(truth[rank].mean()),'top_20pct_recall':float(truth[rank].sum()/truth.sum()),'test_prevalence':float(truth.mean()),'top_20pct_lift':float(truth[rank].mean()/truth.mean())})
    rng=np.random.default_rng(2026)
    boot=[]
    for _ in range(500):
        ix=rng.integers(0,len(truth),len(truth))
        boot.append(average_precision_score(truth[ix],p[ix]))
    metrics['ap_bootstrap_95ci']=np.quantile(boot,[.025,.975]).tolist()
    precision,recall,_=precision_recall_curve(truth,p)
    actual,predicted=calibration_curve(truth,p,n_bins=10,strategy='quantile')
    importance=permutation_importance(model,x.iloc[test],truth,n_repeats=8,scoring='average_precision',random_state=42,n_jobs=-1)
    features=pd.DataFrame({'feature':x.columns,'ap_decrease':importance.importances_mean,'std':importance.importances_std}).sort_values('ap_decrease',ascending=False)
    policies=[]
    for budget in [.05,.1,.2,.3,.4,.5]:
        ranked=top_indices(p,budget)
        # Assumptions: hypothetical $100 retained contribution, $5 contact cost.
        for success in [.1,.2,.3,.4]:
            policies.append({'contact_fraction':budget,'assumed_save_probability_given_churn':success,'contacts':len(ranked),'expected_churners':float(p[ranked].sum()),'scenario_net_value_usd':float(p[ranked].sum()*success*100-len(ranked)*5)})
    policy=pd.DataFrame(policies)
    validation_scores.to_csv(ROOT/'validation_models.csv',index=False)
    features.to_csv(ROOT/'permutation_importance.csv',index=False)
    policy.to_csv(ROOT/'policy_scenarios.csv',index=False)
    # Only anonymous scores, never source customer identifiers.
    pd.DataFrame({'actual_churn':truth,'predicted_risk':p}).to_csv(ROOT/'test_predictions.csv',index=False)
    export_sql_results(pd.DataFrame({'actual_churn':truth,'predicted_risk':p}), 'predictions')
    save_json('metrics.json',{'selected_model':selected,'selection_metric':'validation average precision','train_n':len(train),'validation_n':len(validation),'test_n':len(test),'test_metrics':metrics,'missing_total_charges':int(raw.TotalCharges.isna().sum()),'source_sha256':manifest([source])})
    pr=go.Figure([go.Scatter(x=recall,y=precision,name=selected),go.Scatter(x=[0,1],y=[truth.mean()]*2,name='Prevalence baseline',line={'dash':'dash'})]);pr.update_layout(title='Held-out precision–recall',xaxis_title='Recall',yaxis_title='Precision')
    cal=go.Figure([go.Scatter(x=predicted,y=actual,mode='lines+markers',name='Observed'),go.Scatter(x=[0,1],y=[0,1],name='Perfect calibration',line={'dash':'dash'})]);cal.update_layout(title='Risk calibration: test quantile bins',xaxis_title='Mean predicted risk',yaxis_title='Observed churn fraction')
    figs=[pr,cal,px.bar(features.head(10),x='ap_decrease',y='feature',orientation='h',error_x='std',title='Permutation importance: held-out AP decrease'),px.line(policy,x='contact_fraction',y='scenario_net_value_usd',color='assumed_save_probability_given_churn',markers=True,title='Retention policy sensitivity: hypothetical economics')]
    notes={'Decision':f"The validation-selected {selected} model reaches test AP {metrics['average_precision']:.3f}. Contacting the highest-risk 20% captures {metrics['top_20pct_recall']:.1%} of observed churners at {metrics['top_20pct_lift']:.2f}× prevalence. This supports prioritization research, not a proven retention intervention.",'Economics':'Scenario net value = sum(predicted churn risk) × assumed save probability × $100 contribution − contacts × $5. These inputs are hypothetical. Predicting churn does not identify customers whose behavior can be changed. A randomized retention pilot is required.','Validation':'Stratified 60/20/20 split; preprocessing fit only on training; model selected by validation AP; bootstrap intervals conditional on the fixed test predictions. The test set is not used to choose the budget or model.','Limitations':'IBM sample data has no reliable observation/label horizon for prospective deployment. Random splits do not measure future drift. Calibration is assessed, not guaranteed. Permutation importance is associational and unstable among correlated services. Demographics are excluded; this does not establish fairness.'}
    report('Churn decision engine','From held-out risk ranking to a transparent retention budget sensitivity analysis.',{'Customers':f'{len(raw):,}','Test average precision':f"{metrics['average_precision']:.3f}",'Top-20% lift':f"{metrics['top_20pct_lift']:.2f}×",'Test ROC AUC':f"{metrics['roc_auc']:.3f}"},figs,notes)
    preview('Validation model comparison',validation_scores.model,validation_scores.average_precision,'Average precision')
    (ROOT/'FINDINGS.md').write_text('# Results\n\n'+'\n\n'.join(f'## {k}\n\n{v}' for k,v in notes.items())+'\n')
    print(selected,metrics)

if __name__=='__main__': main()
