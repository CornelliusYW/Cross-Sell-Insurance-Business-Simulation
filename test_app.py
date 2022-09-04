import streamlit as st
import pandas as pd
import plotly.express as px
import random
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

#my own function for effect size
import scipy.stats as ss
def correlation_ratio(categories, measurements):
    fcat, _ = pd.factorize(categories)
    cat_num = np.max(fcat)+1
    y_avg_array = np.zeros(cat_num)
    n_array = np.zeros(cat_num)
    for i in range(0,cat_num):
        cat_measures = measurements[np.argwhere(fcat == i).flatten()]
        n_array[i] = len(cat_measures)
        y_avg_array[i] = np.average(cat_measures)
    y_total_avg = np.sum(np.multiply(y_avg_array,n_array))/np.sum(n_array)
    numerator = np.sum(np.multiply(n_array,np.power(np.subtract(y_avg_array,y_total_avg),2)))
    denominator = np.sum(np.power(np.subtract(measurements,y_total_avg),2))
    if numerator == 0:
        eta = 0.0
    else:
        eta = np.sqrt(numerator/denominator)
    return eta


def cramers_v(x, y):
    confusion_matrix = pd.crosstab(x,y)
    chi2 = ss.chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.sum().sum()
    phi2 = chi2/n
    r,k = confusion_matrix.shape
    phi2corr = max(0, phi2-((k-1)*(r-1))/(n-1))
    rcorr = r-((r-1)**2)/(n-1)
    kcorr = k-((k-1)**2)/(n-1)
    return np.sqrt(phi2corr/min((kcorr-1),(rcorr-1)))

def pairPlot(df):
    fig = sns.pairplot(df, hue = 'Response')
    st.pyplot(fig)


#Business Simulation Function
def business_target_simulation(S, L, M, C, A, T, BT):
    revenue = (S*(L * M)) - (L * (C * (A/T)))
    if revenue > BT:
        return True
    else:
        return False

def simulation_plot(metric_range, S, L, C, A, T, BT):
    result = []
    for score in metric_range:
        result.append(business_target_simulation(S=S, L=L, M = score, C=C, A=A, T=T, BT=BT))

    img = plt.plot(metric_range, result, marker = 'o')
    st.pyplot(img)



st.title('Insurance Cross-Selling Business Simulation and Model Prediction')

"""
Our client is an Insurance company that has provided Health Insurance to its customers now they need your help in building a model to predict whether the policyholders (customers) 
from past year will also be interested in Vehicle Insurance provided by the company.
My target here is to explore the data in a way that business would run and give a suggestion based on the finding. 
I am not try to achieve the best model, but I try to develop model that viable and useful in the long run. 

In this data exploration and machine learning development, I also want to explore the possibility whether we need machine learning model to provide leads or random sampling are enough. 
With simulation we would assess that.

I am assumed the customers in question is an insurance customer with Health insurance from the company and the policy is still In-force. 
The data provided is in customer (policy holder) level and the agent would offer the new policy to the policy holder (whether it is for the insured or the family). 

The data is retrieved from the Kaggle website: https://www.kaggle.com/anmolkumar/health-insurance-cross-sell-prediction

Let's start with take a look at the dataset and data visualization.
"""


# url = 'https://raw.githubusercontent.com/cornelliusyudhawijaya/Cross-Sell-Insurance-Business-Simulation/main/train.csv'
train = pd.read_csv('sample.csv')
train = train.reset_index(drop = True)

# Change to Categorical data
train['Region_Code'] = train['Region_Code'].astype('string')
train['Policy_Sales_Channel'] = train['Policy_Sales_Channel'].astype('string')
train['Driving_License'] = train['Driving_License'].astype('string')
train['Previously_Insured'] = train['Previously_Insured'].astype('string')

st.subheader('100 Sample Data')
st.dataframe(train.sample(100))

st.subheader('Data Numerical Statistic (1% Samples)')
st.dataframe(train.describe())

st.subheader('Data Numerical Pair Plot')
pairPlot(train)

st.subheader('Data Visualization with respect to Response (1% Samples)')

left_column, right_column = st.columns(2)
with left_column:
    'Numerical Plot'
    num_feat = st.selectbox(
     'Select Numerical Feature',
     train.select_dtypes('number').columns)
    fig = px.histogram(train, x = num_feat, color = 'Response')
    # Plot!
    st.plotly_chart(fig, use_container_width=True)

with right_column:
    'Categorical column'
    cat_feat = st.selectbox(
     'Select Categorical Feature',
     train.select_dtypes(exclude = 'number').columns)
    fig = px.histogram(train, x =cat_feat, color = 'Response' )
    # Plot!
    st.plotly_chart(fig, use_container_width=True)


st.subheader('Effect Size and Hypothesis Testing')

"""
To know the prediction strength from the independent feature to the target, 
I would employ various statistical calculation and hypothesis testing
"""

st.subheader('Numerical Column Effect Size with Correlation Ratio')
corr_ratio = []
for i in train.select_dtypes(include = 'number').columns:
    corr_ratio.append(correlation_ratio(train['Response'], train[i]))
res = pd.DataFrame(data = [ train.select_dtypes(include = 'number').columns, corr_ratio]
             ).T
res.columns = ['Column', 'Correlation Ratio']
res.sort_values(by = 'Correlation Ratio', ascending = False).reset_index(drop = True)
st.dataframe(res)

st.subheader("Categorical Column Effect Size with Cramer's V")
cramers = []
for i in train.select_dtypes(exclude = 'number').columns:
    cramers.append(cramers_v(train['Response'], train[i]))
res = pd.DataFrame(data = [train.select_dtypes(exclude = 'number').columns, cramers]
             ).T
res.columns = ['Column', 'Cramers_V']
res.sort_values(by = 'Cramers_V', ascending = False).reset_index(drop = True)
st.dataframe(res)

st.subheader("Insight from Initial Exploration")
"""
From the data exploration above, here are some insight I found:
1. The Response feature (Target) are imbalance, only ~12% customers are interested for Cross-Selling. I am not sure the customer interested for cross-selling here, is it successfully converted or only successfully contacted (interest for calling). In this project, I would assume the interested customer would always do Cross-Selling.

2. Driving License are not useful because almost all customer have a driving license and it would be bad for business to insured someone who did not own Driving License. This feature could become a rule-based decision where we would not offer vehicle insurance to someone without Driving License

3. From the data, all the customer vintage were under a year and follow the uniform distribution even when separated by the dependent. Vintage feature could be useful, but mostly if we have data more than a year as comparison. In our case, vintage are not useful to predict the customer interest.

4. Previously Insured and Vehicle Damage is the most useful feature to predict the Response feature, basically all the customer with no vehicle insurance and have damage their vehicle in the past would interested in the Vehicle Insurance from this company.

5. Many of the customer vehicle age are less than 1 year or between 1 to 2 years,however the customers with 1 to 2 years vehicle age might interested to buy the vehicle insurance
6. Older customer might have slightly better chance for cross-selling.

From the analysis, I would only consider using 6 features for Machine Learning development --> Vehicle Damage, Previously Insured, Vehicle Age, Gender, and High ANP.

"""

st.subheader('Business Simulation')
"""This Business Simulation concerns 
with how the Precision would meet the Business Target or Not"""

"We would use the following equation for simulate the business requirements"
st.latex(r'BT <= (S * (L * M)) - (L*(C * (A/T))')
"""Where BT = Business Target, S = Success Income, L = Number of Leads,
M = Model Metrics, C = Cost per Call, A = Number of Agent, T = Timeline"""

'Fill up your number here (Currency on Indian Rupee)'

one_col, two_col, three_col, four_col = st.columns(4)
with one_col:
    BT = st.number_input('Business Target')
with two_col:
    S = st.number_input('Success Income')
with three_col:
    L = st.number_input('Number of Leads')
with four_col:
    M = st.number_input('Model Metrics')

five_col, six_col, seven_col = st.columns(3)

with five_col:
    C = st.number_input('Cost per Call')
with six_col:
    A= st.number_input('Number of Agents')
with seven_col:
    T = st.number_input('Timeline')

if st.button('Simulate the Business Requirements'):
    if business_target_simulation(S=S, L=L, M=M / 100, C=C, A=A, T=T, BT=BT):
        st.header("Business Target are met")
    else:
        st.header("Business Target are not Met")

st.subheader('Plotting Business Simulation Experiment')
"""
Let's define some of the variable, we have:

1. Business Target; Let's say this year we want to achieve 10.000.000 Indian Rupee from Cross-Selling program
2. Timeline would be a year, which means 12 Months
3. Cost per call let's say 5000 Indian Rupee
4. Agent number let's say 100
5. Number of Leads would related to the existing population because it is Cross-Selling program. The test data contain 127037 people. From the train data, we see around 12% population are interested for sales. If we take 12% of the rest data as leads (assumed the prediction model might follow the same probability distribution) then we have 15244 customer as leads.

For model metric, let's assume we use Precision because we need to calculate the True Positive cases from the predicted interest to buy. Because the business process would only call the leads predicted interested, then we need to maximize the precision of class 1 (interested). Why? because if the False Positive too high (low precision) means there are too many call make without successull income; sans, more cost. Our cost are associated with the Positive cases (Either True or False). We want to define our technical KPI as well, so let's simulate various Precision number.
"""

st.image('simulation_plot.png')

st.subheader('Simulation Result')
"""

Our business target is 10.000.000 Indian Rupee and it seems according to our simulation we need to achieve at the very least 82% in Precision for our model to have an impact on the business. The number is quite high, but certainly possible. With this in mind, when we set up our model and MLOps, we need to make sure that the precision is always higher than 82%. You could always change the metric to another such as Accuracy, Recall, etc. however we need to tweak the equation as well if we did that.
Of course, this simulation is based on the linear assumption and hasn't considered the randomness factor and error. We could try to play around with the confidence interval, but let's keep that for a later discussion.
You could play around with the code and the variable to have a better feeling at simulating the business requirements. I suggest you tweak around the linear function as well because I know it isn't perfect. I am open to suggestions for this function.
What I want to show is the power of simulation capable to set up the technical requirements for the business. With 82% Precision, we could achieve the business target, less than that is not desirable.
"""

st.subheader('Machine Learning Development')

"""
From our simulation, we need to achieve 82% Precision in order to have a model that would meet the business target. We would experimenting with various methods and model available to reach the target. 
Let's start with data preprocessing and preparation.

In this training development, I would use the following features:
'Gender',
'Age', 
'Previously_Insured', 
'Vehicle_Age', 
'Vehicle_Damage', 
'High_ANP', 

With the target 'Response'

For categorical encoding strategy, I would use the One-Hot Encoding as it is the simplest one and the amount of transformed features would not lead to curse of dimensionality.

Lastly, I would split the dataset to train-test dataset with 8:2 ratio.
"""

st.subheader('Machine Learning Benchmarking')

"""
What is Machine Learning Benchmarking? It is a process to create a machine learning model as a standard.
 The benchmark model usually a naive predictor and not learning from any pattern; such as, predicting randomly or predicting same label all the time. I have mention previsouly, do we even need machine learning model for our cases? If the benchmark already did great, then we might not need machine learning model at all.
"""

bn_result  = pd.DataFrame([['Constant Model', 0.123],
             ['Uniform Model', 0.124],
             ['Stratified Strategy', 0.125],
             ['Prior Strategy', 0]], columns  = ['Benchmark Strategy', 'Precision'])
st.dataframe(bn_result)