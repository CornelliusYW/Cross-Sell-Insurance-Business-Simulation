import streamlit as st
import pandas as pd
import plotly.express as px
import random



#Business Simulation Function
def business_target_simulation(S, L, M, C, A, T, BT):
    revenue = (S*(L * M)) - (L * (C * (A/T)))
    if revenue < BT:
        return True
    else:
        return False

p = 0.01
st.title('Insurance Cross-Selling Business Simulation')
url = 'https://raw.githubusercontent.com/cornelliusyudhawijaya/Cross-Sell-Insurance-Business-Simulation/main/train.csv'
train = pd.read_csv(url, index_col=0, skiprows=lambda i: i>0 and random.random() > p)


# Change to Categorical data
train['Region_Code'] = train['Region_Code'].astype('string')
train['Policy_Sales_Channel'] = train['Policy_Sales_Channel'].astype('string')
train['Driving_License'] = train['Driving_License'].astype('string')
train['Previously_Insured'] = train['Previously_Insured'].astype('string')

st.subheader('100 Sample Data')
st.dataframe(train.sample(100))

st.subheader('Data Visualization with respect to Response')

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
    if business_target_simulation(S=S, L=L, M=M/100, C=C, A=A, T=T, BT=BT):
        st.header("Business Target are met")
    else:
        st.header("Business Target are not Met")