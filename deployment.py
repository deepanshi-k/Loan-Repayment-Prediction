import streamlit as st
import pandas as pd
import pickle
import numpy as np
from sklearn.preprocessing import LabelEncoder

pickle_in = open("model.pkl","rb")
model = pickle.load(pickle_in)

data = open(r"C:\Users\deepa\OneDrive\Desktop\Project sem 4\loan_data.csv")
data = pd.read_csv(data)

data = data.drop(['days.with.cr.line', 'revol.bal', 'revol.util', 'delinq.2yrs', 'pub.rec','credit.policy'], axis = 1)

encoder = LabelEncoder()
encoder.classes_ = np.load('model.pkl',allow_pickle=True)

def main():
    st.title("Loan Repayment Prediction Model")
    html_temp = """
        <div style="background-color:tomato;padding:10px">
        <h2 style="color:white;text-align:center;">Loan Repayment Prediction ML App </h2>
        </div>
        """
    st.text_input("Enter your Name: ", key="name")

    if st.checkbox('Show Training Dataframe'):
        data

    st.subheader("Please select relevant purpose!")
    left_column, right_column = st.columns(2)
    with left_column:
        inp_purpose = st.radio(
            'Select the purpose:',
            np.unique(data['purpose']))

    if inp_purpose == "debt_consolidation":
        inp_purpose = 0
    elif inp_purpose == "credit_card":
        inp_purpose = 1
    elif inp_purpose == "all_other":
        inp_purpose = 2
    elif inp_purpose == "home_improvement":
        inp_purpose = 3
    elif inp_purpose == "small_business":
        inp_purpose = 4
    elif inp_purpose == "major_purchase":
        inp_purpose = 5
    elif inp_purpose == "educational":
        inp_purpose = 6

    Interest_Rate = st.slider('Interest Rate', 0.0, 1.0, 0.16)
    Installment = st.slider('Installment', 0.0, 945.0, 351.42)
    Annual_Income = st.slider('Annual Income', 0.0, 20.0, 10.79)
    Debt_to_income_ratio = st.slider('Debt to income ratio', 0.0, 30.0, 19.17)
    Fico_Score = st.slider('Fico score', 0.0, 900.0, 691.36)
    In_Last_six_months = st.slider('Bank transactions done in last 6 months', 0, 35, 5)

    if st.button('Make Prediction'):
        result = model.predict([[inp_purpose, Interest_Rate, Installment, Annual_Income, Debt_to_income_ratio, Fico_Score, In_Last_six_months]])
        if result == 0:
            st.write("Repayment difficult..")
        if result == 1:
            st.write("Repayment possible...")
        st.write(f"Thank you {st.session_state.name}! I hope you liked it.")


if __name__ == '__main__':
    main()