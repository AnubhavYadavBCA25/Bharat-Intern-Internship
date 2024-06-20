import streamlit as st
import joblib
from sklearn.preprocessing import StandardScaler

def main():
    # Load the trained model
    model = joblib.load(open("Task 2 (Titanic Survival Prediction)/artifact/model.pkl", "rb"))
    
    # Header
    st.header("🚢 Titanic Survival Prediction", divider='rainbow')
    st.write("Welcome to the Titanic Survival Prediction App! Enter the details of a passenger to predict if they would have survived the Titanic disaster.")
    st.divider()

    # Get user input
    st.write("Please enter the following details:")

    pclass = st.radio("Enter the Passenger Class:", [1, 2, 3])

    gender = st.radio("Enter the Passenger Gender:", ["Male", "Female"])

    age = st.number_input("Enter the Passenger Age:", min_value=0, max_value=100, value=30)

    sibsp = st.number_input("Number of Siblings/Spouses Aboard", min_value=0, max_value=10, value=0)

    parch = st.number_input("Number of Parents/Children Aboard", min_value=0, max_value=10, value=0)

    fare = st.number_input("Enter the Fare Passenger Paid:", min_value=0.0, max_value=1000.0, value=50.0)

    embarked = st.radio("Enter the Port of Embarkation:", ["Cherbourg", "Queenstown", "Southampton"])
    st.write("*Note*: Port of Embarkation means the port where the passenger boarded the Titanic.")

    if st.button("Predict"):
        
        # Convert categorical variables to numerical
        # Gender conversion
        if gender == 'Male':
            gender = 1
        else:
            gender = 0

        # Embarked conversion
        if embarked == 'Cherbourg':
            embarked = 0
        elif embarked == 'Queenstown':
            embarked = 1
        else:
            embarked = 2

        # feature scaling
        sc = StandardScaler()
        features = sc.fit_transform([[pclass, gender, age, sibsp, parch, fare, embarked]])
        
        # Make prediction
        prediction = model.predict(features)

        # Display the prediction
        if prediction[0] == 0:
            st.error("The passenger would not have survived the Titanic disaster.")
        else:
            st.success("The passenger would have survived the Titanic disaster.")

        st.divider()

    # Footer
    st.markdown("---")
    st.markdown(
    """
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/5.15.4/css/all.min.css">
    <div style="text-align:center;">
        <p>Made with ❤️ by Anubhav Yadav</p>
        <p>Follow me on 
            <a href="https://linkedin.com/in/anubhav-yadav-data-science" target="_blank"><i class="fab fa-linkedin"></i>LinkedIn</a> | 
            <a href="https://github.com/AnubhavYadavBCA25" target="_blank"><i class="fab fa-github"></i>GitHub</a>
        </p>
    </div>
    """, unsafe_allow_html=True
)

if __name__ == "__main__":
    main()