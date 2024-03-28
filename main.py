import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from streamlit_option_menu import option_menu
from PIL import Image
import pickle

# We are going to set the page layout
st.set_page_config(page_title='Machine Learning App with Random Forest',layout='wide')
# Title
st.title('Used Motorcycle Prices Prediction Using Random Forest Regression')
#Load Dataset
global df
df = pd.read_excel("D:/Source Code/Finpro/App/Motorcycle Clean-Data.xlsx") # read a CSV file 

with st.sidebar:
    choose = option_menu("Motorcycle Prediction", ["Home","Data Training", "EDA", "Prediction"],
                         icons=['book', 'clipboard-data', 'graph-up-arrow','box-arrow-in-down'],
                         menu_icon="pepicons:motorcycle", default_index=0,
                         styles={
        "container": {"padding": "5!important", "background-color": "#fafafa"},
        "icon": {"color": "orange", "font-size": "25px"}, 
        "nav-link": {"font-size": "16px", "text-align": "left", "margin":"0px", "--hover-color": "#eee"},
        "nav-link-selected": {"background-color": "#02ab21"},
    }
    )

if choose == "Home":
    st.write('This application will help you predict the selling price of the used motorcycle that you have. We have made the machine know the pattern data from the dataset that we have so that it can help in predicting the selling price with good accuracy.')
    st.subheader('**Dataset Information**')
    df = pd.read_excel("D:/Source Code/Finpro/App/datamotor.xlsx")
    st.write(df)


elif choose == "Data Training":
        st.subheader('**Clean Dataset Information**')
        st.write(df)

        y = np.array(df['selling_price']) # using selling_price
        X = df.drop(columns=['selling_price']).values # using all the colum except for the last column that is going to be predicted (Y)
        X = np.array(X) 

        # we will now split the data into test and train split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 42)
        
        st.subheader('Data Amount')
        st.write('Row & Column')
        st.info(df.shape)

        st.write('Training Data')
        st.info(X_train.shape)
        st.write('Testing Data')
        st.info(X_test.shape)

        st.subheader('**Variables details**')
        st.write('The X Variable')
        st.info(list[df.iloc[:,:-2].columns])
        st.write('The Y Variable')
        st.info(df.iloc[:,-1].name)
        
        
elif choose == "EDA":
    
    st.markdown('#### Exploratory Data Analysis 📊')
    st.markdown('Based on the dataset that is owned, it can be seen in the graph below the name of motorcycles')

    #opening the image
    image = Image.open('D:/Source Code/Finpro/App/Bar Chart Name.png')

    #displaying the image on streamlit app
    st.image(image)

elif choose == "Prediction":

    # loading in the model to predict on the data  
    model = pickle.load(open('D:/Source Code/Finpro/App/Random_forest model-data.pkl', 'rb'))

    
    st.markdown('Choose the brand name of your motorcycle ?')
    CHOICES = {0 : " Ninja 250 " ,1 : " R25 " ,2 : " Lexi " ,3 : " Beat " ,4 : " Primavera " ,5 : " NMAX " ,6 : " Win 100 " ,7 : " Mio " ,8 : " LX " ,9 : " Vespa 125 " ,10 : " Vespa " ,11 : " Ninja R " ,12 : " Touring " ,13 : " Fino " ,14 : " KLX " ,15 : " YZF R25 " ,16 : " Nuovo " ,17 : " Aerox " ,18 : " Vario " ,19 : " Shadow 750 " ,20 : " Estrella " ,21 : " Xeon " ,22 : " CB 150R " ,23 : " GSX " ,24 : " CBR " ,25 : " Blitz " ,26 : " W800 " ,27 : " PCX " ,28 : " Supra " ,29 : " MT 25 " ,30 : " Scoopy " ,31 : " Z Series " ,32 : " Lain-lain " ,33 : " Kaze " ,34 : " YZF R15 " ,35 : " Monkey " ,36 : " 1200 " ,37 : " D-Tracker " ,38 : " Street " ,39 : " CB " ,40 : " K 1600 " ,41 : " Ninja KRR " ,42 : " Sportster " ,43 : " ER-6N " ,44 : " Jupiter " ,45 : " Sprint " ,46 : " Satria " ,47 : " Patagonian Eagle " ,48 : " Shogun " ,49 : " Ninja " ,50 : " GL " ,51 : " X-Ride " ,52 : " TW 225 " ,53 : " 701 Enduro " ,54 : " New 797 " ,55 : " F 1 ZR " ,56 : " GTS " ,57 : " CRF250Rally " ,58 : " Triumph " ,59 : " Xmax " ,60 : " Vixion " ,61 : " Soul GT " }
    def format_func(option):
        return CHOICES[option]
    merek = st.selectbox("Select option", options=list(CHOICES.keys()), format_func=format_func)
    st.write(f"You selected option {merek} for {format_func(merek)}")
    

    year = st.number_input('In what year was the motorcycle produced? ', value=0,min_value=0, max_value=9999)

    st.markdown('What is the type of motorcycle seller you are (Individual/Dealer)?')
    CHOICES = {1: "Individual", 2: "Dealer"}
    def format_func(option):
        return CHOICES[option]
    seller_type = st.selectbox("Select option", options=list(CHOICES.keys()), format_func=format_func)
    st.write(f"You selected option {seller_type} for {format_func(seller_type)}")
    
    st.markdown("How many owners are you? (1st owner, 2nd owner, 3rd owner)?")
    CHOICES = {0: "1st owner", 1: "2nd owner", 2: "3rd owner" }
    def format_func(option):
        return CHOICES[option]
    owner = st.selectbox("Select option", options=list(CHOICES.keys()), format_func=format_func)
    st.write(f"You selected option {owner} for {format_func(owner)}")

    st.markdown('How much distance has this motorcycle traveled?')
    CHOICES = {0 : " 0-5.000 km " ,1 : " 5.000-10.000 km " ,2 : " 40.000-45.000 km " ,3 : " 35.000-40.000 km " ,4 : " 50.000-55.000 km " ,5 : " 65.000-70.000 km " ,6 : " 70.000-75.000 km " ,7 : " 300.000 km " ,8 : " 30.000-35.000 km " ,9 : " 25.000-30.000 km " ,10 : " 15.000-20.000 km " ,11 : " 45.000-50.000 km " ,12 : " 20.000-25.000 km " ,13 : " 10.000-15.000 km " ,14 : " 80.000-85.000 km " ,15 : " 55.000-60.000 km " ,16 : " 110.000-115.000 km " ,17 : " 60.000-65.000 km " ,18 : " 85.000-90.000 km " ,19 : " 265.000-270.000 km " ,20 : " 75.000-80.000 km "}
    def format_func(option):
        return CHOICES[option]
    km_driven = st.selectbox("Select option", options=list(CHOICES.keys()), format_func=format_func)
    st.write(f"You selected option {km_driven} for {format_func(km_driven)}")

    lst_merek = [merek]
    lst_year = [year]
    lst_seller = [seller_type]
    lst_owner = [owner]
    lst_kmdriven = [km_driven]
                  

    data_pred = pd.DataFrame({'name':lst_merek,
                            'year':lst_year,
                            'seller_type':lst_seller,
                            'owner':lst_owner,
                            'km_driven':lst_kmdriven})

    prediksi = model.predict(data_pred)
    data_pred['selling_price']  = prediksi

    if st.button('Check'):
        data_pred
    else:
        st.write('')

