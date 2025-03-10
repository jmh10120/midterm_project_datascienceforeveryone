import pandas as pd
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import sklearn #pip install scikit-learn
import plotly.graph_objects as go

from streamlit_option_menu import option_menu #pip install streamlit-option-menu


selected = option_menu (menu_title=None, options=["01 Introduction", "02 Data Visualisation", "03 Predictions", "04 Conclusion"], default_index=0, orientation="horizontal")

df = pd.read_csv("train.csv")

st.title("Increasing Social Media Usage from a Corporate Standpoint")

if selected == "01 Introduction": #first page
    st.markdown("""
                **Introduction:**
                From a corporate standpoint, the time spent on social media is an integral factor of generating company revenue. The more time users spend online, the more advertisements they will consume—crucial to generating commissioned revenue for social media companies. Moreover, the more time users spend on social media, the more data corporate media companies can sell to advertisers, thus again, increasing revenue. Our analysis aimed to pinpoint the variables most indicative of time spent on social media, to help corporate companies identify the engagement  metrics and interface strategies to address to increase user time on social media.
                """)
    # Define dataset variables (columns and their meanings)
    dataset_variables = {
        "User_ID": "Unique identifier for the user.",
        "Age": "Age of the user.",
        "Gender": "Gender of the user (Female, Male, Non-binary).",
        "Platform": "Social media platform used (e.g., Instagram, Twitter, Facebook, LinkedIn, Snapchat, Whatsapp, Telegram).",
        "Daily_Usage_Time (minutes)": "Daily time spent on the platform in minutes.",
        "Posts_Per_Day": "Number of posts made per day.",
        "Likes_Received_Per_Day": "Number of likes received per day.",
        "Comments_Received_Per_Day": "Number of comments received per day.",
        "Messages_Sent_Per_Day": "Number of messages sent per day.",
        "Dominant_Emotion": "User's dominant emotional state during the day (e.g., Happiness, Sadness, Anger, Anxiety, Boredom, Neutral).",
        }
    st.markdown("### :grey[Dataset Variables]")
    st.table(dataset_variables)

    st.markdown("### :grey[Data Exploration]")
    num = st.number_input("No of rows",5,10) #between 5-10
    st.dataframe(df.head(num))

    st.dataframe(df.describe())

    st.write(df.shape)

    st.markdown("### :grey[Pairplot]")
    st.image("DSpairplot.jpeg")
    st.markdown("""
                The strongest linear relationships when daily usage time is the outcome variable include likes per day, messages per day, and comments per day, indicating that an active, rather than passive engagement with content drives an increase in the time people spend on social media. Moreover, likes, messages, and comments all contain a positive relationship with each other—again indicating that engagement rather than passive consumption is a crucial factor in content consumption.
                """)


if selected == "02 Data Visualisation":
    import plotly.express as px #pip install plotly

    st.markdown("### :grey[Data Visualisation]")

    # Create tabs for visualization options
    tab1, tab2, tab3 = st.tabs(["Correlation", "Bar Charts", "Scatterplot"])

# Bar Charts Tab
    with tab2:
        st.subheader("Bar Chart Visualisations")

        # Gender x Minutes Spent Bar Chart
        st.subheader("Gender x Minutes Spent")
        fig, ax = plt.subplots(figsize=(10, 6))  # Adjust the size if needed
        ax.barh(df["Gender"], df["Daily_Usage_Time (minutes)"], color='#756A43')  # Use hex color #756A43
        ax.set_xlabel("Minutes Spent")
        ax.set_ylabel("Gender")
        ax.set_title("Gender x Minutes Spent")
        st.pyplot(fig)
        st.markdown("""
                    Women spend the most time on social media hitting an aggregate amount of 200 minutes.
                    """)

        # Social Media App x Minutes Spent Bar Chart
        st.subheader("Social Media App x Minutes Spent")
        fig, ax = plt.subplots(figsize=(10, 6))  # Adjust the size if needed
        ax.barh(df["Platform"], df["Daily_Usage_Time (minutes)"], color='#756A43')  # Use hex color #756A43
        ax.set_xlabel("Minutes Spent")
        ax.set_ylabel("Platform")
        ax.set_title("Platform x Minutes Spent")
        st.pyplot(fig)
        st.markdown("""
                    Out of all platforms, users spend the most time on Instagram, hitting an aggregate amount of 200 minutes.
                    """)

    # Correlation Tab
    with tab1:
        from sklearn.preprocessing import LabelEncoder
        # Initialize the LabelEncoder
        label_encoder = LabelEncoder()

        # Fit label encoder and return encoded labels
        ## Let's convert categorical text column in categorical number columsn for the model to understand>>
        testList = ["Age","Gender","Platform","Dominant_Emotion"]
        for element in testList:
            df[element] = label_encoder.fit_transform(df[element])
        st.subheader("Correlation Matrix")
        
        # Select numeric columns for correlation
        numeric_columns = df.select_dtypes(include=["number"]).columns.tolist()
        selected_vars = st.multiselect("Select variables for correlation matrix", numeric_columns, default=numeric_columns[:3])

        if len(selected_vars) > 1:
            correlation = df[selected_vars].corr()

            correlation_rounded = correlation.round(2)

            # Create the correlation matrix plot with a grey color scale
            fig = px.imshow(correlation_rounded.values, 
                            x=correlation_rounded.index, 
                            y=correlation_rounded.columns, 
                            labels=dict(color="Correlation"),
                            color_continuous_scale='speed',
                            text_auto=True)
            
            fig.update_layout(
                height=600,  # Increase height of the matrix
                width=800,   # Increase width of the matrix
            )

            st.plotly_chart(fig, theme="streamlit", use_container_width=True)


    # Line Chart Tab (Third Tab for Line Chart)
    with tab3:
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.scatterplot(data=df, x="Posts_Per_Day", y="Daily_Usage_Time (minutes)", color="#FF5733", alpha=0.3, ax=ax)  # Swap Age and Posts_Per_Day

        # Customize the plot
        ax.set_xlabel("Posts Per Day")  # Posts Per Day on x-axis
        ax.set_ylabel("Minutes Spent")  # Age on y-axis
        ax.set_title("Posts Per Day vs Minutes Spent")

        # Display the plot in Streamlit
        st.markdown("### Scatterplot: Posts Per Day vs Minutes Spent")
        st.pyplot(fig)

        st.markdown("""
        **Analysis:**  
        As the number of posts increases, daily usage time also increases, thus there is a positive correlation between posting frequency and social media usage. 

        Most individuals post 1-4 times a day, spending 40 to 120 minutes on social media, and those who post more frequently (7-8 times per day) typically spend over 120 minutes on social media. 

        Those who do not post very frequently also spend less time on social media. Therefore, if social media platforms aim to increase user time, they increase incentives for users (for example, posting more helps the user's videos gain more exposure).
        """)
       


from sklearn.preprocessing import LabelEncoder
# Initialize the LabelEncoder
label_encoder = LabelEncoder()

# Fit label encoder and return encoded labels
## Let's convert categorical text column in categorical number columsn for the model to understand>>
testList = ["Age","Gender","Platform","Dominant_Emotion"]
for element in testList:
    df[element] = label_encoder.fit_transform(df[element])

if selected == "03 Predictions":
   if selected == "03 Predictions":
    from sklearn.model_selection import train_test_split
    ### Step 1: Split dataset into X and y
    X = df.drop(columns=["Daily_Usage_Time (minutes)"])  # remove columns
    y = df["Daily_Usage_Time (minutes)"]  # correct quotation mark

    ### Step2 : split between train and test set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    from sklearn.linear_model import LinearRegression
    ### Step3 : Initialise the linear regression
    linear = LinearRegression()

    ### Step4 : Training of the model
    linear.fit(X_train, y_train)

    ### Step5 : Make some prediction on a chunk of the dataset that the model hasn't seen before
    predictions = linear.predict(X_test)

    ### Step6 : Evaluation of the model output
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

    # Compute Mean Absolute Error (MAE)
    mae = mean_absolute_error(predictions, y_test)

    # Compute Mean Squared Error (MSE)
    mse = mean_squared_error(predictions, y_test)

    # Compute R-squared (R²)
    r2 = r2_score(y_test, predictions)

    # Display the metrics
    st.markdown("### :grey[Evaluation Metrics]")

    st.write(f"**Mean Absolute Error (MAE):** {mae}")
    st.write(f"**Mean Squared Error (MSE):** {mse}")
    st.write(f"**R-squared (R²):** {r2}")

    # Plot Actual vs Predicted graph
    fig, ax = plt.subplots(figsize=(8, 6))  # Create a figure and axis
    sns.scatterplot(x=y_test, y=predictions, alpha=0.5, ax=ax)  # Scatter plot
    ax.set_xlabel("Actual Minutes Spent")  # X-axis label
    ax.set_ylabel("Predicted Minutes Spent")  # Y-axis label
    ax.set_title("Actual vs Predicted Minutes Spent")  # Plot title
    ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', lw=2, ls='-')  # Red line for perfect predictions
    st.pyplot(fig)  # Display the plot
    
    # Create two columns for side-by-side display of predictions and actual values
    col1, col2 = st.columns(2)

    # In the first column, display predictions table
    with col1:
        st.markdown("### Predictions")
        st.write(pd.DataFrame(predictions, columns=["Predicted Minutes Spent"]))

    # In the second column, display actual values table
    with col2:
        st.markdown("### Actual Values")
        st.write(pd.DataFrame(y_test.values, columns=["Actual Minutes Spent"]))


if selected == "04 Conclusion":
    st.markdown("### :grey[Conclusion]")
    st.markdown("""
                For social media companies, the time users spend on social media plays a crucial role in increasing revenue. Our assessment has found that the number of posts people make, likes and comments they receive, and messages they send are all positively correlated with usage time. Furthermore, females spend the most aggregate time on social media and Instagram is the most popular platform. Taking this into consideration, social media companies should increase user incentives to create content through posting, as well as engage with content through likes, comments and messages. Cultivating a space of shared community and network through engagement is crucial to increasing the time users spend on social media. Social media platforms could implement:
                - A reward system for posting consecutively 
                - Co-authored posts 
                - Interactive post features beyond comments (Q&As, live streaming, or games) 
                """)
    st.image("meme.jpg")