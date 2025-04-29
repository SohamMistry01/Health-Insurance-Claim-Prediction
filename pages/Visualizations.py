import streamlit as st
import pandas as pd
import base64
import plotly.figure_factory as ff
import plotly.express as px

st.toast("These visualizations may take few moments to load!")

# Background Image CSS
def set_bg(image_file):
    with open(image_file, "rb") as f:
        data = base64.b64encode(f.read()).decode("utf-8")

    bg_image_style = f"""
    <style>
    .stApp {{
        background-image: url("data:image/jpg;base64,{data}");
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
    }}
    </style>
    """
    st.markdown(bg_image_style, unsafe_allow_html=True)

# Set Background
set_bg("bg_2.jpg")

st.title("Interactive Data Visualizations on the Dataset")
st.divider()

df = pd.read_csv("Medicalpremium.csv")

st.write("Heatmap for visualizing correlation:")

# Compute correlation matrix
corr_matrix = df.corr()

# Create heatmap using Plotly
fig = ff.create_annotated_heatmap(
    z=corr_matrix.values,
    x=list(corr_matrix.columns),
    y=list(corr_matrix.index),
    annotation_text=corr_matrix.round(2).values,
    colorscale='Cividis',  # You can adjust the color scheme
    showscale=True
)

# Save heatmap as an image (optional)
fig.write_image("heatmap.png")

# Display the Plotly figure in Streamlit
st.plotly_chart(fig)
if st.button('View Heatmap Insights🔎'):
    st.info("""
    Age and Premium Price Correlation: Age has a strong positive correlation (0.7) 
    with Premium Price, indicating that older individuals tend to have higher insurance costs.

    Health Conditions Impact: AnyChronicDiseases (0.29) and BloodPressureProblems (0.24) show a 
    moderate positive correlation with Premium Price, suggesting medical conditions influence premium costs.

    Number of Major Surgeries Effect: The NumberOfMajorSurgeries has a weak correlation (0.26) with 
    Premium Price, implying that past surgeries slightly affect insurance premiums.
    """)

st.divider()
st.write("Distribution of age data:")

# Create KDE distribution plot using Plotly
fig = ff.create_distplot([df['Age']], ['Age'], show_hist=True, show_rug=False)

# Display the Plotly figure in Streamlit
st.plotly_chart(fig)
if st.button('View Age Distribution Insights🔎'):
    st.info("""
    Age Distribution: The histogram shows the distribution of age in the dataset, with most data points 
    spread between 20 and 65 years.

    Density Trend: The KDE (Kernel Density Estimation) curve indicates a peak around 40–50 years, 
    suggesting a higher concentration of individuals in this age range.

    Variability: The histogram has fluctuations, with some age groups having significantly lower frequencies, 
    reflecting gaps or irregular distribution in the dataset.
    """)

st.divider()
st.write("Distribution of Premium Price:")

# Create histogram with KDE
fig = px.histogram(df, x='PremiumPrice', marginal="violin", nbins=30, opacity=0.7, histnorm='probability density')

# Display the Plotly figure in Streamlit
st.plotly_chart(fig)
if st.button('View Premium Price Distribution Insights🔎'):
    st.info("""
    Premium Price Distribution: The histogram shows the distribution of premium prices, with peaks 
    around 15k, 25k, and 30k, indicating common premium cost ranges in the dataset.

    Density Variation: The violin plot at the top represents the probability density of premium prices, 
    showing higher concentration around 15k–30k with some spread towards higher values.

    Skewed Distribution: The right side of the histogram has fewer instances, suggesting that 
    higher premium prices (above 35k) are less frequent, while lower premiums are more common.
    """)

# Creating salary bins to visualize data distribution of premium price and age
pr_lab=['Low','Basic','Average','High','SuperHigh']
df['PremiumLabel']=pr_bins=pd.cut(df['PremiumPrice'],bins=5,labels=pr_lab,precision=0)
df['AgeLabel']=pr_bins=pd.cut(df['Age'],bins=5,labels=pr_lab,precision=0)
df['WeightLabel']=pr_bins=pd.cut(df['Weight'],bins=5,labels=pr_lab,precision=0)
df['HeightLabel']=pr_bins=pd.cut(df['Height'],bins=5,labels=pr_lab,precision=0)

st.divider()
st.write("Scatter plot between Age and Premium Price:")

# Create scatter plot using Plotly
fig = px.scatter(df, x='Age', y='PremiumPrice', labels={'Age': 'Age', 'PremiumPrice': 'Premium Price'})

# Save scatter plot as an image (optional)
fig.write_image("scatterplot.png")

# Display the Plotly figure in Streamlit
st.plotly_chart(fig)
if st.button('View Scatterplot Insights🔎'):
    st.info("""
    Positive Correlation: The scatter plot shows that premium prices tend to increase with age, 
    suggesting older individuals generally have higher insurance costs.

    Clustered Price Ranges: Premium prices are concentrated around 15k, 25k, and 30k, indicating 
    common pricing tiers across different age groups.

    Wide Variability: While younger individuals mostly have lower premium prices, older individual
    exhibit a wider spread, with some paying as high as 40k.
    """)

st.divider()
st.write("Number of people opted for insurance based on premium-category after transplants:")

# Create count plot using Plotly
fig = px.histogram(df, x='PremiumLabel', color='AnyTransplants', barmode='group', text_auto=True)

# Save the figure as an image (optional)
fig.write_image("countplot1.png")

# Display the Plotly figure in Streamlit
st.plotly_chart(fig)
if st.button('View Countplot-1 Insights🔎'):
    st.info("""
    Majority in Basic and Average Premiums: Most individuals fall into the Basic (382) and Average 
    (252) premium categories, indicating that lower-cost insurance plans are more common.

    Impact of Transplants on SuperHigh Premiums: The SuperHigh category has 34 individuals with transplants,
    showing a strong link between transplants and higher insurance costs.

    Minimal Transplants in Lower Premiums: Basic (1), Average (7), and Low (11) premium categories have 
    very few transplant cases, suggesting that transplants significantly drive up insurance costs.
    """)

st.divider()
st.write("Number of people opted for insurance based on premium-category after how many surgeries:")

# Create count plot using Plotly
fig = px.histogram(df, x='PremiumLabel', color='NumberOfMajorSurgeries', barmode='group', text_auto=True)

# Save the figure as an image (optional)
fig.write_image("countplot2.png")

# Display the Plotly figure in Streamlit
st.plotly_chart(fig)
if st.button('View Countplot-2 Insights🔎'):
    st.info("""
    Higher Surgeries Lead to Higher Premiums: Individuals with 3 major surgeries are mostly in the Average 
    category (17), indicating that more surgeries correlate with increased premium costs.

    Most People Have 0 or 1 Surgery: The Basic, Low, and High categories are dominated by individuals 
    with 0 or 1 major surgery, showing that fewer surgeries are common among policyholders.

    SuperHigh Premiums and Surgeries: The SuperHigh category includes only 1 individual with 3 surgeries, 
    suggesting that extreme premium costs are rare but possible with multiple surgeries.
    """)

st.divider()
st.write("Number of people in each premium-label based on their age-group")

# Create count plot using Plotly
fig = px.histogram(df, x='PremiumLabel', color='AgeLabel', barmode='group', text_auto=True)

# Save the figure as an image (optional)
fig.write_image("countplot3.png")

# Display the Plotly figure in Streamlit
st.plotly_chart(fig)
if st.button('View Countplot-3 Insights🔎'):
    st.info("""
    Basic and Low Premiums Dominate: The majority of individuals fall under the Basic (191) and 
    Low (181) AgeLabel categories, suggesting that younger individuals tend to have lower insurance premiums.

    Higher Premiums for Older Individuals: The SuperHigh and High premium categories have a greater proportion 
    of individuals from the High (55) and Average (26) AgeLabel groups, indicating that older individuals are 
    more likely to pay higher premiums.

    Fewer Young Individuals in Expensive Plans: The Low AgeLabel (green) is mostly in the Low premium category, 
    with almost no representation in SuperHigh or High premiums, reinforcing that younger people generally have 
    lower insurance costs.
    """)

st.divider()
st.write("Number of people in each weight-label based on their age-group")

# Create count plot using Plotly
fig = px.histogram(df, x='WeightLabel', color='AgeLabel', barmode='group', text_auto=True)

# Save the figure as an image (optional)
fig.write_image("countplot4.png")

# Display the Plotly figure in Streamlit
st.plotly_chart(fig)
if st.button('View Histogram Insights🔎'):
    st.info("""
    Basic Weight Category is the Most Common: The majority of individuals fall under the Basic weight category, 
    with all age groups (Low, High, Basic, SuperHigh, and Average) having significant representation.

    Low and Average Weight Categories are Evenly Distributed: The Low and Average weight groups have fairly 
    balanced distributions across different AgeLabel categories, with no single age group dominating them significantly.

    Few Individuals in High and SuperHigh Weight Categories: The SuperHigh and High weight categories have the 
    least number of individuals, with only a handful of cases across all age groups, suggesting that extremely 
    high weight is uncommon in the dataset.
    """)




