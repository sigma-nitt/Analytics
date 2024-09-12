import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import seaborn as sns
import statsmodels.api as sm

df = pd.read_csv('FMCG_data.csv')
df['workers_num'] = df['workers_num'].fillna(df['workers_num'].median())
df.drop(columns=['wh_est_year'] , inplace = True)

st.markdown('<h4>Exploratory Data Analysis</h4>', unsafe_allow_html = True)
fig = px.histogram(
        df,
        x='zone',
        color='WH_regional_zone',
        title='Distribution of Warehouses Across Different Regions and Zones',
        category_orders={"WH_regional_zone": ["Zone 1", "Zone 2","Zone 3","Zone 4","Zone 5","Zone 6"]},
        text_auto=True
)
st.plotly_chart(fig)
st.markdown("<h5>Insight:</h5>" , unsafe_allow_html=True)
st.markdown("""This stacked bar chart illustrates the distribution of warehouses across different regions (West, South, North, and East) and zones (Zone 1 to Zone 6). 
            The West region has a total of 8,911 warehouses, with Zone 6 having the highest count (2,398) and Zone 1 the lowest (490). The South region has 5,682 warehouses, 
            where Zone 6 also leads (1,364) and Zone 1 is minimal (680). The North region contains 11,282 warehouses, with Zone 6 again dominant (4,519) and Zone 1 the least (841). 
            The East region has the smallest total of 58 warehouses, with each zone contributing similarly. This chart highlights significant regional disparities in warehouse distribution, 
            with the North region having the highest concentration, particularly in Zone 6""")
fig = px.histogram(df, x='zone', y='Competitor_in_mkt', color='WH_regional_zone', title='Number of Competitors in the Market Across Different Regions and Zones', barmode='group')
st.plotly_chart(fig)
st.markdown("<h5>Insight:</h5>" , unsafe_allow_html=True)
st.markdown("""This bar chart shows the number of competitors in the market across different regions (West, South, North, and East) and zones (Zone 1 to Zone 6). 
                In the West region, Zone 6 has the highest number of competitors at around 9,000, followed by Zone 5 with approximately 5,000. In the South, Zone 4 leads 
                with around 7,000 competitors, while other zones show lower counts. The North region also sees a peak in Zone 6, with approximately 13,000 competitors, 
                followed by Zone 5 with around 5,000. The East region has significantly fewer competitors, with each zone contributing minimally. This chart highlights 
                the competitive landscape, with Zone 6 being particularly prominent in both the West and North regions.""")
fig = px.scatter(df, x='Competitor_in_mkt', y='product_wg_ton', color='zone', title='Impact of Competitors on Product Demand')
st.plotly_chart(fig)
st.markdown("<h5>Insight:</h5>" , unsafe_allow_html=True)
st.markdown("""This scatter plot demonstrates the impact of competitors on product demand across different regions (West, North, South, and East). 
                The x-axis represents the number of competitors in the market, while the y-axis shows product weight in tons. Each color corresponds to a 
                different region: West (red), North (orange), South (green), and East (purple). The plot reveals a significant clustering of data points 
                between 0 to 8 competitors, with product weight varying from 10,000 to over 50,000 tons. Notably, the South and East regions (green and purple) 
                exhibit the highest product weights across various competitor counts. The East region also shows some outliers with up to 12 competitors. 
                This chart highlights the relationship between market competition and product demand across different regions.""")
fig = px.histogram(df, x='zone', y='govt_check_l3m', title='Frequency of Government Checks in Different Regions', text_auto=True)
st.plotly_chart(fig)
st.markdown("<h5>Insight:</h5>" , unsafe_allow_html=True)
st.markdown("""The North zone has the most frequent government closures, at around 204,827. The West zone has a mid-range value of government closures, 
                at around 126,761. The South zone has a value close to the West zone, at around 128,944. The East zone has the least frequent government closures, 
                at around 9,775. Overall, the graph suggests a significant difference in the frequency of government closures between the North zone and the other three zones. 
                The West, South, and East zones have comparable closure rates, while the North zone experiences substantially more closures.""")
fig = px.scatter(df, x='govt_check_l3m', y='storage_issue_reported_l3m', color='zone', title='Correlation between Government Checks and Reported Storage Issues')
st.plotly_chart(fig)
st.markdown("<h5>Insight:</h5>" , unsafe_allow_html=True)
st.markdown("""The scatter plot depicts the correlation between government checks conducted in the last three months (govt_check_l3m) and reported storage issues 
            (storage_issue_reported_l3m) across different zones. The zones are color-coded: West (blue), North (red), South (green), and East (purple). Each dot represents
            a warehouse, showing the frequency of government checks on the x-axis and the number of storage issues on the y-axis. There is no apparent linear trend,
            suggesting that frequent government checks do not necessarily correlate with the number of reported storage issues. Notably, warehouses in the South zone (green)
            exhibit a higher concentration of both government checks and reported storage issues.""")
fig = px.histogram(df,
                   x='Location_type',
                   title='Distribution of Location Type',
                   text_auto=True,
                   category_orders={'Location_type': ['urban', 'rural']})
st.plotly_chart(fig)
fig = px.scatter(df,
                  x='dist_from_hub',
                  y='num_refill_req_l3m',
                  color = 'zone',
                  title='Effect of Distance from Production Hub on Number of Refills',
                  labels={
                      'dist_from_hub': 'Distance from Hub (km)',
                      'num_refill_req_l3m': 'Number of Refills (Last 3 Months)'
                  },
                  trendline='ols',
                  trendline_scope='overall',
                  template='plotly_white')
st.plotly_chart(fig)
st.markdown("<h5>Insight:</h5>" , unsafe_allow_html=True)
st.markdown("""The graph shows the effect of distance from a production hub on the number 
            of refills needed for machines over a three-month period. The x-axis represents 
            the distance from the hub in kilometers, ranging from 0 to 250 kilometers. The y-axis represents the number of refills needed in the last three months.There is a general downward trend in the graph, indicating that as the distance from the production hub increases, the number of refills needed decreases. This could be because machines located further away are newer models that require fewer refills, or because they are used less frequently.It's important to note that the graph doesn't show the specific type of machine or what the refills are for.""")
fig = px.scatter(df, x='dist_from_hub', y='transport_issue_l1y', color='zone', title='Relation Between Distance from Hub and Transport Issues')
st.plotly_chart(fig)
st.markdown("<h5>Insight:</h5>" , unsafe_allow_html=True)
st.markdown("This scatter plot illustrates the relationship between the distance from the hub (dist_from_hub) and transport issues reported yearly (transport_issue_1ly) across four regions: West (blue), North (orange), South (green), and East (purple). The x-axis represents the distance from the hub in unspecified units, while the y-axis indicates the frequency of transport issues, ranging from 0 to 5. The data points are evenly distributed across the y-axis for all distances, indicating no clear correlation between the distance from the hub and the number of transport issues. Each region's data points are spread similarly, suggesting consistent transport issue patterns regardless of the zone's distance from the hub. This chart underscores that transport issues do not increase with greater distance from the hub")
competitor_df = df[df['Competitor_in_mkt'].notnull()]
competitor_presence = competitor_df.groupby('zone').size().reset_index(name='Competitor Count')
color_scale = px.colors.qualitative.Dark24[:len(competitor_presence)]
fig = px.scatter(competitor_presence, x='zone', y='Competitor Count', size='Competitor Count', color='zone',
                 color_discrete_sequence=color_scale, hover_name='zone', size_max=50,
                 title='Competitor Presence by Zone')
fig.update_layout(xaxis_title='Zone', yaxis_title='Competitor Count')
st.plotly_chart(fig)
st.markdown("<h5>Insight:</h5>" , unsafe_allow_html=True)
st.markdown("This bubble chart visualizes the competitor presence by zone, with each zone represented by a distinct color: East (blue), North (pink), South (green), and West (red). The x-axis categorizes the zones, while the y-axis indicates the competitor count. The bubble size reflects the number of competitors, highlighting significant differences among zones. The North zone has the highest competitor count, followed by the West and South zones. The East zone has the smallest bubble, indicating the fewest competitors. This visualization effectively communicates the distribution of competitors across different regions, emphasizing where market competition is most intense and where it is less pronounced")
fig = px.scatter(df, x = 'workers_num' , y = 'wh_breakdown_l3m' , title = 'Relation between number of workers and breakdown')
st.plotly_chart(fig)
st.markdown("<h5>Insight:</h5>" , unsafe_allow_html=True)
st.markdown("This shows the correlation between the number of workers (workers_num) and the frequency of warehouse breakdowns in the last three months (wh_breakdown_l3m). The x-axis represents the number of workers, ranging from 10 to 100, while the y-axis represents the number of breakdowns, ranging from 0 to 6.Warehouses with fewer workers (10-50) exhibit a wide range of breakdowns (0-6), indicating no clear pattern. As the number of workers increases (50-100), the frequency of breakdowns tends to stabilize, mostly around 0-3. The plot suggests a potential negative correlation, where warehouses with more workers may experience fewer breakdowns, but this is not conclusive from the given data.")
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
columns_to_include = [col for col in df.columns if col not in ['Ware_house_ID', 'WH_Manager_ID']]
df1 = df[columns_to_include].copy()
df1['Location_type'] = le.fit_transform(df1['Location_type'])
df1['WH_capacity_size'] = le.fit_transform(df1['WH_capacity_size'])
df1['approved_wh_govt_certificate'] = le.fit_transform(df1['approved_wh_govt_certificate'])
df1['zone'] = le.fit_transform(df1['zone'])
df1['WH_regional_zone'] = le.fit_transform(df1['WH_regional_zone'])
df1['wh_owner_type'] = le.fit_transform(df1['wh_owner_type'])
fig, ax = plt.subplots()
sns.heatmap(df1.corr(), ax=ax)
st.pyplot(fig)
st.markdown("<h5>Insight:</h5>",unsafe_allow_html= True)
st.markdown("""In this correlation heatmap, we see the relationships between various features. Strong positive correlations (dark red) indicate features that tend to move together. Strong negative correlations (dark blue) suggest features that move in opposite directions. Weaker correlations (white) show little to no linear relationship.""")
fig = px.histogram(df, x='WH_regional_zone', y='retail_shop_num', color='WH_regional_zone',
                   title='Histogram of Number of Retail Shops by Region',
                   labels={'WH_regional_zone': 'Regional Zone', 'retail_shop_num': 'Number of Retail Shops'})
st.plotly_chart(fig)
st.markdown("<h5>Insight:</h5>",unsafe_allow_html=True)
st.markdown("The distribution of retail shop locations across various regions is depicted in this histogram. The vertical axis (y-axis) quantifies the number of shops within each region, while the horizontal axis (x-axis) categorizes the regions. Evidently, Zone 6 houses the most significant number of retail shops, followed by Zones 1 and 5. In contrast, Zone 4 exhibits the lowest number of retail establishments.")
fig = px.violin(df, x='zone', y='WH_capacity_size', color='zone',
                title='Violin Plot of Warehouse Capacity Sizes by Zone',
                labels={'zone': 'Zone', 'WH_capacity_size': 'Warehouse Capacity Size'})
st.plotly_chart(fig)
st.markdown("<h5>Insight:</h5>" , unsafe_allow_html=True)
st.markdown("The violin plot reveals distinct warehouse capacity patterns across zones. The East zone houses the largest warehouses, with a significant cluster in the mid-range. In contrast, the North and South zones predominantly feature mid-sized warehouses. The West zone is characterized by a concentration of smaller warehouses, with a few exceptions in the mid-range. Notably, the East zone also showcases a few outlier warehouses, significantly larger than their counterparts. This visual representation underscores the diverse warehouse capacities across regions, offering valuable insights for strategic planning and resource allocation.")

fig = px.box(df, x='zone', y='dist_from_hub', color='zone',
             title='Box Plot of Distance from Hub by Zone',
             labels={'zone': 'Zone', 'dist_from_hub': 'Distance from Hub (km)'})
st.plotly_chart(fig)
st.markdown("<h5>Insight:</h5>",unsafe_allow_html=True)
st.markdown("The box plot illustrates the distribution of distances from the hub across different zones. East zone has the highest median distance, with a wider range compared to other zones. West and North zones have similar median distances but with varying spread. South zone displays the smallest median distance and the narrowest range, indicating more consistent proximity to the hub. Overall, the plot highlights regional disparities in distance from the central hub, which could impact logistics and transportation planning.")

fig = px.scatter_3d(df, x='WH_capacity_size', y='dist_from_hub', z='workers_num',
                    color='zone', size='num_refill_req_l3m',
                    title='3D Scatter Plot of Warehouse Attributes',
                    labels={'WH_capacity_size': 'Warehouse Capacity Size', 'dist_from_hub': 'Distance from Hub (km)', 'workers_num': 'Number of Workers'})
st.plotly_chart(fig)
st.markdown("<h5>Insight:</h5>",unsafe_allow_html=True)
st.markdown("The 3D scatter plot illustrates the relationship between warehouse capacity size, distance from the hub, and the number of workers across different zones. East zone houses large warehouses closer to the hub with a higher workforce. West zone predominantly comprises small warehouses further from the hub with fewer workers. North and South zones exhibit a mix of warehouse sizes at varying distances and workforce numbers. This visualization offers insights into regional warehouse characteristics, aiding in understanding operational differences and resource allocation strategies.")
fig = px.funnel(df, x='num_refill_req_l3m', y='zone',
                title='Funnel Chart of Warehouse Refills and Breakdowns',
                labels={'num_refill_req_l3m': 'Number of Refills (Last 3 Months)', 'zone': 'Zone'})
st.plotly_chart(fig)
st.markdown("<h5>Insight:</h5>",unsafe_allow_html=True)
st.markdown("The funnel chart illustrates warehouse refills and breakdowns across different zones. The West zone demonstrates the highest number of refills, followed by the North zone. A significantly lower number of refills are observed in the South zone, and the East zone has the lowest refill count. The chart highlights a substantial drop in the number of refills from the West to the East zone, indicating potential issues with warehouse management or supply chain efficiency in the eastern region. ")

