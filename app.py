# Standard imports
import pandas as pd

# matplotlib
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
#plotly
import plotly.express as px
import plotly.graph_objects as go

import streamlit as st

st.title("MPG")

df = pd.read_csv("data/mpg.csv")

# Basic set-up of the page:
# First the checkbox to show the data frame
if st.sidebar.checkbox('Show dataframe'):
    st.header("dataframe")
    st.dataframe(df.head())

# Then the radio botton for the plot type
show_plot = st.sidebar.radio(
    label='Choose Plot type', options=['Matplotlib', 'Plotly'])

st.header("Highway Fuel Efficiency")
years = ["All"]+sorted(pd.unique(df['year']))
year = st.sidebar.selectbox("choose a Year", years)   # Here the selection of the year.
car_classes = ['All'] + sorted(pd.unique(df['class']))
car_class = st.sidebar.selectbox("choose a Class", car_classes)  # and the selection of the class.

show_means = st.sidebar.radio(
    label='Show Class Means', options=['Yes', 'No'])

models = ['No','linear']
model = st.sidebar.selectbox("Fit a model", models)  # and the selection of the class.

st.subheader(f'Fuel efficiency vs. engine displacement for {year}')

############ add by me ##############
import statsmodels.formula.api as smf

def fit_and_predict(df):
    # fit a model explaining hwy fuel mileage through displacement
    lm = smf.ols(formula="hwy ~ displ", data=df).fit()
    
    # find two points on the line represented by the model
    x_bounds = [df['displ'].min(), df['displ'].max()]
    preds_input = pd.DataFrame({'displ': x_bounds})
    predictions = lm.predict(preds_input)
    return lm, pd.DataFrame({'displ': x_bounds, 'hwy': predictions})

##################################################

# With these functions we wrangle the data and plot it.
def mpg_mpl(year, car_class, show_means):
    fig, ax = plt.subplots()
    if year == 'All':
        group = df
    else:
        group = df[df['year'] == year]
    if car_class != 'All':
        st.text(f'plotting car class: {car_class}')
        group = group[group['class'] == car_class]
#############add by me#############
    if model == "linear" :
        lm, pred = fit_and_predict(group)
        rsquared = lm.rsquared
        ax.plot('displ', 'hwy', data =pred, c = "#ff7f00", label= "linear model, r2 = "+str(round(rsquared,3)))
        ax.legend(loc='upper right', bbox_to_anchor=(1.1, 1))
#################################    
    group.plot('displ', 'hwy', marker='.', linestyle='', ms=12, alpha=0.5, ax=ax, legend=None)
    if show_means == "Yes":
        means = df.groupby('class').mean(numeric_only=True)
        for cc in means.index:
            ax.plot(means.loc[cc, 'displ'], means.loc[cc, 'hwy'], marker='.', linestyle='', ms=12, alpha=1, label=cc)
        ax.legend(loc='upper right', bbox_to_anchor=(1.1, 1))
    ax.set_xlim([1, 8])
    ax.set_ylim([10, 50])
    plt.close()
    return fig


def mpg_plotly(year, car_class, show_means):
    if year == 'All':
        group = df
    else:
        group = df[df['year'] == year]
    if car_class != 'All':
        group = group[group['class'] == car_class]      
    fig = px.scatter(group, x='displ', y='hwy', opacity=0.5, range_x=[1, 8], range_y=[10, 50])
    if show_means == "Yes":
        means = df.groupby('class').mean(numeric_only=True).reset_index()
        fig = px.scatter(means, x='displ', y='hwy', opacity=0.5, color='class', range_x=[1, 8], range_y=[10, 50])
        fig.add_trace(go.Scatter(x=group['displ'], y=group['hwy'], mode='markers', name=f'{year}_{car_class}',
                                 opacity=0.5, marker=dict(color="RoyalBlue")))
#############add by me#############
    if model == "linear" :
        lm, pred = fit_and_predict(group)
        rsquared = lm.rsquared
        fig.add_trace(go.Scatter(x=pred['displ'], y=pred['hwy'], mode='lines', name=f'linear model, r2 = {round(rsquared,3)}',
                                 opacity=0.5, marker=dict(color="#ff7f00")))
        #ax.plot('displ', 'hwy', data =pred, c = "#ff7f00", label= "linear model, r2 = "+str(round(rsquared,3)))
#################################    
    return fig
  

if show_plot == 'Plotly':
    st.plotly_chart(mpg_plotly(year, car_class, show_means))
    
else:
    st.pyplot(mpg_mpl(year, car_class, show_means))

