import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import scipy.stats as stats
import random

def transform_df(df):
    teams =[]
    ppda = []
    xg=[]
    for i, row in df.iterrows():

        if i%2!=0 and i>1:
            teams.append(row["Équipe"])
            ppda.append(row["PPDA"])
            xg.append(row["xG"])
            df.drop(i, inplace=True)
        elif i<=1:
            df.drop(i, inplace=True)
    df = df.assign(teams_against=teams)
    df = df.assign(PPDA_Against=ppda)
    df = df.assign(xG_Against=xg)
    return df

def create_final_df(df, matcha, matchb):
    df.reset_index(inplace=True)
    if matchb ==0:
        ind1 = df[df["Match"]== matcha].index.item()

        index_good = df.index[ind1+1:].tolist()
        new_df=df.drop(index = index_good)
    else:
        ind1 = df[df["Match"]== matcha].index.item()
        ind2 = df[df["Match"]== matchb].index.item()
        new_df=df.loc[ind2:ind1]
    return new_df

def create_chart(data_x, data_y, data_x_match, data_y_match, nb_matchs, name):
    data=[]
    colors = ['#FF4136', '#FF851B', '#FFDC00', '#2ECC40', '#3D9970', '#39CCCC', '#7FDBFF', '#0074D9', '#B10DC9', '#F012BE', '#85144b', '#2F4F4F', '#2E8B57', '#FF69B4', '#00BFFF', '#BA55D3', '#00FF7F', '#FF1493', '#FFD700', '#F08080']
    for i in range(nb_matchs):
        rand = random.randint(1,20)
        if i ==nb_matchs-1:
            trace = go.Scatter(
                x=data_x,
                y=data_y,
                mode='markers',
                name ="Others",
                marker=dict(color="grey", opacity=0.5)
                )

        else:
            trace = go.Scatter(
                x=data_x_match[i],
                y=data_y_match[i],
                mode='markers',
                name = name[i],
                marker=dict(size=10, color=colors[i] ),
                )
        data.append(trace)

    fig = go.Figure(data=data)
    fig.add_vline(x=0, line_width=1, line_dash="dash", opacity =0.5)
    st.plotly_chart(fig)

def get_data_match_outside(df,i, columns, mean, std):
    x=[]
    y=[]
    val = []
    match = df.loc[df["Match"]==i]
    for col in columns:
        x.append((match[col].values[0]-mean[col])/std[col])
        y.append(col)
        val.append(match[col].values[0])
    return x, y, val

def create_data_chart(df_chart, df_tot, columns, matchs):
    size = len(columns)
    nb_matchs = len(matchs)+1
    row_match = []
    val_match=[]
    data_x_match=[]
    data_y_match=[]
    name_x_match=[]
    selected = []
    df_z_score = stats.zscore(df_chart.select_dtypes(["float","int"]))
    mean_z_score = df_chart.select_dtypes(["float","int"]).mean()
    std_z_score = df_chart.select_dtypes(["float","int"]).std()
    for i in matchs:
        inside = df_chart.loc[df_chart["Match"]==i]
        if inside.empty:
            a, b, v = get_data_match_outside(df_tot, i, columns, mean_z_score, std_z_score)
            data_x_match.append(a)
            data_y_match.append(b)
            name_x_match.append(i)
            val_match.append(v)
        else:
            row_match.append(df_chart.loc[df_chart["Match"]==i].index.item())
            name_x_match.append(i)

    for i in row_match:
        selected.append(df_z_score.loc[[i]])
    df_z_score = df_z_score.drop(df_z_score.index[row_match])

    for m in selected:
        x =[]
        y =[]
        for col in columns:
            x.append(m[col].values[0])
            y.append(col)
        data_x_match.append(x)
        data_y_match.append(y)
    data_x=[]
    data_y=[]
    for cols in columns:
         data_x.append(df_z_score[cols].values)
         for e in range(len(df_z_score)):
             data_y.append(cols)


    data_x=list(np.concatenate(data_x))

    create_chart(data_x, data_y, data_x_match, data_y_match, nb_matchs, name_x_match)


st.title("Z-Score sur une période donnée")
uploaded_file = st.sidebar.file_uploader("Choose a file")
if uploaded_file is not None:
    df = pd.read_excel(io=uploaded_file)
    df = df.fillna(0)
    news_df = transform_df(df)
    st.sidebar.title("Choisir les données de comparaison")
    value = st.sidebar.multiselect("Choix variables",news_df.select_dtypes(["float","int"]).columns)
    st.sidebar.title("Choisir les matchs à mettre en évidence")
    matches = st.sidebar.multiselect("Choix match",news_df["Match"])
    st.sidebar.title("Chosir la période")
    tab1, tab2 = st.tabs(["à partir d'un match jusqu'à présent", "entre deux match"])
    with tab1:
        choice1 = tab1.selectbox("Last match",news_df["Match"])
        final_df = create_final_df(news_df, choice1,0 )
        if len(matches)>0 and len(value)>0:
            create_data_chart(final_df, news_df, value, matches )
    with tab2:
        choice2 = tab2.selectbox("A partir",news_df["Match"])
        choice3 = tab2.selectbox("Jusqu'à",news_df["Match"])
        final_df = create_final_df(news_df, choice2,choice3 )
        if len(matches)>0 and len(value)>0:
            create_data_chart(final_df, news_df, value, matches )
