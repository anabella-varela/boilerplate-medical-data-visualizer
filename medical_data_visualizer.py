import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Import data
df = pd.read_csv("medical_examination.csv")

# Add 'overweight' column
df['overweight']= (df['weight'] / (df['height']*0.01)**2).apply(lambda x : 1 if x > 25 else 0)


# Normalize data by making 0 always good and 1 always bad. If the value of 'cholesterol' or 'gluc' is 1, make the value 0. If the value is more than 1, make the value 1.
df.loc[df['cholesterol'] == 1, 'cholesterol'] = 0
df.loc[df['cholesterol'] > 1, 'cholesterol'] = 1
df['gluc'] = np.where(
    df['gluc'] == 1, 0, np.where(df['gluc'] > 1, 1, df['gluc']) 
    )

# Draw Categorical Plot
def draw_cat_plot():
    # Create DataFrame for cat plot using `pd.melt` using just the values from 'cholesterol', 'gluc', 'smoke', 'alco', 'active', and 'overweight'.
    df_cat = pd.melt(df, ['cardio'], ['cholesterol', 'gluc', 'smoke', 'alco', 'active', 'overweight'])


    # Group and reformat the data to split it by 'cardio'. Show the counts of each feature. You will have to rename one of the columns for the catplot to work correctly.
    df_cat = df_cat.groupby(['cardio','variable', 'value']).size().reset_index(name='total')
    

    # Draw the catplot with 'sns.catplot()'
    # Get the figure for the output
    fig = sns.catplot(x="variable", 
            y="total", 
            hue="value", 
            data= df_cat,
            kind="bar",
            col= "cardio").fig

    # Do not modify the next two lines
    fig.savefig('catplot.png')
    return fig


# Draw Heat Map
def draw_heat_map():
    # Clean the data
    df_heat = df.loc[
    (df['ap_lo'] <= df['ap_hi']) &
    (df['height'] >= df['height'].quantile(0.025)) &
    (df['height'] <= df['height'].quantile(0.975)) &
    (df['weight'] >= df['weight'].quantile(0.025)) &
    (df['weight'] <= df['weight'].quantile(0.975))
    ]

    # Calculate the correlation matrix
    corr = df_heat.corr('pearson')

    # Generate a mask for the upper triangle
    mask = np.triu(corr)

    # Set up the matplotlib figure
    fig, ax =  plt.subplots()

    # Draw the heatmap with 'sns.heatmap()'
    sns.heatmap(corr,
            annot=True, fmt=".1f",
            square= True,
            linewidths= 0.5,
            center= -0.08,
            cbar_kws= {"shrink":0.5},
            mask= mask)

    # Do not modify the next two lines
    fig.savefig('heatmap.png')
    return fig
