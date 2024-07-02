import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# 1
df = pd.read_csv('/workspace/boilerplate-medical-data-visualizer/medical_examination.csv')

# 2
df['bmi'] = df['weight']/(df['height']*0.01)**2
df['overweight'] = df['bmi'].apply(lambda x : 1 if x > 25 else 0)

# 3
df['cholesterol'] = df['cholesterol'].apply(lambda x: 1 if x > 1 else 0)
df['gluc'] = df['gluc'].apply(lambda x: 1 if x > 1 else 0)
df.drop(columns=['bmi'], inplace=True)

# 4
df_long = pd.melt(df, id_vars=['cardio','age', 'height','weight', 'ap_hi','ap_lo'],
                     value_vars=['cholesterol', 'gluc', 'smoke', 'alco', 'active', "overweight"], var_name='variable', value_name='value')





def draw_cat_plot():
    # 5


    # 6
    df_cat = df_long.copy()
    

    # 7

    # 8
    g = sns.catplot(data=df_long,x='variable', col='cardio', kind='count',hue='value',aspect=1.2)
    g.set_axis_labels("variable", "total")
    g.set_xticklabels([ 'active', 'alco','cholesterol', 'gluc', "overweight", 'smoke'])
    
    fig = g.fig
    # 9
    fig.savefig('catplot.png')
    return fig


# 10
def draw_heat_map():
    # 11
    df_heat = df.copy()
    df_heat = df_heat[(df_heat['ap_lo'] <= df_heat['ap_hi']) 
                        &(df_heat['height'] >= df_heat['height'].quantile(0.025))
                        &(df_heat['height'] <= df_heat['height'].quantile(0.975))
                        &(df_heat['weight'] >= df_heat['weight'].quantile(0.025))
                        &(df_heat['weight'] <= df_heat['weight'].quantile(0.975))]
    # 12
    corr =  df_heat.corr()

    # 13
    mask = np.zeros_like(corr, dtype=np.bool_)
    mask[np.triu_indices_from(mask)] = True


    # 14
    #fig, ax = None
    fig = plt.figure(figsize=(10,7))

    sns.heatmap(data=corr, vmax=0.3, fmt=".1f" ,mask=mask, annot=True)
    
    # 15
    


    # 16
    fig.savefig('heatmap.png')
    return fig
