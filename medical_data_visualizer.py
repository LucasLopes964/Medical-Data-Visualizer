import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# 1
df = pd.read_csv('medical_examination.csv')

# Garanta compatibilidade de nomes esperados pelo teste
# Muitos CSVs trazem 'gender' (1 = female, 2 = male no FCC). O teste espera 'sex'.
if 'gender' in df.columns and 'sex' not in df.columns:
    df = df.rename(columns={'gender': 'sex'})

# 2
bmi = df['weight'] / ((df['height'] / 100) ** 2)
df['overweight'] = (bmi > 25).astype(int)

# 3
df['cholesterol'] = (df['cholesterol'] > 1).astype(int)
df['gluc'] = (df['gluc'] > 1).astype(int)

# 4
def draw_cat_plot():
    # 5
    df_cat = pd.melt(
        df,
        id_vars=['cardio'],
        value_vars=['cholesterol', 'gluc', 'smoke', 'alco', 'active', 'overweight']
    )

    # 6
    df_cat = (
        df_cat
        .groupby(['cardio', 'variable', 'value'])
        .size()
        .reset_index(name='total')
    )
    
    # 7
    order_vars = ['active', 'alco', 'cholesterol', 'gluc', 'overweight', 'smoke']
    g = sns.catplot(
        data=df_cat,
        x='variable',
        y='total',
        hue='value',
        col='cardio',
        kind='bar',
        height=5,
        aspect=1.1,
        order=order_vars
    )
    g.set_axis_labels("variable", "total")

    # 8
    fig = g.fig

    # 9
    fig.savefig('catplot.png')
    return fig


# 10
def draw_heat_map():
    # 11
    df_heat = df[
        (df['ap_lo'] <= df['ap_hi']) &
        (df['height'] >= df['height'].quantile(0.025)) &
        (df['height'] <= df['height'].quantile(0.975)) &
        (df['weight'] >= df['weight'].quantile(0.025)) &
        (df['weight'] <= df['weight'].quantile(0.975))
    ].copy()

    # 12
    # O teste espera a ordem e presença exata das seguintes colunas:
    cols_order = ['id', 'age', 'sex', 'height', 'weight', 'ap_hi', 'ap_lo',
                  'cholesterol', 'gluc', 'smoke', 'alco', 'active', 'cardio', 'overweight']
    # Garanta que todas existem; se alguma faltar, vai acusar erro (melhor falhar cedo)
    missing = [c for c in cols_order if c not in df_heat.columns]
    if missing:
        raise KeyError(f"Colunas ausentes no CSV para o teste: {missing}")

    corr = df_heat[cols_order].corr()

    # 13
    mask = np.triu(np.ones_like(corr, dtype=bool))

    # 14
    fig, ax = plt.subplots(figsize=(12, 10))

    # 15
    sns.heatmap(
        corr,
        mask=mask,
        annot=True,
        fmt=".1f",
        center=0,
        vmax=0.3,
        vmin=-0.3,
        square=True,
        linewidths=0.5,
        cbar_kws={"shrink": 0.8},
        ax=ax
    )
    # Garanta que a ordem dos labels no eixo bate com o esperado
    ax.set_xticklabels(cols_order, rotation=0)
    ax.set_yticklabels(cols_order, rotation=0)

    # 16
    fig.savefig('heatmap.png')
    return fig