import pandas as pd

def reformat_pd(df):

    df.rename(columns={df.columns[0]: 'Compound'}, inplace=True)
    df.columns = df.columns.str.replace('smiles', 'SMILES') 
    df.columns = df.columns.str.replace('（', '(')
    df.columns = df.columns.str.replace('）', ')')
    df.columns = df.columns.str.replace('₅₀', '50')
    df.columns = df.columns.str.replace('ECsc', 'EC50')
    df.columns = df.columns.str.replace('ECso', 'EC50')
    df.columns = df.columns.str.replace('ICsc', 'IC50')
    df.columns = df.columns.str.replace('ICso', 'IC50')
    df.columns = df.columns.str.replace('µ', 'u')
    df.columns = df.columns.str.replace('μ', 'u')
    df.columns = df.columns.str.replace(' ', '')

    return df