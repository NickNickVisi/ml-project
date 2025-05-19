def type(df):
    df['fumator'] = df['fumator'].astype('category')
    df['risc_diabet'] = df['risc_diabet'].astype('category')
    df['activitate_fizica'] = df['activitate_fizica'].astype('category')
    return df