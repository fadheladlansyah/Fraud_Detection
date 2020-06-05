import gc

def map_email(df, mapping):
    for col in ['P_emaildomain', 'R_emaildomain']:
        df[col] = df[col].map(mapping)
    
    gc.collect()
    return df        


def map_DeviceInfo(df, col='DeviceInfo'):
    
    df['have_DeviceInfo'] = df[col].isna().astype(int)
    
    df[col] = df[col].fillna('unknown').str.lower()
    df[col] = df[col].str.split('/', expand=True)[0]
    df[col] = df[col].apply(lambda x: x.lower())

    df.loc[df[col].str.contains('windows', na=False), col] = 'Windows'
    df.loc[df[col].str.contains('sm-', na=False), col] = 'Samsung'
    df.loc[df[col].str.contains('samsung', na=False), col] = 'Samsung'
    df.loc[df[col].str.contains('gt-', na=False), col] = 'Samsung'
    df.loc[df[col].str.contains('moto', na=False), col] = 'Motorola'
    df.loc[df[col].str.contains('lg-', na=False), col] = 'LG'
    df.loc[df[col].str.contains('rv:', na=False), col] = 'RV'
    df.loc[df[col].str.contains('huawei', na=False), col] = 'Huawei'
    df.loc[df[col].str.contains('ale-', na=False), col] = 'Huawei'
    df.loc[df[col].str.contains('-l', na=False), col] = 'Huawei'
    df.loc[df[col].str.contains('hi6', na=False), col] = 'Huawei'
    df.loc[df[col].str.contains('blade', na=False), col] = 'ZTE'
    df.loc[df[col].str.contains('trident', na=False), col] = 'Trident'
    df.loc[df[col].str.contains('lenovo', na=False), col] = 'Lenovo'
    df.loc[df[col].str.contains('xt', na=False), col] = 'Sony'
    df.loc[df[col].str.contains('f3', na=False), col] = 'Sony'
    df.loc[df[col].str.contains('f5', na=False), col] = 'Sony'
    df.loc[df[col].str.contains('lt', na=False), col] = 'Sony'
    df.loc[df[col].str.contains('htc', na=False), col] = 'HTC'
    df.loc[df[col].str.contains('mi', na=False), col] = 'Xiaomi'

    cat_so_far = ['unknown', 'windows', 'ios device', 'macos', 'Samsung', 'Trident', 'RV',
                    'Motorola', 'Huawei', 'LG', 'Sony', 'ZTE', 'HTC', 'Lenovo', 'Xiaomi']
    df[col] = df[col].apply(lambda x: x if x in cat_so_far else 'other')
    
    gc.collect()
    return df


def map_id30(df, col='id_30'):
    
    df['have_id30'] = df[col].isna().astype(int)
    
    df[col] = df[col].fillna('unknown')
    df[col] = df[col].str.split(' ', expand=True)[0]
    
    gc.collect()
    return df


def map_id31(df, col='id_31'):
    
    df['have_id31'] = df[col].isna().astype(int)
    
    df[col] = df[col].fillna('unknown')

    df.loc[df[col].str.contains('chrome', na=False), col] = 'chrome'
    df.loc[df[col].str.contains('safari', na=False), col] = 'safari'
    df.loc[df[col].str.contains('firefox', na=False), col] = 'firefox'
    df.loc[df[col].str.contains('edge', na=False), col] = 'edge'
    df.loc[df[col].str.contains('ie', na=False), col] = 'ie'
    df.loc[df[col].str.contains('android', na=False), col] = 'default'
    df.loc[df[col].str.contains('samsung', na=False), col] = 'default'
    df.loc[df[col].str.contains('browser', na=False), col] = 'default'

    df.loc[df[col].isin(df[col].value_counts()[df[col].value_counts() < 200].index), col] = "other"

    gc.collect()
    return df


def map_id33(df, col='id_33'):
    
    df['have_id33'] = df[col].isna().astype(int)
    
    df[col] = df[col].fillna('0')
    df[col] = df[col].str.split('x', expand=True)[0].astype(int)
    df[col] = df[col].map(lambda x: int(x/1000))

    gc.collect()
    return df

