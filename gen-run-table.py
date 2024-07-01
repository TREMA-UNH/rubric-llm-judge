#!/usr/bin/env python3

from pathlib import Path
import pandas as pd
import math

x = pd.read_csv('out-final/summary', names=['prompt', 'classifier', 'restart', 'train-kappa', 'validate-kappa'], delimiter=' ')

x['classifier'] = x['classifier'].str.replace('Method.', '')
best = x.groupby(['prompt','classifier'])['validate-kappa'].transform('max')
x['best'] = x['validate-kappa'] == best
print(x)

x['dev-kappa'] = None
for i in x.index:
    y = x.iloc[i]
    if y['best']:
        predict_log = Path('out-final') / f"{y['prompt']}-{y['classifier']}" / y['classifier'] / 'predict.stderr.log'
        kappa = predict_log.read_text().split('\n')[-2].split('=')[1]
        x.at[i,'dev-kappa'] = float(kappa)

table = x.rename(columns={
        'prompt': 'feature set',
        'train-kappa': r'train $\kappa$',
        'validate-kappa': r'validate $\kappa$',
        'dev-kappa': r'all $\kappa$',
        'best': 'submitted',
    }).to_latex(
        index=False,
        na_rep='',
        formatters={
            r'train $\kappa$': lambda n: f'{n:1.3f}',
            r'validate $\kappa$': lambda n: f'{n:1.3f}',
            'best': lambda n: r'\checkmark' if n else '',
            r'all $\kappa$': lambda n: f'{n:1.3f}',
    })
print(table)
