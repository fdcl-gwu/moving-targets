import pandas as pd
import numpy as np
import argparse

measures = {
    'Adj. Targets it {}, train set cost_DIDI': 'DIDI tr OPT',
    'Adj. Targets it {}, test set cost_MSE': 'MSE tr OPT',
    'Adj. Targets it {}, test set cost_R2': 'R2 tr OPT',
    'Trained model it {}, train set cost_DIDI perc. index': 'DIDI tr model',
    'Trained model it {}, test set cost_DIDI perc. index': 'DIDI ts model',
    'Trained model it {}, train set score_MSE': 'MSE tr model',
    'Trained model it {}, train set score_R2': 'R2 tr model',
    'Trained model it {}, test set score_MSE': 'MSE ts model',
    'Trained model it {}, test set score_R2': 'R2 ts model',

    'Adj. Targets it {}, train set cost_DIDI Std': 'Std DIDI tr OPT',
    'Adj. Targets it {}, test set cost_MSE Std': 'Std MSE tr OPT',
    'Adj. Targets it {}, test set cost_R2 Std': 'Std R2 tr OPT',
    'Trained model it {}, train set cost_DIDI perc. index Std': 'Std DIDI tr model',
    'Trained model it {}, test set cost_DIDI perc. index Std': 'Std DIDI ts model',
    'Trained model it {}, train set score_MSE Std': 'Std MSE tr model',
    'Trained model it {}, train set score_R2 Std': 'Std R2 tr model',
    'Trained model it {}, test set score_MSE Std': 'Std MSE ts model',
    'Trained model it {}, test set score_R2 Std': 'Std R2 ts model',
}


def analyze_results():

    parser = argparse.ArgumentParser()
    # parser.add_argument('--name', nargs='+')
    parser.add_argument('--name')
    # parser.add_argument('--refine', action='store_false', default='True')

    args = parser.parse_args()
    path = './'
    filename = args.name

    with open(path+filename+'.log', mode='r') as f:
        lines = f.readlines()

    info = dict()
    for line in lines:
        x = line.strip()
        if x.startswith('###'):
            break
        # Simulation information
        name, val = x.split(': ')
        info[name] = val

    for num, line in enumerate(lines):
        x = line.strip()
        if x.startswith('Adj. Targets it'):
            idx_start = num
            break

    N_iter = int(info['iterations'])
    print(info)
    columns = list(measures.values())
    N_measures = int(len(columns)/2)
    results = pd.DataFrame(np.full(fill_value=np.nan, shape=(N_iter, len(columns))), columns=columns)

    for it in range(N_iter):
        for i in range(N_measures):
            line = lines[idx_start + it*N_measures + i]
            x = line.strip()
            name, val = x.split(': ')
            mean, std = [float(st.strip('()')) for st in val.split(' ')]
            results[columns[i]].iloc[it] = mean
            results[columns[i+N_measures]].iloc[it] = std

    # Drop rows that contain only Nan values.
    results = results.dropna(axis=0, how='all')
    out_name = path + filename

    out_df = pd.DataFrame(results)
    # out_df.to_excel(path+filename+'.xlsx', header=True, index=False)
    # print(out_df)
    print(f'Saving results to: {out_name}')
    out_df.to_csv(out_name, header=True, index=False, sep=',', decimal='.')


if __name__ == '__main__':
    analyze_results()
