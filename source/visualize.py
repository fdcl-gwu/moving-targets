import pandas as pd
import numpy as np
import argparse
import matplotlib.pyplot as plt

measures = {
    'Adj. Targets it {}, train set cost_DIDI': 'DIDI tr OPT',
    'Adj. Targets it {}, test set cost_MSE': 'MSE tr OPT',
    'Adj. Targets it {}, test set cost_R2': 'R2 tr OPT',
    'Trained model it {}, train set cost_DIDI perc. index': 'DIDI tr',
    'Trained model it {}, test set cost_DIDI perc. index': 'DIDI ts',
    'Trained model it {}, train set score_MSE': 'MSE tr',
    'Trained model it {}, train set score_R2': 'R2 tr',
    'Trained model it {}, test set score_MSE': 'MSE ts',
    'Trained model it {}, test set score_R2': 'R2 ts',

    'Adj. Targets it {}, train set cost_DIDI Std': 'Std DIDI tr OPT',
    'Adj. Targets it {}, test set cost_MSE Std': 'Std MSE tr OPT',
    'Adj. Targets it {}, test set cost_R2 Std': 'Std R2 tr OPT',
    'Trained model it {}, train set cost_DIDI perc. index Std': 'Std DIDI tr',
    'Trained model it {}, test set cost_DIDI perc. index Std': 'Std DIDI ts',
    'Trained model it {}, train set score_MSE Std': 'Std MSE tr',
    'Trained model it {}, train set score_R2 Std': 'Std R2 tr',
    'Trained model it {}, test set score_MSE Std': 'Std MSE ts',
    'Trained model it {}, test set score_R2 Std': 'Std R2 ts',
}


def analyze_results():

    parser = argparse.ArgumentParser()
    parser.add_argument('--name', nargs='+')
    # parser.add_argument('--name')

    args = parser.parse_args()
    path = './'
    N_files = len(args.name)
    # data = dict.fromkeys(range(N_files))
    data = dict.fromkeys(args.name)
    info = dict.fromkeys(args.name)

    for filename in args.name:
        with open(path+filename, mode='r') as f:
            lines = f.readlines()

        inf = dict()
        for line in lines:
            x = line.strip()
            if x.startswith('###'):
                break
            # Simulation information
            name, val = x.split(': ')
            inf[name] = val

        for num, line in enumerate(lines):
            x = line.strip()
            if x.startswith('Adj. Targets it'):
                idx_start = num
                break

        N_iter = int(inf['iterations'])
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
        print('\n')
        print('The results for the final 2 iterations of {} with these parameters,'.format(filename))
        print([(key, inf[key]) for key in ['dataset','loss','alpha']])
        print(results[[columns[col] for col in [3,4,6,8,12,13,15,17]]].iloc[-2:])
        data[filename] = results
        info[filename] = inf
        out_df = pd.DataFrame(results)

        # # Save data
        # out_name = path + filename
        # print(f'Saving results to: {out_name}')
        # out_df.to_csv(out_name, header=True, index=False, sep=',', decimal='.')

    # Comparison between two results
    plt.rcParams['font.size'] = 14
    plt.style.use('seaborn-darkgrid')
    if N_files == 1:
        def plot_measure(R2, ylims='auto'):
            plt.figure()
            Std = 'Std ' + R2
            i_file = 0
            results = data[args.name[i_file]]
            inf = info[args.name[i_file]]
            plt.errorbar(np.array(range(N_iter)), results[R2].to_numpy(),
                    yerr=results[Std].to_numpy(), fmt='-o', c='b',
                    label=inf['algo']+' '+inf['alpha'], ls='--', capsize=5)
            plt.legend()
            plt.xlabel('Iterations')
            plt.ylabel(R2 + ' (with Std)')
            plt.title([(key, inf[key]) for key in ['dataset','loss']])
            # plt.ylim(ylims)
            # plt.show()
            plt.savefig(R2.replace(' ','_')+'_'+inf['dataset'][:2]+'_'+
                    inf['loss']+'_'+inf['alpha'][:3]+'.png')
            plt.close()

        for meas in ['R2 tr', 'DIDI tr', 'R2 ts']:
            plot_measure(meas)

    elif N_files == 2:
        def plot_measure(R2, ylims='auto'):
            plt.figure()
            Std = 'Std ' + R2
            i_file = 0
            results = data[args.name[i_file]]
            inf = info[args.name[i_file]]
            plt.errorbar(np.array(range(N_iter)), results[R2].to_numpy(),
                    yerr=results[Std].to_numpy(), fmt='-o', c='r',
                    label=inf['algo']+' '+inf['alpha'], ls='--', capsize=5)
            i_file = 1
            results = data[args.name[i_file]]
            inf = info[args.name[i_file]]
            x_gap = 0.25
            plt.errorbar(np.array(range(N_iter))+x_gap, results[R2].to_numpy(),
                    yerr=results[Std].to_numpy(), fmt='-o', c='b',
                    label=inf['algo']+' '+inf['alpha'], ls='--', capsize=5)
            plt.legend()
            plt.xlabel('Iterations')
            plt.ylabel(R2 + ' (with Std)')
            plt.title([(key, inf[key]) for key in ['dataset','loss']])
            # plt.ylim(ylims)
            # plt.show()
            plt.savefig(R2.replace(' ','_')+'_'+inf['dataset'][:2]+'_'+
                    inf['loss']+'_'+inf['alpha'][:3]+'.png')
            plt.close()

        for meas in ['R2 tr', 'DIDI tr', 'R2 ts']:
            plot_measure(meas)

if __name__ == '__main__':
    analyze_results()
