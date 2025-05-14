import pandas as pd
from sklearn.model_selection import train_test_split
import os

def main():
    book = pd.ExcelFile('labels/ODIR-5K_Training_Annotations(Updated)_V2.xlsx')
    patients = book.parse(book.sheet_names[0], index_col=0)
    print(patients)

    print('Age', patients['Patient Age'].describe().to_dict())
    print('Sex', patients.groupby('Patient Sex').size().to_dict())

    df = pd.read_csv('labels/eye_labels.csv', sep=',')
    del df['Total']

    train, test = train_test_split(df, train_size=0.9)

    train.to_csv('labels/train.csv', sep=',', index=False)
    test.to_csv('labels/test.csv', sep=',', index=False)

    stats = pd.concat([train.loc[:, 'Normal':'Others'].sum(axis=0), test.loc[:, 'Normal':'Others'].sum(axis=0)], axis=1)
    stats.columns = ['train', 'test']
    stats['train+test'] = stats['train'] + stats['test']
    stats['%'] = stats['train+test'] / stats['train+test'].sum(axis=0) * 100
    stats.loc['Total', :] = stats.sum(axis=0)

    print(stats.round(1))

if __name__ == '__main__':
    main()
