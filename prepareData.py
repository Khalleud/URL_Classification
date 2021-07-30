
import os
import glob
import pandas as pd

if __name__ == "__main__":



    os.chdir('./data/')
    extension = 'parquet'
    all_filenames = [i for i in glob.glob('*.{}'.format(extension))]


    df = pd.concat([pd.read_parquet(f) for f in all_filenames ])

    print(df.isna().any())


    df.reset_index(inplace=True, drop = True)

    os.chdir('../')
    df.to_csv('Data.csv')


    print('Saving data to csv...')
