import pandas as pd
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datafile",
                        type=str,
                        default=None)
    args = parser.parse_args()
    datafile_name = args.datafile

    datapath = './data/embeddings/' + datafile_name + '.pkl'
    df = pd.read_pickle(datapath)
    savepath = './data/embeddings/' + datafile_name + '.csv'
    df.to_csv(savepath, index=None)

if __name__ == '__main__':
    main()