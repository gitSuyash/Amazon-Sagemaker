import argparse
import pandas as pd
import os

from sklearn import tree
from sklearn.externals import joblib
from sklearn.model_selection import GridSearchCV


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Hyperparameters are described here. In this simple example we are just including one hyperparameter.
    parser.add_argument('--max_leaf_nodes', type=int, default=15)
    parser.add_argument('--max_depth',type=int,default=5)

    # Sagemaker specific arguments. Defaults are set in the environment variables.
    # parser.add_argument('--output-data-dir', type=str, default=os.environ['SM_OUTPUT_DATA_DIR'])
    # parser.add_argument('--model-dir', type=str, default=os.environ['SM_MODEL_DIR'])
    parser.add_argument('--train', type=str, default=os.environ['SM_CHANNEL_TRAIN'])

    args = parser.parse_args()

    # Take the set of files and read them all into a single pandas dataframe
    input_files = [ os.path.join(args.train, file) for file in os.listdir(args.train) ]
    if len(input_files) == 0:
        raise ValueError(('There are no files in {}.\n' +
                          'This usually indicates that the channel ({}) was incorrectly specified,\n' +
                          'the data specification in S3 was incorrectly specified or the role specified\n' +
                          'does not have permission to access the data.').format(args.train, "train"))
    raw_data = [ pd.read_csv(file, header=None, engine="python") for file in input_files ]
    train_data = pd.concat(raw_data)

    # labels are in the first column
    train_y = train_data.iloc[:,0]
    train_X = train_data.iloc[:,1:]

    #seting hyperparameters
    max_leaf_nodes = args.max_leaf_nodes
    max_depth = args.max_depth

    # Now use scikit-learn's decision tree classifier to train the model.
    clf = tree.DecisionTreeRegressor(max_leaf_nodes=max_leaf_nodes,max_depth=max_depth)
    
    
    param_grid = {
    'max_leaf_nodes':range(24,30),
    'max_depth':range(3,8)
    }
    tuner = GridSearchCV(clf,param_grid=param_grid)
    tuner.fit(train_X.train_y)

    # Print the coefficients of the trained classifier, and save the coefficients
    joblib.dump(tuner, os.path.join(args.model_dir, "model.joblib"))
