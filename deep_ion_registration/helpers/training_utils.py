import pickle
import pandas as pd


class TrainingLogger:
    def __init__(self, verbose=False):
        self.train_loss = {}
        self.val_loss = {}
        self.train_examples = {}
        self.val_examples = {}
        self.verbose = verbose

    def log(self, epoch, **kwargs):
        for key, value in kwargs.items():
            if key in self.__dict__:
                if key.endswith('examples'):
                    self.__dict__[key][epoch] = self.prepare_example(value)
                else:
                    self.__dict__[key][epoch] = value
                    if self.verbose:
                        print('\n' + '-' * 10)
                        print(f'{key}: {value}')

    @staticmethod
    def prepare_example(examples):
        return [example.detach().cpu().numpy() for example in examples]

    def save(self, path):
        with open(path, 'wb') as f:
            pickle.dump(self, f)

    def to_dataframe(self):
        df = self.nested_dict_to_dataframe(self.__dict__)
        return df

    @staticmethod
    def nested_dict_to_dataframe(nested_dict):
        df = pd.DataFrame(nested_dict)
        df = df.T
        return df


class EarlyStopping:
    def __init__(self, patience=5):
        self.patience = patience
        self.counter = 0
        self.best_loss = None
        self.early_stop = 'wait'

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss * 10:
            self.early_stop = 'divergence'
        elif val_loss > self.best_loss:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = 'stop'
        else:
            self.best_loss = val_loss
            self.counter = 0

        return self.early_stop


class Saver:
    def __init__(self, path):
        self.path = path

    def save(self, model, logger, identifier=None):
        model.save(self.path + f'/model_{identifier}.pt')
        logger.save(self.path + f'/logger_{identifier}.pkl')



def test_nested_dict_to_dataframe():
    nested_dict = {
        0: {'train_loss': 0.1, 'val_loss': 0.2},
        1: {'train_loss': 0.3, 'val_loss': 0.4},
    }
    df = TrainingLogger.nested_dict_to_dataframe(nested_dict)
    assert df.shape == (2, 2)
    assert df.columns.tolist() == ['train_loss', 'val_loss']
    assert df.index.tolist() == [0, 1]
    assert df.iloc[0].tolist() == [0.1, 0.2]
    assert df.iloc[1].tolist() == [0.3, 0.4]
    print('test_nested_dict_to_dataframe passed')



if __name__ == '__main__':
    test_nested_dict_to_dataframe()
    print('TrainingLogger passed')