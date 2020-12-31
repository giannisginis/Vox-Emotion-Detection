from Dataloader.Dataloader import Dataloader
from utils.config import Config
from model.sklearn_model import Sklearn
import argparse


def main(args):
    # create an instance of configuration file
    cfg = Config.from_yaml(args.config)

    # Preprocess and feature extraction
    cl_instance = Dataloader(cfg.data['labels_metadata'], cfg.data['path_main'], cfg.data['outpath'])
    cl_instance.load_data(save2disk=cfg.data['save2disk'])
    cl_instance.feature_extraction(feature_type=cfg.train["feature_type"], pooling=cfg.train["pooling"])
    x_train, x_test, y_train, y_test = cl_instance.preprocess_data()

    # Train and eval
    model = Sklearn(cfg, x_train, x_test, y_train, y_test, "rfc")
    model.train()
    model.predict()
    model.evaluate()


if __name__ == '__main__':
    # Initiate the parser
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, help="path to yaml file")

    # Read arguments from the command line
    args = parser.parse_args()
    main(args)
