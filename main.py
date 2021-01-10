from Dataloader.Dataloader import Dataloader
from utils.config import Config
from model.sklearn_model import Sklearn
import argparse


def main(args):
    # create an instance of configuration file
    cfg = Config.from_yaml(args.config)

    # Preprocess and feature extraction
    cl_instance = Dataloader(cfg.config["data"]['labels_metadata'], cfg.config["data"]['path_main'],
                             cfg.config["data"]['outpath'])

    if cfg.config["train"]["load_feats"]:
        cl_instance.load_features_from_disk()
    elif not cfg.config["train"]["load_feats"]:
        cl_instance.load_data(save2disk=cfg.config["data"]['save2disk'])
        cl_instance.feature_extraction(feature_type=cfg.config["train"]["feature_type"],
                                       pooling=cfg.config["train"]["pooling"],
                                       save_local=cfg.config["train"]["save_feats"])

    x_train, x_test, y_train, y_test = cl_instance.preprocess_data(split=cfg.config["train"]["split"],
                                                                   normalize=cfg.config["train"]["normalize"],
                                                                   test_size=cfg.config["train"]["test_size"],
                                                                   encoder=cfg.config["train"]["encoder"])

    # Train and eval
    model = Sklearn(cfg.config, x_train, x_test, y_train, y_test)
    if cfg.config["train"]["train"]:
        model.train(save2disk=cfg.config["train"]["save2disk"])
        if cfg.config["train"]["split"]:
            model.evaluate(load_model=cfg.config["evaluation"]["load_model"])

    # eval set
    if cfg.config["evaluation"]["eval"]:
        ev_instance = Dataloader(cfg.config["data"]['labels_metadata'], cfg.config["data"]['eval_set'],
                                 cfg.config["data"]['outpath'])
        ev_instance.load_data(save2disk=cfg.config["data"]['save2disk'])
        ev_instance.feature_extraction(feature_type=cfg.config["train"]["feature_type"],
                                       pooling=cfg.config["train"]["pooling"])
        x_test, _, y_test, _ = ev_instance.preprocess_data(split=False,
                                                           normalize=cfg.config["train"]["normalize"],
                                                           encoder=cfg.config["train"]["encoder"])

        model.x_test = x_test
        model.y_test = y_test
        model.evaluate(load_model=cfg.config["evaluation"]["load_model"])


if __name__ == '__main__':
    # Initiate the parser
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, help="path to yaml file")

    # Read arguments from the command line
    args = parser.parse_args()
    main(args)
