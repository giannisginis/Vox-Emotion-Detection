from Dataloader.Dataloader import Dataloader
from utils.config import Config
from model.sklearn_model import Sklearn
import argparse
from utils.Logger import LogSystem
import datetime


def main(args):
    logfile = f'{datetime.datetime.now()}.txt'

    # create an instance of configuration file
    cfg = Config.from_yaml(args.config)

    # initialize logger
    logger = LogSystem(log_dir=cfg.config["data"]["log_dir"], name=logfile, log_name='generic')
    logger.log_info('Starting Code')

    # Preprocess and feature extraction
    cl_instance = Dataloader(cfg.config["data"]['labels_metadata'], cfg.config["data"]['path_main'],
                             cfg.config["data"]['outpath'], logger)

    if cfg.config["train"]["load_feats"]:
        cl_instance.load_features_from_disk(filename=cfg.config["train"]["feats_filename"])
    elif not cfg.config["train"]["load_feats"]:
        cl_instance.load_data(save2disk=cfg.config["data"]['save2disk'])
        cl_instance.feature_extraction(feature_type=cfg.config["train"]["feature_type"],
                                       pooling=cfg.config["train"]["pooling"],
                                       save_local=cfg.config["train"]["save_feats"],
                                       feats_filename=cfg.config["train"]["feats_filename"])

    x_train, x_test, y_train, y_test = cl_instance.preprocess_data(split=cfg.config["train"]["split"],
                                                                   normalize=cfg.config["train"]["normalize"],
                                                                   test_size=cfg.config["train"]["test_size"],
                                                                   encoder=cfg.config["train"]["encoder"])

    # Train and eval
    model = Sklearn(cfg.config, x_train, x_test, y_train, y_test, logger)
    if cfg.config["train"]["train"]:
        logger.log_info("Training")
        model.train(save2disk=cfg.config["train"]["save2disk"])
        if cfg.config["train"]["split"]:
            logger.log_info("Validation")
            model.evaluate(load_model=cfg.config["evaluation"]["load_model"])

    # eval set
    if cfg.config["evaluation"]["eval"]:
        ev_instance = Dataloader(cfg.config["data"]['labels_metadata'], cfg.config["data"]['eval_set'],
                                 cfg.config["data"]['outpath'], logger=logger)

        if cfg.config["evaluation"]["load_feats"]:
            ev_instance.load_features_from_disk(filename=cfg.config["evaluation"]["feats_filename"])
        elif not cfg.config["evaluation"]["load_feats"]:
            ev_instance.load_data(save2disk=cfg.config["data"]['save2disk'])
            ev_instance.feature_extraction(feature_type=cfg.config["train"]["feature_type"],
                                           pooling=cfg.config["train"]["pooling"],
                                           save_local=cfg.config["evaluation"]["save_feats"],
                                           feats_filename=cfg.config["evaluation"]["feats_filename"])
        x_test, _, y_test, _ = ev_instance.preprocess_data(split=False,
                                                           normalize=cfg.config["train"]["normalize"],
                                                           encoder=cfg.config["train"]["encoder"])

        model.x_test = x_test
        model.y_test = y_test
        logger.log_info("Evaluation")
        model.evaluate(load_model=cfg.config["evaluation"]["load_model"])


if __name__ == '__main__':
    # Initiate the parser
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, help="path to yaml file")

    # Read arguments from the command line
    args = parser.parse_args()
    main(args)
