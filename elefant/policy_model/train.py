"""Training entry: initialization and resume must now be explicit."""
import argparse
import logging


def lightning_main():
    from elefant.integration.training import add_transfer_arguments, run_transfer
    logging.basicConfig(level=logging.INFO, force=True)
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--fast_dev_run", action="store_true")
    parser.add_argument("--no_compile", action="store_true")
    parser.add_argument("--data_folder", default=None)
    add_transfer_arguments(parser)
    args = parser.parse_args()
    from elefant.policy_model.config import LightningPolicyConfig
    from elefant.config import load_config
    from elefant.torch import pytorch_setup
    config = load_config(args.config, LightningPolicyConfig)
    config.shared.fast_dev_run = args.fast_dev_run
    if args.data_folder is not None:
        config.stage3_finetune.training_dataset.local_prefix = args.data_folder
        for dataset in config.stage3_finetune.validation_datasets:
            dataset.local_prefix = args.data_folder
    pytorch_setup()
    run_transfer(config, args)


if __name__ == "__main__":
    lightning_main()
