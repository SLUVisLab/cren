from setups import BaseSetup


def train(setup: BaseSetup, epoch: int, pooling: str, loader='train') -> float:
    setup.set_train()

    for batch in setup.loaders[loader]:
        setup.run(batch, epoch, pooling)
