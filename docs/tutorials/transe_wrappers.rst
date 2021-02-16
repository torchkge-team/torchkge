=================
Shortest training
=================

TorchKGE also provides simple utility wrappers for model training. Here is an example on how to use them::

    from torch.optim import Adam

    from torchkge.evaluation import LinkPredictionEvaluator
    from torchkge.models import TransEModel
    from torchkge.utils.datasets import load_fb15k
    from torchkge.utils import Trainer, MarginLoss


    def main():
        # Define some hyper-parameters for training
        emb_dim = 100
        lr = 0.0004
        margin = 0.5
        n_epochs = 1000
        batch_size = 32768

        # Load dataset
        kg_train, kg_val, kg_test = load_fb15k()

        # Define the model and criterion
        model = TransEModel(emb_dim, kg_train.n_ent, kg_train.n_rel,
                            dissimilarity_type='L2')
        criterion = MarginLoss(margin)
        optimizer = Adam(model.parameters(), lr=lr, weight_decay=1e-5)

        trainer = Trainer(model, criterion, kg_train, n_epochs, batch_size,
                          optimizer=optimizer, sampling_type='bern', use_cuda='all',)

        trainer.run()

        evaluator = LinkPredictionEvaluator(model, kg_test)
        evaluator.evaluate(200)
        evaluator.print_results()


    if __name__ == "__main__":
        main()

