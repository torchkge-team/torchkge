===============
Link Prediction
===============

To evaluate a model on link prediction::

    from torch import cuda
    from torchkge.models import TransEModel
    from torchkge.utils.datasets import load_fb15k
    from torchkge.evaluation import LinkPredictionEvaluator

    _, _, kg_test = load_fb15k()

    model = TransEModel(100, pre_trained='fb15k')
    if cuda.is_available():
        model.cuda()

    # Link prediction evaluation on test set.
    evaluator = LinkPredictionEvaluator(model, kg_test)
    evaluator.evaluate(b_size=32, k_max=10)
    evaluator.print_results()

