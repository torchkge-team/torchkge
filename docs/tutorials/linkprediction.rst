===============
Link Prediction
===============

To evaluate a model on link prediction::

    from torch import cuda
    from torchkge.utils.pretrained_models import load_pretrained_transe
    from torchkge.utils.datasets import load_fb15k
    from torchkge.evaluation import LinkPredictionEvaluator

    _, _, kg_test = load_fb15k()

    model = load_pretrained_transe('fb15k', 100):
    if cuda.is_available():
        model.cuda()

    # Link prediction evaluation on test set.
    evaluator = LinkPredictionEvaluator(model, kg_test)
    evaluator.evaluate(b_size=32)
    evaluator.print_results()

