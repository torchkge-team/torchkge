======================
Triplet Classification
======================

To evaluate a model on triplet classification::

    from torch import cuda
    from torchkge.evaluation import TripletClassificationEvaluator
    from torchkge.utils.pretrained_models import load_pretrained_transe
    from torchkge.utils.datasets import load_fb15k

    _, kg_val, kg_test = load_fb15k()

    model = load_pretrained_transe('fb15k', 100):
    if cuda.is_available():
        model.cuda()

    # Triplet classification evaluation on test set by learning thresholds on validation set
    evaluator = TripletClassificationEvaluator(model, kg_val, kg_test)
    evaluator.evaluate(b_size=128)

    print('Accuracy on test set: {}'.format(evaluator.accuracy(b_size=128)))

