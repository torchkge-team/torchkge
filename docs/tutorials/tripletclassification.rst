======================
Triplet Classification
======================

To evaluate a model on triplet classification::

    from torchkge.data.DataLoader import load_fb15k
    from torchkge.evaluation import TripletClassificationEvaluator

    _, kg_val, kg_test = load_fb15k()

    # Assume the variable `model` was trained on the training part of FB15k
    global model

    # Triplet classification evaluation on test set by learning thresholds on validation set
    evaluator = TripletClassificationEvaluator(model, kg_val, kg_test)
    evaluator.evaluate(batch_size=128)
    print('Accuracy on test set: {}'.format(evaluator.accuracy(batch_size=128)))
