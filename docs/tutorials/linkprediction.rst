===============
Link Prediction
===============

To evaluate a model on link prediction::

    from torchkge.data.DataLoader import load_fb15k
    from torchkge.evaluation import LinkPredictionEvaluator

    _, _, kg_test = load_fb15k()

    # Assume the variable `model` was trained on the training subset of FB15k
    global model

    # Link prediction evaluation on test set.
    evaluator = LinkPredictionEvaluator(model, kg_test)
    evaluator.evaluate(batch_size=512, k_max=10)
    evaluator.evaluate(k=10)
    evaluator.print_results()
