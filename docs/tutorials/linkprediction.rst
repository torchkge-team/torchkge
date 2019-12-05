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
    evaluator.evaluate(batch_size=32, k_max=10)

    print('Hit@{} : {}'.format(10, evaluator.hit_at_k(k=10)[0]))
    print('Mean Rank : {}'.format(evaluator.mean_rank()[0]))
    print('MRR : {}'.format(evaluator.mrr()[0]))
    print('Filt. Hit@{} : {}'.format(10, evaluator.hit_at_k(k=10)[1]))
    print('Filt. Mean Rank : {}'.format(evaluator.mean_rank()[1]))
    print('Filt. MRR : {}'.format(evaluator.mrr()[1]))
