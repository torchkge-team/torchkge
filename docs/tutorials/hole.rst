=============
HolE Tutorial
=============

To run HolE on FB15k::

    from torch.optim import SGD
    from torch import cuda
    from torch.utils.data import DataLoader

    from torchkge.data.DataLoader import load_fb15k
    from torchkge.models import HolEModel
    from torchkge.utils import LogisticLoss
    from torchkge.sampling import BernoulliNegativeSampler
    from torchkge.evaluation import LinkPredictionEvaluator, TripletClassificationEvaluator

    # Load dataset
    kg_train, kg_val, kg_test = load_fb15k()

    # Define some hyper-parameters for training
    lr, nb_epochs, batch_size = 0.001, 500, 1024
    n_ent, n_rel = kg_train.n_ent, kg_train.n_rel
    ent_emb_dim = 50

    # Define the model and criterion
    model = HolEModel(ent_emb_dim, n_ent, n_rel)
    criterion = LogisticLoss()

    # Move everything to CUDA is available
    use_cuda = True
    if use_cuda and cuda.is_available():
        model.cuda()
        criterion.cuda()
    cuda.empty_cache()

    # Define the torch optimizer to be used
    optimizer = SGD(model.parameters(), lr=lr)

    # Define the sampler useful for negative sampling during training
    sampler = BernoulliNegativeSampler(kg_train, kg_test=kg_test)

    dataloader = DataLoader(kg_train, batch_size=batch_size, shuffle=False, pin_memory=use_cuda)

    for epoch in range(nb_epochs):
        for i, batch in enumerate(dataloader):
            running_loss = 0.0

            # get the input
            heads, tails, rels = batch[0], batch[1], batch[2]
            if heads.is_pinned():
                heads, tails, rels = heads.cuda(), tails.cuda(), rels.cuda()

            # Create Negative Samples
            neg_heads, neg_tails = sampler.corrupt_batch(heads, tails, rels)
            # zero model gradient
            model.zero_grad()

            # forward + backward + optimize
            output = model(heads, tails, neg_heads, neg_tails, rels)

            loss = criterion(output)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / batch_size))

    model.normalize_parameters()

    # Triplet classification evaluation on test set by learning thresholds on validation set
    evaluator = TripletClassificationEvaluator(model, kg_val, kg_test)
    evaluator.evaluate(100)
    print('Accuracy on test set: {}'.format(evaluator.accuracy(100)))

    # Link prediction evaluation on test set.
    evaluator = LinkPredictionEvaluator(model, kg_test)
    evaluator.evaluate(batch_size=1, k_max=10)
    print('Hit@{} : {}'.format(1, evaluator.hit_at_k(k=10)[0]))
    print('Mean Rank : {}'.format(evaluator.mean_rank()[0]))
    print('MRR : {}'.format(evaluator.mrr()[0]))
    print('Filt. Hit@{} : {}'.format(10, evaluator.hit_at_k(k=10)[1]))
    print('Filt. Mean Rank : {}'.format(evaluator.mean_rank()[1]))
    print('Filt. MRR : {}'.format(evaluator.mrr()[1]))
