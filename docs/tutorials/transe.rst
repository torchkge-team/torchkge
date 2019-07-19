===============
TransE Tutorial
===============

To run TransE on FB15k::

    import pandas as pd
    import torch.cuda as cuda

    from time import time

    from torch.utils.data import DataLoader
    from torch.optim import SGD

    from torchkge.data import KnowledgeGraph
    from torchkge.models import TransEModel
    from torchkge.sampling import BernoulliNegativeSampler
    from torchkge.evaluation import LinkPredictionEvaluator, TripletClassificationEvaluator
    from torchkge.utils import l2_dissimilarity, MarginLoss

    #############################################################################################
    # Data loading
    #############################################################################################
    df1 = pd.read_csv('datasets/FB15K/freebase_mtr100_mte100-train.txt',
                      sep='\t', header=None, names=['from', 'rel', 'to'])
    df2 = pd.read_csv('datasets/FB15K/freebase_mtr100_mte100-valid.txt',
                      sep='\t', header=None, names=['from', 'rel', 'to'])
    df3 = pd.read_csv('datasets/FB15K/freebase_mtr100_mte100-test.txt',
                      sep='\t', header=None, names=['from', 'rel', 'to'])
    df = pd.concat([df1, df2, df3])

    kg = KnowledgeGraph(df)
    kg_train, kg_test = kg.split_kg(sizes=(len(df1), len(df2), len(df3))

    #############################################################################################
    # Model definition
    #############################################################################################
    lr, nb_epochs, batch_size, margin = 0.01, 50, 500, 1
    ent_emb_dim = 50
    n_ent = kg_train.n_ent
    n_rel = kg_train.n_rel

    model = TransEModel(ent_emb_dim, n_entities, n_relations, dissimilarity=l2_dissimilarity)
    criterion = MarginLoss(margin)
    optimizer = SGD(model.parameters(), lr=lr)

    #############################################################################################
    # CUDA
    #############################################################################################
    if cuda.is_available():
        model.cuda()
        criterion.cuda()

    cuda.empty_cache()

    #############################################################################################
    # Begin of training
    #############################################################################################
    dataloader = DataLoader(kg_train, batch_size=batch_size, shuffle=False,
                            pin_memory=cuda.is_available())
    sampler = BernoulliNegativeSampler(kg_train, kg_test=kg_test)

    for epoch in range(nb_epochs):

        epoch_time = time()
        first_loss, current_loss = 0, 0
        for i, batch in enumerate(dataloader):
            # get the input
            heads, tails, rels = batch[0], batch[1], batch[2]
            if heads.is_pinned():
                heads, tails, rels = heads.cuda(), tails.cuda(), rels.cuda()

            # Create Negative Samples
            neg_heads, neg_tails = sampler..corrupt_batch(heads, tails, rels)

            # zero model gradient
            model.zero_grad()

            # forward + backward + optimize
            output = model(heads, tails, neg_heads, neg_tails, rels)
            loss = criterion(output)
            loss.backward()
            optimizer.step()

            current_loss = loss.item()
            if i == 0:
                first_loss = current_loss

        model.normalize_parameters()
        print('Epoch {} loss: {} to {} (duration: {}s)'.format(epoch + 1,
                                                                 first_loss, current_loss,
                                                                 time() - epoch_time))

    #############################################################################################
    # Evaluate model
    #############################################################################################
    b_size_eval = 10
    link_evaluator = LinkPredictionEvaluator(model, kg_test)
    triplet_evaluator = TripletClassificationEvaluator(model, kg_val, kg_test)

    link_evaluator.evaluate(batch_size=b_size_eval, k_max=10)
    print('Hit@{}: {}'.format(10, link_evaluator.hit_at_k(k=10)))
    print('Mean Rank: {}'.format(link_evaluator.mean_rank()))

    triplet_evaluator.evaluate(b_size_eval)
    print('Accuracy: {}'.format(triplet_evaluator.accuracy(b_size_eval)))
