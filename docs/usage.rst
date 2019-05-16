=====
Usage
=====

To use TorchKGE in a project::

    import pandas as pd
    import torch.cuda as cuda

    from time import time

    from torch.utils.data import DataLoader
    from torch.optim import SGD

    from torchkge.data import KnowledgeGraph
    from torchkge.data.utils import corrupt_batch
    from torchkge.evaluation import l2_dissimilarity, LinkPredictionEvaluator
    from torchkge.utils import Config
    from torchkge.models import TransEModel, MarginLoss


    #############################################################################################
    # Data loading
    #############################################################################################
    df1 = pd.read_csv('datasets/FB15K/freebase_mtr100_mte100-train.txt',
                      sep='\t', header=None, names=['from', 'rel', 'to'])
    df2 = pd.read_csv('datasets/FB15K/freebase_mtr100_mte100-test.txt',
                      sep='\t', header=None, names=['from', 'rel', 'to'])
    df = pd.concat([df1, df2])

    kg = KnowledgeGraph(df)
    kg_train, kg_test = kg.split_kg(train_size=len(df1))

    #############################################################################################
    # Model definition
    #############################################################################################
    lr, nb_epochs, batch_size, margin = 0.01, 50, 500, 1
    config = Config(ent_emb_dim=50, rel_emb_dim=50,
                    n_ent=kg_train.n_ent, n_rel=kg_train.n_rel, norm_type=2)

    model, criterion = TransEModel(config, dissimilarity=l2_dissimilarity), MarginLoss(margin)
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

    for epoch in range(nb_epochs):

        epoch_time = time()
        first_loss, current_loss = 0, 0
        for i, batch in enumerate(dataloader):
            # get the input
            heads, tails, rels = batch[0], batch[1], batch[2]
            if heads.is_pinned():
                heads, tails, rels = heads.cuda(), tails.cuda(), rels.cuda()

            # Create Negative Samples
            neg_heads, neg_tails = corrupt_batch(heads, tails, n_ent=kg.n_ent)

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
        print('Epoch {} loss : {} to {} (duration : {}s)'.format(epoch + 1,
                                                                 first_loss, current_loss,
                                                                 time() - epoch_time))

    #############################################################################################
    # Evaluate model
    #############################################################################################
    b_size_eval = 10
    train_evaluator = LinkPredictionEvaluator(model, kg_train)
    test_evaluator = LinkPredictionEvaluator(model, kg_test)

    train_evaluator.evaluate(batch_size=b_size_eval, k_max=30)
    print('Hit@{} : {}'.format(10, train_evaluator.hit_at_k(k=10)))
    print('Mean Rank : {}'.format(train_evaluator.mean_rank()))

    test_evaluator.evaluate(batch_size=b_size_eval, k_max=30)
    print('Hit@{} : {}'.format(10, test_evaluator.hit_at_k(k=10)))
    print('Mean Rank : {}'.format(test_evaluator.mean_rank()))
