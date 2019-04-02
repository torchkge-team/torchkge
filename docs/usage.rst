=====
Usage
=====

To use TorchKGE in a project::

    import pandas as pd
    from torch import cuda
    from torch.utils.data import DataLoader
    from torch.optim import SGD

    from torchkge.data.KnowledgeGraph import KnowledgeGraph
    from torchkge.utils import Config
    from torchkge.models.TranslationModels import TransEModel
    from torchkge.models.Losses import MarginLoss
    from torchkge.evaluation.Dissimilarities import l2_dissimilarity
    from torchkge.evaluation.LinkPrediction import LinkPredictionEvaluator

    # import data
    df_train = pd.read_csv('../datasets/FB15K/train2id.txt',
                           sep=' ', header=0, names=['from', 'to', 'rel'])
    df_test = pd.read_csv('../datasets/FB15K/test2id.txt',
                          sep=' ', header=0, names=['from', 'to', 'rel'])
    kg_train = KnowledgeGraph(df_train)
    kg_test = KnowledgeGraph(df_test, ent2ix=kg_train.ent2ix, rel2ix=kg_train.rel2ix)

    # define parameters of problem
    lr, nb_epochs, batch_size, margin = 0.01, 50, 500, 1
    config = Config(ent_emb_dim=50, rel_emb_dim=50,
                    n_ent=kg_train.n_ent, n_rel=kg_train.n_rel,
                    norm_type=2)

    # define model and optimizer
    model, criterion = TransEModel(config, dissimilarity=l2_dissimilarity), MarginLoss(margin)
    optimizer = SGD(model.parameters(), lr=lr)

    # move to CUDA
    if cuda.is_available():
        use_cuda = True
        model.cuda()
        criterion.cuda()
        kg_train.cuda()
     else:
     use_cuda = False

    dataloader = DataLoader(kg_train, batch_size=batch_size, shuffle=False)
    for epoch in range(nb_epochs):

        running_loss = 0.0

        for i, batch in enumerate(dataloader):
            # get the input
            heads, tails, rels = batch[0], batch[1], batch[2]

            if use_cuda:
                heads, tails, rels = heads.cuda(), tails.cuda(), rels.cuda()

            neg_heads, neg_tails = kg_train.corrupt_batch(heads, tails)

            # zero model gradient
            model.zero_grad()

            # forward + backward + optimize
            output = model(heads, tails, neg_heads, neg_tails, rels)
            loss = criterion(output)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()


            print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss))
            running_loss = 0.0
        model.normalize_parameters()

    # print train performance
    evaluator = LinkPredictionEvaluator(model.entity_embeddings,
                                        model.relation_embeddings,
                                        l2_dissimilarity, kg_train)
    if use_cuda:
        evaluator.cuda()
    evaluator.evaluate(batch_size=100, k_max=50)
    print('Hit@{} : {}'.format(10, evaluator.hit_at_k(k=10)))
    print('Mean Rank : {}'.format(evaluator.mean_rank()))

    # print test performance
    evaluator = LinkPredictionEvaluator(model.entity_embeddings,
                                        model.relation_embeddings,
                                        l2_dissimilarity, kg_test)
    if use_cuda:
        evaluator.cuda()
    evaluator.evaluate(batch_size=100, k_max=50)
    print('Hit@{} : {}'.format(10, evaluator.hit_at_k(k=10)))
    print('Mean Rank : {}'.format(evaluator.mean_rank()))


