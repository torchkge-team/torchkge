======
TransE
======

To train TransE on FB15k::

    from torch.optim import SGD
    from torch import cuda
    from torch.utils.data import DataLoader

    from torchkge.data.DataLoader import load_fb15k
    from torchkge.models import TransEModel
    from torchkge.utils import MarginLoss
    from torchkge.sampling import BernoulliNegativeSampler

    # Load dataset
    kg_train, _, _ = load_fb15k()

    # Define some hyper-parameters for training
    lr, nb_epochs, batch_size, margin = 0.01, 500, 1024, 1
    n_ent, n_rel = kg_train.n_ent, kg_train.n_rel
    ent_emb_dim = 50

    # Define the model and criterion
    model = TransEModel(ent_emb_dim, n_ent, n_rel, dissimilarity_type='L2')
    criterion = MarginLoss(margin)

    # Move everything to CUDA if available
    if cuda.is_available():
        cuda.empty_cache()
        model.cuda()
        criterion.cuda()

    # Define the torch optimizer to be used
    optimizer = SGD(model.parameters(), lr=lr)

    # Define the sampler useful for negative sampling during training
    sampler = BernoulliNegativeSampler(kg_train, kg_test=kg_test)

    dataloader = DataLoader(kg_train, batch_size=batch_size, shuffle=False, 
                            pin_memory=cuda.is_available())

    for epoch in range(nb_epochs):
        running_loss = 0.0
        for i, batch in enumerate(dataloader):
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

