====
HolE
====

To train HolE on FB15k::

    from torch.optim import Adam
    from torch import cuda

    from torchkge.data.DataLoader import load_fb15k
    from torchkge.evaluation import LinkPredictionEvaluator
    from torchkge.models import HolEModel
    from torchkge.sampling import BernoulliNegativeSampler
    from torchkge.utils import LogisticLoss, DataLoader

    from tqdm.autonotebook import tqdm


    # Load dataset
    kg_train, kg_val, kg_test = load_fb15k()

    # Define some hyper-parameters for training
    lr = 0.001
    n_ent, n_rel = kg_train.n_ent, kg_train.n_rel
    ent_emb_dim = 150
    nb_epochs = 50
    batch_size = 16384

    # Define the model and criterion
    model = HolEModel(ent_emb_dim, n_ent, n_rel)
    criterion = LogisticLoss()
    sampler = BernoulliNegativeSampler(kg_train, kg_test=kg_test)

    # Move everything to CUDA if available
    if cuda.is_available():
        cuda.set_device(0)
        cuda.empty_cache()
        model.cuda()
        criterion.cuda()
        cuda_val = 'all'
    else:
        cuda_val = None

    # Define the torch optimizer to be used
    optimizer = Adam(model.parameters(), lr=lr, weight_decay=1e-5)

    dataloader = DataLoader(kg_train, batch_size=batch_size, use_cuda=cuda_val)

    iterator = tqdm(range(nb_epochs), unit='epoch')
    for epoch in iterator:
        running_loss = 0.0

        for i, batch in enumerate(get_batches(h, t, r, batch_size)):
            # get the input
            heads, tails, rels = batch[0], batch[1], batch[2]
            neg_heads, neg_tails = sampler.corrupt_batch(heads, tails, rels)

            # zero model gradient
            model.zero_grad()

            # forward + backward + optimize
            positive_triplets, negative_triplets = model(heads, tails, neg_heads, neg_tails, rels)
            loss = criterion(positive_triplets, negative_triplets)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        iterator.set_description('Epoch mean loss: {:.5f}'.format(running_loss / len(kg_train)))

    model.normalize_parameters()
