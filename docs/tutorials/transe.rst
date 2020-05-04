======
TransE
======

This is the python code to train TransE without any wrapper. This script shows how all parts of TorchKGE should be used
togeter.::

    from torch import cuda
    from torch.optim import Adam

    from torchkge.data.Datasets import load_fb15k
    from torchkge.models import TransEModel
    from torchkge.sampling import BernoulliNegativeSampler
    from torchkge.utils import MarginLoss, DataLoader

    from tqdm.autonotebook import tqdm

    # Load dataset
    kg_train, _, _ = load_fb15k()

    # Define some hyper-parameters for training
    lr, n_epochs, b_size, margin = 0.01, 500, 1024, 1
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
    optimizer = Adam(model.parameters(), lr=lr)

    # Define the sampler useful for negative sampling during training
    sampler = BernoulliNegativeSampler(kg_train)

    dataloader = DataLoader(kg_train, batch_size=b_size, use_cuda='all')

    iterator = tqdm(range(n_epochs), unit='epoch')
    for epoch in iterator:
        running_loss = 0.0
        for i, batch in enumerate(dataloader):
            h, t, r = batch[0], batch[1], batch[2]
            n_h, n_t = sampler.corrupt_batch(h, t, r)

            optimizer.zero_grad()

            # forward + backward + optimize
            pos, neg = model(h, t, n_h, n_t, r)
            loss = criterion(pos, neg)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
        iterator.set_description(
            'Epoch {} | mean loss: {:.5f}'.format(epoch + 1,
                                                  running_loss / len(dataloader)))

    model.normalize_parameters()
