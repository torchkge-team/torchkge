from torch import Tensor, tensor, bernoulli, ones, randint

from torchkge.utils import concatenate_diff_sizes


def get_dictionaries(df, ent=True):
    """Build entities or relations dictionaries.

    Parameters
    ----------
    df : pandas Dataframe
        Data frame containing three columns [from, to, rel].
    ent : bool
        if True then ent2ix is returned, if False then rel2ix is returned.

    Returns
    -------
    dict : dictionary
        Either ent2ix or rel2ix.
    """
    if ent:
        tmp = list(set(df['from'].unique()).union(set(df['to'].unique())))
        return {ent: i for i, ent in enumerate(tmp)}
    else:
        tmp = list(df['rel'].unique())
        return {rel: i for i, rel in enumerate(tmp)}


def lists_from_dicts(dictionary, entities, relations, targets, cuda):
    """

    Parameters
    ----------
    dictionary : dict
        keys : (ent, rel), values : list of entities
    entities : torch tensor, dtype = long, shape = (batch_size)
        Heads (resp. tails) of facts.
    relations : torch tensor, dtype = long, shape = (batch_size)
        Relations of facts
    targets : torch tensor, dtype = long, shape = (batch_size)
        Tails (resp. heads) of facts.
    cuda : bool
        If True, result is returned as CUDA tensor.

    Returns
    -------
    result : torch tensor, dtype = long, shape = (k)
        k is the largest number of possible alternative to the target in a fact.
        This tensor contains for each line (fact) the list of possible alternatives to the target.
        If there are no alternatives, then the line is full of -1.
    """
    result = Tensor().long()

    if entities.is_cuda:
        result = result.cuda()

    for i in range(entities.shape[0]):
        current = dictionary[(entities[i].item(), relations[i].item())]
        current.remove(targets[i])
        if len(current) == 0:
            current.append(-1)
        current = tensor(current).long().view(1, -1)
        if entities.is_cuda:
            current = current.cuda()
        result = concatenate_diff_sizes(result, current)
    if cuda:
        return result.cuda()
    else:
        return result.cpu()


def corrupt_batch(heads, tails, n_ent):
    """For each golden triplet, produce a corrupted one not different from any other golden triplet.

    Parameters
    ----------
    heads : torch tensor, dtype = long, shape = (batch_size)
        Tensor containing the integer key of heads of the relations in the current batch.
    tails : torch tensor, dtype = long, shape = (batch_size)
        Tensor containing the integer key of tails of the relations in the current batch.
    n_ent : int
        Number of entities in the entire dataset.

    Returns
    -------
    neg_heads : torch tensor, dtype = long, shape = (batch_size)
        Tensor containing the integer key of negatively sampled heads of the relations \
        in the current batch.
    neg_tails : torch tensor, dtype = long, shape = (batch_size)
        Tensor containing the integer key of negatively sampled tails of the relations \
        in the current batch.
    """
    use_cuda = heads.is_cuda
    assert (use_cuda == tails.is_cuda)
    if use_cuda:
        device = 'cuda'
    else:
        device = 'cpu'

    batch_size = heads.shape[0]
    neg_heads, neg_tails = heads.clone(), tails.clone()

    # TODO : implement smarter corruption (cf TransH paper)
    # Randomly choose which samples will have head/tail corrupted
    mask = bernoulli(ones(size=(batch_size,), device=device)/2).double()
    n_heads_corrupted = int(mask.sum().item())
    neg_heads[mask == 1] = randint(1, n_ent, (n_heads_corrupted,), device=device)
    neg_tails[mask == 0] = randint(1, n_ent, (batch_size - n_heads_corrupted,), device=device)

    return neg_heads.long(), neg_tails.long()
