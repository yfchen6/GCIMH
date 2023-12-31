import torch


def mean_average_precision(query_code,
                           retrieval_code,
                           query_targets,
                           retrieval_targets,
                           device,
                           topk=None
                           ):
    """
    Calculate mean average precision(map).

    Args:
        query_code (torch.Tensor): Query data hash code.
        retrieval_code (torch.Tensor): Retrieval data hash code.
        query_targets (torch.Tensor): Query data targets, one-hot
        retrieval_targets (torch.Tensor): retrieval data targets, one-hot
        device (torch.device): Using CPU or GPU.
        topk: int

    Returns:
        meanAP (float): Mean Average Precision.
    """
    query_code = query_code.to(device)
    retrieval_code = retrieval_code.to(device)
    query_targets = query_targets.to(device)
    retrieval_targets = retrieval_targets.to(device)
    num_query = query_targets.shape[0]
    if topk == None:
        topk = retrieval_targets.shape[0]
    mean_AP = 0.0

    for i in range(num_query):
        # Retrieve images from database
        retrieval = (query_targets[i, :] @ retrieval_targets.t() > 0).float()

        # Calculate hamming distance
        hamming_dist = 0.5 * (retrieval_code.shape[1] - query_code[i, :] @ retrieval_code.t())

        # Arrange position according to hamming distance
        retrieval = retrieval[torch.argsort(hamming_dist)][:topk]

        # Retrieval count
        retrieval_cnt = retrieval.sum().int().item()

        # Can not retrieve images
        if retrieval_cnt == 0:
            continue

        # Generate score for every position
        score = torch.linspace(1, retrieval_cnt, retrieval_cnt).to(device)  # 返回一个1维张量，包含在区间start和end上均匀间隔的step个点。
        # score = torch.linspace(1, retrieval_cnt, retrieval_cnt)  # 返回一个1维张量，包含在区间start和end上均匀间隔的step个点。

        # Acquire index
        index = (torch.nonzero(retrieval).squeeze() + 1.0).float()

        mean_AP += (score / index).mean()

    mean_AP = mean_AP / num_query
    return mean_AP.item()

