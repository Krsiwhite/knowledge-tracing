import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn import metrics

from pkg.model import BaseModel

logger: logging.Logger = logging.getLogger(__name__)


# TODO: Fix hardcoded seq_len of window_len. Is this even necessary? Only be consistent with pykt...
def eval(
    model: BaseModel,
    test_file: Path,
    max_concepts: int,
    swap_q_and_c: bool = False,
    batch_size: int = 100,
    window_len: int = 200,
):
    device = list(model.parameters())[0].device

    df = pd.read_csv(test_file)

    ### construct test dataset
    complete_tensors = []
    complete_selectmasks = []
    logger.info(f"{len(df)=}")
    swap_q_and_c_logged = False
    for num_row, row in df.iterrows():
        if num_row % 1000 == 0:
            logger.info(
                f"Constructing test tensors on the fly progress: {num_row/len(df)*100:.2f}%"
            )

        # get
        responses = row["responses"].split(",")
        seq_len = len(responses)
        questions = (
            ["0"] * seq_len if "questions" not in row else row["questions"].split(",")
        )
        concepts = row["concepts"].split(",")
        selectmasks = row["selectmasks"].split(",")
        assert len(questions) == len(concepts) == len(selectmasks) == seq_len

        # Make dense_concept_tensor
        # Add concepts of individual questions to separate lists
        list_of_concept_lists = [[int(x) for x in c.split("_")] for c in concepts]
        # Delete duplicate concepts
        list_of_concept_lists = [list(set(l)) for l in list_of_concept_lists]
        # Pad each list to equal max_concepts in length
        for l in list_of_concept_lists:
            if len(l) > max_concepts:
                raise ValueError("Something wrong with max_concepts")
            l += [-1] * (max_concepts - len(l))
        dense_concept_tensor = torch.tensor(list_of_concept_lists)

        # cast
        c = dense_concept_tensor
        q = torch.tensor([int(s) for s in questions])
        r = torch.tensor([int(s) for s in responses])
        selectmasks = torch.tensor([int(s) for s in selectmasks])

        # swap if applicable
        if swap_q_and_c:
            assert q.unsqueeze(-1).shape == c.shape
            assert model.max_concepts == 1
            assert "_" not in row["concepts"]
            if not swap_q_and_c_logged:
                logger.info("Attention: swapping `concepts` and `questions`")
                swap_q_and_c_logged = True
            c, q = q.unsqueeze(-1), c.squeeze()  # TODO: check with appropriate dataset

        # Initialize new tensor and fill with relevant information
        complete_tensor = -1 * torch.ones(
            [seq_len, 1 + max_concepts + 1],
            dtype=torch.int64,
        )
        complete_tensor[:, :max_concepts] = c
        complete_tensor[:, -2] = q
        complete_tensor[:, -1] = r

        complete_tensors.append(complete_tensor)
        complete_selectmasks.append(selectmasks)

    tensor_dataset = torch.stack(complete_tensors, dim=0)
    tensor_selectmasks = torch.stack(complete_selectmasks, dim=0)

    ### run eval ###
    y_trues_, y_scores_ = [], []
    tensor_selectmasks[:, 0] = (
        -1
    )  # always skip first prediction (in line with what pykt does)
    tensor_selectmasks = tensor_selectmasks == 1  # to bool

    dataset_batches = torch.split(
        tensor_dataset, split_size_or_sections=batch_size, dim=0
    )
    selectmasks_batches = torch.split(
        tensor_selectmasks, split_size_or_sections=batch_size, dim=0
    )
    logger.info(f"{len(dataset_batches)=}")
    for iteration, (batch, selectmasks_batch) in enumerate(
        zip(dataset_batches, selectmasks_batches)
    ):
        if iteration % 10 == 0:
            logger.info(f"{iteration=}")
        batch = batch.to(device)
        selectmasks_batch = selectmasks_batch.to(device)

        # through model
        mask = ~(batch == -1).all(dim=-1)
        y_pred_, *num_row_in_construction = model.forward(data=batch, padding_mask=mask)
        y_pred_, y_true_ = y_pred_, batch[:, :, -1]

        # subsetting a la pykt
        assert (selectmasks_batch == selectmasks_batch & mask).all()
        y_pred_ = torch.masked_select(y_pred_, selectmasks_batch)
        y_true_ = torch.masked_select(y_true_, selectmasks_batch)

        y_trues_.append(y_true_.detach().cpu().numpy())
        y_scores_.append(y_pred_.detach().cpu().numpy())

    y_trues_ = np.concatenate(y_trues_, axis=0)
    y_scores_ = np.concatenate(y_scores_, axis=0)
    y_preds_ = [1 if p >= 0.5 else 0 for p in y_scores_]

    auc = metrics.roc_auc_score(y_true=y_trues_, y_score=y_scores_)
    acc = metrics.accuracy_score(y_true=y_trues_, y_pred=y_preds_)

    num_evaluated_questions = len(y_preds_)

    return auc, acc, num_evaluated_questions
