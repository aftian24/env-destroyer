# this file was created based on code from various sources, including: axolotl, ...
import torch
import torch.nn.functional as F
import transformers
from typing import Optional
import sys
import random 
from torch.utils.data import Dataset
from typing import Any, Dict, List, Tuple


def get_max_seqlen_in_batch(attention_mask):
    max_num = torch.max(attention_mask)
    # attention_mask: B x N
    counts = []
    for i in range(1, max_num + 1):
        counts.append(
            torch.sum(attention_mask == i, axis=-1)
        )  # shape: B, count length of data point maksed with i
    result = torch.stack(counts, axis=1)
    result = result.flatten()
    return result[result.nonzero()].squeeze(-1).to(dtype=torch.int32)


def get_unpad_data(attention_mask):
    seqlens_in_batch = get_max_seqlen_in_batch(
        attention_mask
    )  # attention_mask.sum(dim=-1, dtype=torch.int32)
    indices = torch.nonzero(attention_mask.flatten(), as_tuple=False).flatten()
    max_seqlen_in_batch = seqlens_in_batch.max().item()
    cu_seqlens = F.pad(
        torch.cumsum(seqlens_in_batch, dim=0, dtype=torch.torch.int32), (1, 0)
    )
    return (
        indices,
        cu_seqlens,
        max_seqlen_in_batch,
    )


def load_balancing_loss_func(
    gate_logits: torch.Tensor,
    num_experts: torch.Tensor = None,
    top_k=2,
    attention_mask: Optional[torch.Tensor] = None,
) -> float:
    if gate_logits is None or not isinstance(gate_logits, tuple):
        return 0

    if isinstance(gate_logits, tuple):
        compute_device = gate_logits[0].device
        concatenated_gate_logits = torch.cat(
            [layer_gate.to(compute_device) for layer_gate in gate_logits], dim=0
        )

    routing_weights = torch.nn.functional.softmax(concatenated_gate_logits, dim=-1)

    _, selected_experts = torch.topk(routing_weights, top_k, dim=-1)

    expert_mask = torch.nn.functional.one_hot(selected_experts, num_experts)

    if attention_mask is None:
        # Compute the percentage of tokens routed to each experts
        tokens_per_expert = torch.mean(expert_mask.float(), dim=0)

        # Compute the average probability of routing to these experts
        router_prob_per_expert = torch.mean(routing_weights, dim=0)
    else:
        # ONLY ADD THIS LINE OF CODE, AND REPLACE attention_mask WITH new_attention_mask
        new_attention_mask = (attention_mask != 0).int().to(attention_mask.device)
        batch_size, sequence_length = new_attention_mask.shape
        num_hidden_layers = concatenated_gate_logits.shape[0] // (
            batch_size * sequence_length
        )

        # Compute the mask that masks all padding tokens as 0 with the same shape of expert_mask
        expert_attention_mask = (
            new_attention_mask[None, :, :, None, None]
            .expand(
                (num_hidden_layers, batch_size, sequence_length, top_k, num_experts)
            )
            .reshape(-1, top_k, num_experts)
            .to(compute_device)
        )

        # Compute the percentage of tokens routed to each experts
        tokens_per_expert = torch.sum(
            expert_mask.float() * expert_attention_mask, dim=0
        ) / torch.sum(expert_attention_mask, dim=0)

        # Compute the mask that masks all padding tokens as 0 with the same shape of tokens_per_expert
        router_per_expert_attention_mask = (
            new_attention_mask[None, :, :, None]
            .expand((num_hidden_layers, batch_size, sequence_length, num_experts))
            .reshape(-1, num_experts)
            .to(compute_device)
        )

        # Compute the average probability of routing to these experts
        router_prob_per_expert = torch.sum(
            routing_weights * router_per_expert_attention_mask, dim=0
        ) / torch.sum(router_per_expert_attention_mask, dim=0)

    overall_loss = torch.sum(tokens_per_expert * router_prob_per_expert.unsqueeze(0))
    return overall_loss * num_experts


def monkey_patch_packing_for_model(pretrained_model):
    model_config = transformers.AutoConfig.from_pretrained(pretrained_model)
    config_type = type(model_config).__name__.lower()
    if hasattr(transformers, "modeling_flash_attention_utils"):
        transformers.modeling_flash_attention_utils._get_unpad_data = get_unpad_data
    if config_type == "mixtralconfig":
        transformers.models.mixtral.modeling_mixtral.load_balancing_loss_func = (
            load_balancing_loss_func
        )


def pack_data_points_FA(
    data_points: List[Dict], tokenizer: Any, model_max_length: int
) -> Dict:
    input_ids = []
    lengths = []
    label_ids = []
    attention_mask = []

    for index, item in enumerate(data_points):
        input_ids += item["input_ids"]
        # assert item["labels"][0] == -100 # This is to make sure that the first token won't be included in computing loss
        labels = list(item["labels"])
        labels[0] = -100
        label_ids += labels
        lengths.append(len(item["input_ids"]))
        attention_mask += [index + 1 for _ in range(len(item["input_ids"]))]

    pad_leng = model_max_length - len(input_ids)  # padding to model_max_length

    if tokenizer.padding_side == "right":
        input_ids = input_ids + [tokenizer.pad_token_id for _ in range(pad_leng)]
        label_ids = label_ids + [-100 for _ in range(pad_leng)]
        attention_mask = attention_mask + [0 for _ in range(pad_leng)]
    else:
        input_ids = [tokenizer.pad_token_id for _ in range(pad_leng)] + input_ids
        label_ids = [-100 for _ in range(pad_leng)] + label_ids
        attention_mask = [0 for _ in range(pad_leng)] + attention_mask

    assert len(input_ids) == len(label_ids) == len(attention_mask) == model_max_length
    return {
        "input_ids": torch.tensor(input_ids),
        "labels": torch.tensor(label_ids),
        "attention_mask": torch.tensor(
            attention_mask
        ),  # unsqueeze <-- because the shape is: B x 1 x N x N
    }
    

def pack_data_points_by_length(
    lengths: List[int], max_length: int, max_size: int = -1
) -> List[List[int]]:
    result = []
    current_concatenated_length = 0
    current_list = []
    for i in range(len(lengths)):
        cur_length = lengths[i]
        if cur_length + current_concatenated_length <= max_length and (
            max_size == -1 or len(current_list) < max_size
        ):
            current_concatenated_length += cur_length
            current_list.append(i)
        else:  # current_list is done, create a new one
            if len(current_list) > 0:
                result.append(current_list)
            current_list = [i]
            current_concatenated_length = cur_length

    if len(current_list) > 0:
        result.append(current_list)

    # assert to make sure no indices were missing
    assert sum([len(indices) for indices in result]) == len(lengths)
    return result


def merge_intervals(
    lengths: List[int], max_length: int, max_size: int = -1, min_item_num: int = -1
) -> List[List[int]]:
    result = [] # list of list
    current_concatenated_length = 0
    current_list = []
    total_item_num = len(lengths)
    for i in range(len(lengths)):
        cur_length = lengths[i]
        # if we merge this to current_list, remaining items: (i + 1 --> total_item_num - 1)
        max_item_after_merge = len(result) + 1 + (total_item_num - 1 - i)
        if cur_length + current_concatenated_length <= max_length and (
            max_size == -1 or len(current_list) < max_size
        ) and max_item_after_merge >= min_item_num:
            current_concatenated_length += cur_length
            current_list.append(i)
        else:  # current_list is done, create a new one
            if len(current_list) > 0:
                result.append(current_list)
            current_list = [i]
            current_concatenated_length = cur_length

    if len(current_list) > 0:
        result.append(current_list)

    # assert to make sure no indices were missing
    assert sum([len(indices) for indices in result]) == len(lengths)
    return result
    

def pack_with_min_item_num(
    lengths: List[int], max_length: int, min_item_num: int = -1
) -> List[List[int]]:
    if min_item_num == -1:
        print("Pack as much as we can as min_item_num is not set", flush=True)
        return pack_data_points_by_length(lengths, max_length, -1)
    
    greedy_packed_groups = pack_data_points_by_length(lengths, max_length, -1)
    if len(greedy_packed_groups) >= min_item_num: # pack as much as we can 
        print(f"max as much as we can still > min_item_num={min_item_num}, size={len(greedy_packed_groups)}", flush=True)
        return greedy_packed_groups
    
    # choose the biggest max_size so that number of groups > min_item_num
    no_packed_groups = [[index] for index in range(len(lengths))] # no packing 
    if len(no_packed_groups) <= min_item_num: # number of items is too small to merge
        print(f"number of original items ({len(lengths)}) is already smaller than: {min_item_num}", flush=True)
        return no_packed_groups
    
    # so if max_size = 1 --> > min_item_num; merge all < min_item_num, so we need to choose the threshold between 1 and 10
    for i in range(2, 51):
        groups = pack_data_points_by_length(lengths, max_length, max_size=i)
        print(f"number of items if max_size={i} is: {len(groups)}", flush=True)
        if len(groups) < min_item_num: # using i-1 --> so many items, i --> so small 
            result = merge_intervals(lengths, max_length, max_size=i, min_item_num=min_item_num)
            print(f"THE BEST MAX GROUP SIZE IN PACKING IS: {i}, size={len(groups)}; number of final groups: {len(result)}", flush=True)
            return result
    return groups


class PackedDataset(Dataset):
    def __init__(
        self,
        dataset: Dataset,
        tokenizer: Any,
        max_input_length: int,
        pack_length: int = -1,
        max_packed_size: int = -1,
        min_item_num: int = -1,
        seed: int = 42
    ) -> None:
        super().__init__()
        self.tokenizer = tokenizer
        self.max_input_length = max_input_length

        self.pack_length = pack_length
        if pack_length == -1:
            self.pack_length = self.max_input_length

        self.lengths = []
        self.data_points = []
        size = len(dataset.eval_dataset)
        for i in range(size):
            dp = dataset.eval_dataset[i]
            input_length = len(dp["input_ids"])
            assert input_length == len(dp["attention_mask"]) == len(dp["labels"])
            self.data_points.append(dp)
            self.lengths.append(input_length)
        self.groups =  pack_with_min_item_num(
            self.lengths, self.pack_length, min_item_num
        )
        random.seed(seed)
        random.shuffle(self.groups)

    def __len__(self) -> int:
        return len(self.groups)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        group = self.groups[i]
        group_data_points = [self.data_points[index] for index in group]
        return pack_data_points_FA(
            group_data_points, self.tokenizer, self.max_input_length
        )

    def stat(self) -> str:
        result_str = ""
        result_str += f"number of original data points:{len(self.data_points)}; packed to: {len(self.groups)} data points"
        original_avg_length = sum(self.lengths) / len(self.lengths)
        result_str += f"original avg length: {original_avg_length}; "
        original_avg_length = sum(self.lengths) / len(self.lengths)

        packed_lengths = []
        for group in self.groups:
            lengths = [self.lengths[index] for index in group]
            packed_lengths.append(sum(lengths))

        avg_packed_length = sum(packed_lengths) / len(packed_lengths)
        result_str += f"original avg length: {original_avg_length}; avg packed length: {avg_packed_length}"
        return result_str