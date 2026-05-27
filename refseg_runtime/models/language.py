from __future__ import annotations

from collections import OrderedDict
from typing import Dict, List, Sequence

import torch
import torch.nn as nn

from ..checkpoint import load_prefixed_state_dict, load_prefixed_state_dict_from_state_dict
from ..runtime_env import ensure_transformers_backend


def generate_masks_with_special_tokens_and_transfer_map(
    tokenized,
    special_tokens_list: List[int],
):
    input_ids = tokenized["input_ids"]
    batch_size, num_token = input_ids.shape
    special_tokens_mask = torch.zeros((batch_size, num_token), device=input_ids.device).bool()

    for special_token in special_tokens_list:
        special_tokens_mask |= input_ids == special_token

    idxs = torch.nonzero(special_tokens_mask)
    attention_mask = torch.eye(num_token, device=input_ids.device).bool().unsqueeze(0).repeat(batch_size, 1, 1)
    position_ids = torch.zeros((batch_size, num_token), device=input_ids.device)
    previous_col = 0
    for i in range(idxs.shape[0]):
        row, col = idxs[i]
        if (col == 0) or (col == num_token - 1):
            attention_mask[row, col, col] = True
            position_ids[row, col] = 0
        else:
            attention_mask[row, previous_col + 1 : col + 1, previous_col + 1 : col + 1] = True
            position_ids[row, previous_col + 1 : col + 1] = torch.arange(
                0, col - previous_col, device=input_ids.device
            )
        previous_col = col

    return attention_mask, position_ids.to(torch.long)


class PortableBertEncoder(nn.Module):
    def __init__(
        self,
        name: str,
        add_pooling_layer: bool = False,
        num_layers_of_embedded: int = 1,
        use_checkpoint: bool = False,
        optional_site_packages: str = "",
    ) -> None:
        super().__init__()
        optional_site_packages = ensure_transformers_backend(optional_site_packages)
        from transformers import BertConfig
        from transformers import BertModel as HFBertModel

        config = BertConfig.from_pretrained(name)
        config.gradient_checkpointing = use_checkpoint
        self.model = HFBertModel.from_pretrained(
            name,
            add_pooling_layer=add_pooling_layer,
            config=config,
        )
        self.language_dim = config.hidden_size
        self.num_layers_of_embedded = num_layers_of_embedded

    def forward(self, x: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        mask = x["attention_mask"]
        outputs = self.model(
            input_ids=x["input_ids"],
            attention_mask=mask,
            position_ids=x["position_ids"],
            token_type_ids=x["token_type_ids"],
            output_hidden_states=True,
        )
        encoded_layers = outputs.hidden_states[1:]
        features = torch.stack(encoded_layers[-self.num_layers_of_embedded :], 1).mean(1)
        features = features / self.num_layers_of_embedded
        if mask.dim() == 2:
            embedded = features * mask.unsqueeze(-1).float()
        else:
            embedded = features
        return {
            "embedded": embedded,
            "masks": mask,
            "hidden": encoded_layers[-1],
        }


class PortableGroundingBertTextEncoder(nn.Module):
    """Runtime BERT tower aligned with GroundingDINO/XeMap text inputs."""

    def __init__(
        self,
        name: str = "bert-base-uncased",
        max_tokens: int = 256,
        pad_to_max: bool = False,
        use_sub_sentence_represent: bool = True,
        special_tokens_list: Sequence[str] | None = None,
        add_pooling_layer: bool = False,
        num_layers_of_embedded: int = 1,
        use_checkpoint: bool = False,
        output_dims: int = 256,
        optional_site_packages: str = "",
    ) -> None:
        super().__init__()
        optional_site_packages = ensure_transformers_backend(optional_site_packages)
        from transformers import AutoTokenizer

        self.max_tokens = max_tokens
        self.pad_to_max = pad_to_max
        self.use_sub_sentence_represent = use_sub_sentence_represent
        self.tokenizer = AutoTokenizer.from_pretrained(name)
        self.language_backbone = nn.Sequential(
            OrderedDict(
                [
                    (
                        "body",
                        PortableBertEncoder(
                            name=name,
                            add_pooling_layer=add_pooling_layer,
                            num_layers_of_embedded=num_layers_of_embedded,
                            use_checkpoint=use_checkpoint,
                            optional_site_packages=optional_site_packages,
                        ),
                    )
                ]
            )
        )
        language_dim = self.language_backbone.body.language_dim
        self.text_feat_map = nn.Linear(language_dim, output_dims)
        self.text_feat_map_sim = nn.Linear(output_dims, output_dims)
        self.special_tokens = None
        if use_sub_sentence_represent:
            tokens = list(special_tokens_list or ["[CLS]", "[SEP]", ".", "?"])
            self.special_tokens = self.tokenizer.convert_tokens_to_ids(tokens)

    def load_partial_checkpoint(self, checkpoint_path: str, strict: bool = False, state_dict=None):
        reports = {}
        prefix_candidates = (
            ("language_model.language_backbone", ("language_model.language_backbone",)),
            ("text_feat_map", ("language_model.text_feat_map", "text_feat_map")),
            ("text_feat_map_sim", ("language_model.text_feat_map_sim", "text_feat_map_sim")),
        )
        for report_key, candidates in prefix_candidates:
            module = getattr(self, report_key if report_key != "language_model.language_backbone" else "language_backbone")
            load_error = None
            for prefix in candidates:
                try:
                    if state_dict is None:
                        missing, unexpected = load_prefixed_state_dict(module, checkpoint_path, prefix, strict=strict)
                    else:
                        missing, unexpected = load_prefixed_state_dict_from_state_dict(
                            module,
                            state_dict,
                            prefix,
                            strict=strict,
                        )
                except RuntimeError as exc:
                    load_error = exc
                    continue
                reports[report_key] = {
                    "resolved_prefix": prefix,
                    "missing": missing,
                    "unexpected": unexpected,
                }
                break
            else:
                reports[report_key] = {"missing": [str(load_error)], "unexpected": []}
        return reports

    def forward(self, captions: Sequence[str], device: torch.device) -> Dict[str, torch.Tensor]:
        tokenized = self.tokenizer.batch_encode_plus(
            captions,
            max_length=self.max_tokens,
            padding="max_length" if self.pad_to_max else "longest",
            return_special_tokens_mask=True,
            return_tensors="pt",
            truncation=True,
        ).to(device)

        input_ids = tokenized.input_ids
        if self.use_sub_sentence_represent:
            attention_mask, position_ids = generate_masks_with_special_tokens_and_transfer_map(
                tokenized,
                self.special_tokens or [],
            )
            token_type_ids = tokenized["token_type_ids"]
            text_token_mask = tokenized.attention_mask.bool()
        else:
            attention_mask = tokenized.attention_mask
            position_ids = None
            token_type_ids = None
            text_token_mask = tokenized.attention_mask.bool()

        language_dict = self.language_backbone(
            {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "position_ids": position_ids,
                "token_type_ids": token_type_ids,
            }
        )
        language_dict["embedded_before_proj"] = language_dict["embedded"]
        language_dict["embedded"] = self.text_feat_map(language_dict["embedded"])
        language_dict["text_token_mask"] = text_token_mask
        language_dict["position_ids"] = position_ids
        language_dict["memory_text"] = language_dict["embedded"]
        language_dict["text_attention_mask"] = ~text_token_mask
        language_dict["text_self_attention_masks"] = language_dict["masks"]
        return language_dict
