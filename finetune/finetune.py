# -*- coding: utf-8 -*-
import dataclasses as dc
import functools
import os
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Annotated, Any, Optional, Union

import jieba
import numpy as np
import ruamel.yaml as yaml
import torch
import typer
from datasets import Dataset, DatasetDict, NamedSplit, Split, load_dataset
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
from peft import PeftConfig, get_peft_config, get_peft_model
from rouge_chinese import Rouge
from torch import nn
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    EvalPrediction,
    GenerationConfig,
    PreTrainedTokenizer,
    Seq2SeqTrainingArguments,
)
from transformers import DataCollatorForSeq2Seq as _DataCollatorForSeq2Seq
from transformers import Seq2SeqTrainer as _Seq2SeqTrainer


# For Ascend NPU, please add this
# import torch_npu
# from torch_npu.contrib import transfer_to_npu

app = typer.Typer(pretty_exceptions_show_locals=False)

# 用于seq2seq任务的数据整理，比如机器翻译、摘要
# 原始样本要整理成模型可以直接输入的 batch 数据，DataCollatorForSeq2Seq是默认的collator
class DataCollatorForSeq2Seq(_DataCollatorForSeq2Seq):
    # # 输入：features 是一个batch，通常是一个字典列表，每个字典代表一个样本。return_tensors=None 表示不返回张量
    def __call__(self, features, return_tensors=None):
        output_ids = [feature["output_ids"] for feature in features] if "output_ids" in features[0].keys() else None
        #  output_ids（通常是目标序列的token id）
        if output_ids is not None:
            max_output_length = max(len(out) for out in output_ids)# 计算目标序列的最大长度
            if self.pad_to_multiple_of is not None:
                max_output_length = (
                    (max_output_length + self.pad_to_multiple_of - 1)
                    // self.pad_to_multiple_of
                    * self.pad_to_multiple_of
                )  # 将最大长度补齐到这个数的倍数（有些硬件/模型对齐有要求）
                # 注意上面这里的计算方法，是向上取整
            for feature in features:#遍历样本
                # 用pad_token_id填充到max_output_length，得到一个list
                remainder = [self.tokenizer.pad_token_id] * (max_output_length - len(feature["output_ids"]))
                if isinstance(feature["output_ids"], list):#如果原本是list，直接拼接
                    feature["output_ids"] = feature["output_ids"] + remainder
                else:  # 如果是 numpy 数组，则用 np.concatenate 拼接，并转成 int64 类型
                    feature["output_ids"] = np.concatenate([feature["output_ids"], remainder]).astype(np.int64)
        return super().__call__(features, return_tensors)


class Seq2SeqTrainer(_Seq2SeqTrainer):
    def prediction_step(
        self,
        model: nn.Module,
        inputs: dict[str, Any],
        prediction_loss_only: bool,
        ignore_keys=None,
        **gen_kwargs,
    ) -> tuple[Optional[float], Optional[torch.Tensor], Optional[torch.Tensor]]:
        with torch.no_grad():  # Ensure no gradient computation，推理时不需要反向传播
            if self.args.predict_with_generate:
                # 如果当前是生成模式（如做生成式评测），从 inputs 中移除 output_ids
                output_ids = inputs.pop("output_ids")
            # 取出 input_ids
            input_ids = inputs["input_ids"]
            # 删除 labels 键，因为用output_ids作为标签
            del inputs["labels"]
            # 调用父类（Seq2SeqTrainer）的 prediction_step 方法，计算损失、生成 token 和标签（这里labels通常指的是正确答案）
            loss, generated_tokens, labels = super().prediction_step(
                model, inputs, prediction_loss_only, ignore_keys, **gen_kwargs
            )

            # 只保留生成序列中“新生成”的部分，去掉输入 prompt 部分
            # 常用判断形状的方法：打印size；前几个元素；断点调试
            generated_tokens = generated_tokens[:, input_ids.size()[1]:]
            labels = output_ids  # 用之前 pop 出来的 output_ids 作为标签，方便后续评测

            del inputs, input_ids, output_ids
            torch.cuda.empty_cache()

        return loss, generated_tokens, labels#返回：损失、生成token、标签


@dc.dataclass
class DataConfig(object):
    train_file: Optional[str] = None
    val_file: Optional[str] = None
    test_file: Optional[str] = None
    num_proc: Optional[int] = None

    @property
    def data_format(self) -> str:
        return Path(self.train_file).suffix#打乱顺序

    @property
    def data_files(self) -> dict[NamedSplit, str]:
        return {
            split: data_file
            for split, data_file in zip(
                [Split.TRAIN, Split.VALIDATION, Split.TEST],
                [self.train_file, self.val_file, self.test_file],
            )
            if data_file is not None
        }


@dc.dataclass
class FinetuningConfig(object):
    data_config: DataConfig

    max_input_length: int
    max_output_length: int
    combine: bool
    freezeV: bool

    training_args: Seq2SeqTrainingArguments = dc.field(
        default_factory=lambda: Seq2SeqTrainingArguments(output_dir="./output")
    )
    peft_config: Optional[PeftConfig] = None
    swanlab: Optional[str] = "cloud"

    # 是 dataclasses 的特殊方法，会在对象初始化后自动调用，用于做一些额外的参数检查和环境设置
    def __post_init__(self):
        # 判断是否需要做验证
        if not self.training_args.do_eval or self.data_config.val_file is None:
            self.training_args.do_eval = False
            self.training_args.evaluation_strategy = "no"
            self.data_config.val_file = None
        else:
            self.training_args.per_device_eval_batch_size = (
                self.training_args.per_device_eval_batch_size or self.training_args.per_device_train_batch_size
            )
        if self.swanlab != "disabled":
            os.environ["SWANLAB_PROJ_NAME"] = "GLM4-Finetune"
        if self.swanlab == "local":
            os.environ["SWANLAB_MODE"] = "local"

    @classmethod
    def from_dict(cls, **kwargs) -> "FinetuningConfig":
        training_args = kwargs.get("training_args", None)
        # 加载配置文件
        if training_args is not None and not isinstance(training_args, Seq2SeqTrainingArguments):
            gen_config = training_args.get("generation_config")
            if not isinstance(gen_config, GenerationConfig):
                training_args["generation_config"] = GenerationConfig(**gen_config)
            kwargs["training_args"] = Seq2SeqTrainingArguments(**training_args)

        data_config = kwargs.get("data_config")
        if not isinstance(data_config, DataConfig):
            kwargs["data_config"] = DataConfig(**data_config)

        peft_config = kwargs.get("peft_config", None)
        if peft_config is not None and not isinstance(peft_config, PeftConfig):
            kwargs["peft_config"] = get_peft_config(config_dict=peft_config)
        return cls(**kwargs)

    @classmethod
    def from_file(cls, path: Union[str, Path]) -> "FinetuningConfig":
        path = Path(path)
        parser = yaml.YAML(typ="safe", pure=True)
        parser.indent(mapping=2, offset=2, sequence=4)
        parser.default_flow_style = False
        kwargs = parser.load(path)
        return cls.from_dict(**kwargs)


def _load_datasets(
    data_dir: str,
    data_format: str,
    data_files: dict[NamedSplit, str],
    num_proc: Optional[int],
) -> DatasetDict:
    if data_format == ".jsonl":
        dataset_dct = load_dataset(
            data_dir,
            data_files=data_files,
            split=None,
            num_proc=num_proc,
        )
    else:
        raise NotImplementedError(f"Cannot load dataset in the '{data_format}' format.")
    return dataset_dct


class DataManager(object):
    def __init__(self, data_dir: str, data_config: DataConfig):
        self._num_proc = data_config.num_proc

        self._dataset_dct = _load_datasets(
            data_dir,
            data_config.data_format,
            data_config.data_files,
            self._num_proc,
        )

    def _get_dataset(self, split: NamedSplit) -> Optional[Dataset]:
        return self._dataset_dct.get(split, None)

    def get_dataset(
        self,
        split: NamedSplit,
        process_fn: Callable[[dict[str, Any]], dict[str, Any]],
        batched: bool = True,
        remove_orig_columns: bool = True,
    ) -> Optional[Dataset]:
        orig_dataset = self._get_dataset(split)
        if orig_dataset is None:
            return
        # 移除数据集中的一些列，比如id、index等
        if remove_orig_columns:
            remove_columns = orig_dataset.column_names
        else:
            remove_columns = None
        return orig_dataset.map(
            process_fn,
            batched=batched,
            remove_columns=remove_columns,
            num_proc=self._num_proc,
        )


def process_message(message):
    if "tools" in message and message["role"] == "system":
        for tool in message["tools"]:
            # 每个工具一定有输入参数。这里是从信息中提取出输入参数，并存入tool中
            parameters = tool["function"]["parameters"]["properties"]
            tool["function"]["parameters"]["properties"] = {k: v for k, v in parameters.items() if v is not None}
    elif "tools" in message:#如果不是系统消息，就删除消息中的tools（规范中，要求，只有系统消息可以有tools，User和assistant只能调用，不能定义工具）
        del message["tools"]
    return message

# 预处理对话数据
def process_batch(
    batch: Mapping[str, Sequence],
    tokenizer: PreTrainedTokenizer,
    max_input_length: int,
    max_output_length: int,
    combine: bool,  # 是否将多轮对话合并为一个整体
) -> dict[str, list]:
    batched_conv = batch["messages"]
    batched_input_ids = []
    batched_labels = []
    for conv in batched_conv:
        input_ids = [151331, 151333]# 用前两个token标识系统消息和用户消息
        loss_masks = [False, False]  # 初始化损失掩码，前两个token不参与loss
        if combine:  # 把整个对话用 tokenizer.apply_chat_template 一次性处理成 token id
            new_input_ids = tokenizer.apply_chat_template(conv, tokenize=True, return_dict=False)
            input_ids = new_input_ids
            loss_masks = [False] * len(input_ids)  # loss_masks 全部初始化为 False
            last_assistant_index = len(input_ids) - input_ids[::-1].index(151337) - 1# 找到最后一个assistant消息的索引
            # 查找方法：先反转，然后找到第一个151337的索引，这是因为每个助手消息都以151337开头
            for j in range(last_assistant_index + 1, len(input_ids)):
                loss_masks[j] = True# 只有最后一个assistant消息，让所有token都参与loss
        else:
            for message in conv:  # 逐条处理每个 message
                message = process_message(message)
                loss_mask_val = False if message["role"] in ("system", "user", "observation") else True#如果是助手消息，就要参与loss
                new_input_ids = tokenizer.apply_chat_template([message], tokenize=True, return_dict=False)[2:]# 处理整条消息，返回token id列表。去掉前两个token（是占位符，我们最开始就添加了）
                input_ids += new_input_ids
                loss_masks += [loss_mask_val] * len(new_input_ids)

        input_ids.append(151336)  # EOS for chat
        loss_masks = [False, *loss_masks]  # 在前面添加一个False，表示EOS不参与loss
        labels = []# 初始化labels列表，用于存储每个token的标签
        for input_id, mask in zip(input_ids, loss_masks):
            if mask:
                labels.append(input_id)# 如果mask为True，则将input_id添加到labels列表中
            else:
                labels.append(-100)# 如果mask为False，则将-100添加到labels列表中
        max_length = max_input_length + max_output_length + 1#最大长度等于最大输入长度+最大输出长度+1（EOS），这两个最大长度都是通过函数参数直接传过来的
        batched_input_ids.append(input_ids[:max_length])
        batched_labels.append(labels[:max_length])

    del batched_conv, conv, input_ids, loss_masks, new_input_ids, labels
    torch.cuda.empty_cache()
# 下面这里，input_id是原始token id，labels是用掩码处理之后的，只保留了助手消息的部分
# 在训练的时候，labels会作为正确答案，input_ids会作为模型输入，labels表示希望模型输出的内容，要根据交叉熵损失计算loss
    return {"input_ids": batched_input_ids, "labels": batched_labels}#最后返回处理后的token id和labels


def process_batch_eval(
    batch: Mapping[str, Sequence],
    tokenizer: PreTrainedTokenizer,
    max_input_length: int,
    max_output_length: int,
    combine: bool,
) -> dict[str, list]:
    batched_conv = batch["messages"]
    batched_input_ids = []
    batched_output_ids = []

    for conv in batched_conv:
        if combine:#如果combine为True，则将整个对话合并为一个整体
            new_input_ids = tokenizer.apply_chat_template(conv, tokenize=True, return_dict=False)
            input_ids = new_input_ids
            last_assistant_index = len(input_ids) - input_ids[::-1].index(151337) - 1
            output_prompt, output_ids = (
                input_ids[:1],# output_prompt是第一个token_id
                input_ids[last_assistant_index:],# output_ids是助手消息的token_id
            )
            output_ids.append(151336)#给output_ids添加EOS
            # 这里的input_ids是整个对话的token_id
            batched_input_ids.append(input_ids[:max_input_length] + output_prompt[:1])
            batched_output_ids.append(output_ids[:max_output_length])
        else:
            input_ids = [151331, 151333]
            for message in conv:
                if len(input_ids) >= max_input_length:
                    break
                else:
                    message = process_message(message)
                    new_input_ids = tokenizer.apply_chat_template([message], tokenize=True, return_dict=False)[2:]
                    if message["role"] == "assistant":
                        output_prompt, output_ids = (
                            new_input_ids[:1],
                            new_input_ids[1:],
                        )
                        output_ids.append(151336)
                        batched_input_ids.append(input_ids[:max_input_length] + output_prompt[:1])
                        batched_output_ids.append(output_ids[:max_output_length])
                    input_ids += new_input_ids

    del batched_conv, conv, input_ids, new_input_ids, output_prompt, output_ids
    torch.cuda.empty_cache()

    return {"input_ids": batched_input_ids, "output_ids": batched_output_ids}

# 从目录中加载tokenizer和模型
def load_tokenizer_and_model(
    model_dir: str,
    peft_config: Optional[PeftConfig] = None,
):
    tokenizer = AutoTokenizer.from_pretrained(model_dir, padding_side="left", trust_remote_code=True)# 加载tokenizer，padding_side="left"表示在左边填充
    if peft_config is not None:
        model = AutoModelForCausalLM.from_pretrained(
            model_dir,
            use_cache=False,
            torch_dtype=torch.bfloat16,  # Must use BFloat 16
        )
        model = get_peft_model(model, peft_config)# 加载PEFT配置，将模型转换为PEFT模型，这是为了方便后续的微调
        model.print_trainable_parameters()
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_dir,
            use_cache=False,
            torch_dtype=torch.bfloat16,
        )
    return tokenizer, model

# 计算评估指标
def compute_metrics(eval_preds: EvalPrediction, tokenizer):
    # 模型预测的 token id（batched_pred_ids）和真实标签的 token id（batched_label_ids）
    batched_pred_ids, batched_label_ids = eval_preds
    batched_pred_ids[batched_pred_ids == -100] = tokenizer.pad_token_id# 将-100替换为pad_token_id
    batched_label_ids[batched_label_ids == -100] = tokenizer.pad_token_id# 将-100替换为pad_token_id
    metrics_dct = {"rouge-1": [], "rouge-2": [], "rouge-l": [], "bleu-4": []}
    for pred_ids, label_ids in zip(batched_pred_ids, batched_label_ids):
        # 将 token id 转为文本，并去除首尾空格
        pred_txt = tokenizer.decode(pred_ids).strip()
        label_txt = tokenizer.decode(label_ids).strip()
        pred_tokens = list(jieba.cut(pred_txt))  # 用 jieba 对中文文本分词，得到 txt列表
        label_tokens = list(jieba.cut(label_txt))
        rouge = Rouge()  # 创建一个 ROUGE 评测对象，专门用于计算文本之间的 ROUGE 分数
        # 把预测文本和标签文本的分词结果用空格拼接成字符串（因为 ROUGE 需要以“词”为单位对比）
        scores = rouge.get_scores(
            " ".join(pred_tokens), " ".join(label_tokens))
        # 遍历 ROUGE 结果中的每一项（如 "rouge-1"、"rouge-2"、"rouge-l"），取出每项的 F1 分数，乘以 100 变成百分比
        for k, v in scores[0].items():
            metrics_dct[k].append(round(v["f"] * 100, 4))
        metrics_dct["bleu-4"].append(
            sentence_bleu(  # 计算 BLEU-4 分数
                [label_tokens],#传入txt列表
                pred_tokens,
                smoothing_function=SmoothingFunction().method3,  # BLEU 分数在短文本时容易为 0，平滑函数可以让分数更合理
            )
        )
    # 对每个指标（ROUGE-1、ROUGE-2、ROUGE-L、BLEU-4），计算所有样本的平均分数
    return {k: np.mean(v) for k, v in metrics_dct.items()}


@app.command()
# 包含：加载配置、模型；加载数据；训练；评估；保存模型
def main(
    data_dir: Annotated[str, typer.Argument(help="")],
    model_dir: Annotated[
        str,
        typer.Argument(
            help="A string that specifies the model id of a pretrained model configuration hosted on huggingface.co, or a path to a directory containing a model configuration file."
        ),
    ],
    config_file: Annotated[str, typer.Argument(help="")],
    auto_resume_from_checkpoint: str = typer.Argument(
        default="",
        help="If entered as yes, automatically use the latest save checkpoint. If it is a numerical example 12 15, use the corresponding save checkpoint. If the input is no, restart training",
    ),
):
    ft_config = FinetuningConfig.from_file(config_file)  # 从配置文件加载微调参数
    tokenizer, model = load_tokenizer_and_model(model_dir, peft_config=ft_config.peft_config)  # 加载tokenizer和模型
    data_manager = DataManager(data_dir, ft_config.data_config)  # 加载数据

    # 加载训练数据
    train_dataset = data_manager.get_dataset(
        Split.TRAIN,
        functools.partial(
            process_batch,#用 process_batch 进行预处理，返回input_ids和labels
            tokenizer=tokenizer,
            combine=ft_config.combine,
            max_input_length=ft_config.max_input_length,
            max_output_length=ft_config.max_output_length,
        ),
        batched=True,
    )
    print("train_dataset:", train_dataset)
    val_dataset = data_manager.get_dataset(
        Split.VALIDATION,
        functools.partial(
            process_batch_eval,
            tokenizer=tokenizer,
            combine=ft_config.combine,
            max_input_length=ft_config.max_input_length,
            max_output_length=ft_config.max_output_length,
        ),
        batched=True,
    )
    if val_dataset is not None:
        print("val_dataset:", val_dataset)
    test_dataset = data_manager.get_dataset(
        Split.TEST,
        functools.partial(
            process_batch_eval,
            tokenizer=tokenizer,
            combine=ft_config.combine,
            max_input_length=ft_config.max_input_length,
            max_output_length=ft_config.max_output_length,
        ),
        batched=True,
    )
    if test_dataset is not None:
        print("test_dataset:", test_dataset)
# 设置生成时的 pad token 和 eos token
    ft_config.training_args.generation_config.pad_token_id = 151329
    ft_config.training_args.generation_config.eos_token_id = [151329, 151336, 151338]

    trainer = Seq2SeqTrainer(
        model=model,
        args=ft_config.training_args,
        data_collator=DataCollatorForSeq2Seq(#传入修改过的collator，因为要处理labels
            tokenizer=tokenizer,
            padding="longest",
            return_tensors="pt",
        ),
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        # 这里的 compute_metrics 用于评估时自动计算ROUGE、BLEU等指标
        compute_metrics=functools.partial(compute_metrics, tokenizer=tokenizer),
    )

    if auto_resume_from_checkpoint.upper() == "" or auto_resume_from_checkpoint is None:
        trainer.train()
    else:#如果指定了检查点，则从检查点继续训练
        output_dir = ft_config.training_args.output_dir
        dirlist = os.listdir(output_dir)
        checkpoint_sn = 0
        for checkpoint_str in dirlist:
            if checkpoint_str.find("eckpoint") > 0 and checkpoint_str.find("tmp") == -1:
                checkpoint = int(checkpoint_str.replace("checkpoint-", ""))
                if checkpoint > checkpoint_sn:
                    checkpoint_sn = checkpoint
        if auto_resume_from_checkpoint.upper() == "YES":
            if checkpoint_sn > 0:
                model.gradient_checkpointing_enable()
                model.enable_input_require_grads()
                checkpoint_directory = os.path.join(output_dir, "checkpoint-" + str(checkpoint_sn))
                print("resume checkpoint from checkpoint-" + str(checkpoint_sn))
                trainer.train(resume_from_checkpoint=checkpoint_directory)
            else:
                trainer.train()
        else:
            if auto_resume_from_checkpoint.isdigit():
                if int(auto_resume_from_checkpoint) > 0:
                    checkpoint_sn = int(auto_resume_from_checkpoint)
                    model.gradient_checkpointing_enable()
                    model.enable_input_require_grads()
                    checkpoint_directory = os.path.join(output_dir, "checkpoint-" + str(checkpoint_sn))
                    print("resume checkpoint from checkpoint-" + str(checkpoint_sn))
                    trainer.train(resume_from_checkpoint=checkpoint_directory)
            else:
                print(
                    auto_resume_from_checkpoint,
                    "The specified checkpoint sn("
                    + auto_resume_from_checkpoint
                    + ") has not been saved. Please search for the correct checkpoint in the model output directory",
                )

    if test_dataset is not None:
        trainer.predict(test_dataset)  # 在测试集上做推理，输出预测结果


if __name__ == "__main__":
    app()
