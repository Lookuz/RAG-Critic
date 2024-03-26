import os
import json

from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from utils.incorrect_response_gen import TaskPrompt, generate_incorrect_responses
from utils import const


def bootstrap_dataset_mistral(
    prompt: TaskPrompt,
    dataset, data_path,
    dataset_args: dict,
    model, tokenizer,
    batch_size,
    save_path,
    num_workers=1,
    save_every=1,
):
    # Build dataset from original examples
    dataset = ContextualizedQADatasetForBootstrapping.from_dataset(
        dataset=dataset, data_path=data_path, **dataset_args)
    dataloader = ContextualizedQADataLoaderForBootstrapping(
        dataset, batch_size=batch_size, num_workers=num_workers)

    # Check for existing generated examples
    if os.path.exists(save_path):
        with open(save_path, "r") as f:
            bootstrapped_examples = json.load(f)
        num_generated = len(bootstrapped_examples)
    else:
        bootstrapped_examples, num_generated = [], 0

    # Generate additional examples
    for i, batch in enumerate(tqdm(dataloader)):
        # i enumerates batches while the len is the real number of questions answered
        if i*batch_size < num_generated:
            continue
        if prompt.task == const.BOOTSTRAP_INCORRECT_RESPONSE_TASK:
            # Get rid of the context
            evidences = [x[-1] for x in batch]
            batch = [x[:-1] for x in batch]
            # Generate responses only based on question and correct answer
            outputs = generate_incorrect_responses(model, tokenizer, prompt, batch)

            # Add additional responses to existing examples
            for ((q, r,), r_, d) in zip(batch, outputs, evidences):
                bootstrapped_examples.append({
                    "question": q, "answer": r, "evidence": d, "generated": r_
                })

            if (i + 1) % save_every == 0:
                # Save additional examples to new data files
                with open(save_path, "w") as f:
                    json.dump(bootstrapped_examples, f, indent=4)

    return bootstrapped_examples


class ContextualizedQADataLoaderForBootstrapping(DataLoader):
    def __init__(self,
                 dataset: Dataset,
                 batch_size: int = 1,
                 shuffle: bool = False,
                 num_workers: int = 0):

        super().__init__(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers,
                         collate_fn=ContextualizedQADataLoaderForBootstrapping.collate_fn)

    @classmethod
    def collate_fn(cls, batch):
        return batch


class ContextualizedQADatasetForBootstrapping(Dataset):
    """
    Dataset for text in the form of [Q, R, D] triples, containing the question, response and context respectively.
    Format of data: Each entry should be in the form {"question" : ..., "answer" : ..., "evidence : ...}
    """

    def __init__(
        self, data
    ) -> None:
        super().__init__()

        self.data = data

    def __getitem__(self, index):
        return (self.data[index]["question"], self.data[index]["answer"], self.data[index]["evidence"])

    def __len__(self):
        return len(self.data)

    @classmethod
    def from_dataset(cls, dataset, data_path, **kwargs):
        return {
            "triviaqa": cls.from_trivia_qa
            # TODO: Other datasets, if needed
        }[dataset](data_path, **kwargs)

    @classmethod
    def from_trivia_qa(cls, data_path, evidence_path=None, top_k=1):
        """
        Creates a ContextualizedQADataset for the TriviaQA dataset, using the path to the data provided.
        data_path should be a path to the json file containing the respective split for the TriviaQA dataset.
        """
        # NOTE (Wey Yeh): Currently adapted for web-<split>.json files. The wikipedia-<split>.json files adapt a different format,
        # Where the "SearchResult" key storing the evidence is not present? It seems like extra effort is required to
        # extract these evidence, so I ignored it for now
        with open(data_path, "r") as f:
            data = json.load(f)["Data"]

        examples = []
        for x in data:
            question, answer = x["Question"], x["Answer"]["Value"]
            if evidence_path is not None:
                # Retrieve necessary evidence documents
                evidence = [os.path.join(evidence_path, e['Filename'])
                            for e in x["EntityPages"]]
                evidence = evidence[:top_k] if top_k is not None else evidence
            else:
                evidence = None

            # Store (Q, R, D) triple
            examples.append(
                {"question": question, "answer": answer, "evidence": evidence})

        return cls(data=examples)
