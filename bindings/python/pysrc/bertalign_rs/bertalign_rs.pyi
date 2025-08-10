from typing import List, Tuple, Optional

class BertAlignArgs:
    def __init__(
        max_align=5,
        top_k=3,
        win=5,
        skip=-0.1,
        margin=True,
        len_penalty=True,
    ) -> None:
        """Arguments used for aligning sentences"""
        ...

class BertAlign:
    def __init__(args: Optional[BertAlignArgs]) -> None:
        """BertAlinger.
        
        Args:
            args (BertAlignArgs): Alignment arguments. When not provided default arguments are used.
        """
        ...

    def align(src_sents: List[str], tgt_sents: List[str]) -> List[Tuple[List[int], List[int]]]:
        """Align a list of source and target sentences.
        
        Args:
            src_sents (List[str]): List of source sentences
            tgt_sents (List[str]): List of target sentences

        Returns:
            List[Tuple[List[int], List[int]]]: List of tuples containing two lists.

            The first element in the tuple is the indicies of the source sentences. The
            second element in the tuple is the indicies of the target sentences.
        """
        ...


class LaBSE:
    def __init__(use_safetensors=True, batch_size=32) -> None:
        """LaBSE implementation in Candle

        Args:
            use_safetensors (bool): use safetensors to load model
            batch_size (int): number of sentences to embed per batch
        """
        ...