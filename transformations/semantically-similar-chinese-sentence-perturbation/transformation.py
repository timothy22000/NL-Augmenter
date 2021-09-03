import itertools
import random
import opencc
import os

from interfaces.SentenceOperation import SentenceOperation
from tasks.TaskTypes import TaskType
import nlpcda
# from nlpcda import Simbert


def chinese_semantically_similar_sentence(text,
                          prob,
                          seed,
                          max_outputs,
                          model_path,
                          max_len,
                          cuda
                          ):
    random.seed(seed)

    perturbed_texts = []

    config = {
        'model_path' : model_path,
        'CUDA_VISIBLE_DEVICES': cuda,
        'max_len': max_len,
        'seed': seed
    }

    simbert = nlpcda.Simbert(config=config)
    outputs = simbert.replace(text, create_num=max_outputs+1)
    outputs = outputs[1:]
    perturbed_texts = outputs
    return perturbed_texts




"""
Chinese Words and Characters Butter Fingers Perturbation
"""

class ChineseSemanticallySimilarSentencePerturbation(SentenceOperation):
    tasks = [
        TaskType.TEXT_CLASSIFICATION,
        TaskType.TEXT_TO_TEXT_GENERATION
    ]
    languages = ["zh"]

    def __init__(self, seed=0, max_outputs=1, prob=1, cuda='-1'):
        super().__init__(seed, max_outputs=max_outputs)
        self.prob = prob
        self.seed = seed
        self.max_outputs = max_outputs
        dirname = os.path.dirname(__file__)
        self.model_path = os.path.join(dirname, 'chinese_L-12_H-768_A-12')
        self.cuda = cuda

    def generate(self, sentence: str, max_len: int = 32):
        perturbed_texts = chinese_semantically_similar_sentence(
            text=sentence,
            prob=self.prob,
            seed=self.seed,
            max_outputs=self.max_outputs,
            model_path=self.model_path,
            max_len = max_len,
            cuda = self.cuda
        )
        return perturbed_texts

if __name__ == '__main__':
    simp_text = "随着两个遗迹文明的发展，他们终于开始了争斗。遗迹之间的能量冲突是战争的导火索，因为一方出现，另一方的遗迹能量就会相应的颓落。"
    perturb_func = ChineseSemanticallySimilarSentencePerturbation()
    new_text = perturb_func.generate(simp_text)
    print(new_text)



