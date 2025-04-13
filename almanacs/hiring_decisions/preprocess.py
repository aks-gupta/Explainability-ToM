import tqdm 
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import pandas as pd
from pathlib import Path
from abc import ABC, abstractmethod
from string import ascii_uppercase, ascii_lowercase
from typing import Any, Callable, Mapping, Optional, Union
import numpy as np
from collections import abc
import json
import itertools
import copy
import os 
import re

print(os.getcwd())

MAX_COMBINATIONS = 50000

@dataclass
class Substitution:
    """
    Represents a possible replacement for a variable in a question template.

    Attributes:
        variable_name (str): The name of the variable to be replaced.
        value (str): The value with which the variable will be replaced.
    """
    variable_name: str
    value: str

class Question(ABC):
    text: str

    @abstractmethod
    def is_correct(self, answer: Any) -> bool:
        ...

    def __repr__(self):
        return f'{self.__class__.__name__}: {self.text}'

class Prompter(ABC):
    @abstractmethod
    def make_prompt(self, question: Question) -> str:
        ...

    @abstractmethod
    def get_answer(self, completion: str) -> Any:
        ...

    def __repr__(self) -> str:
        return self.__class__.__name__

@dataclass
class Completion:
    prompt: str
    text: str
    meta: Optional[Dict] = None
    logprobs: Optional[List[Dict[str, float]]] = None
    token_logprobs: Optional[List[Union[None, float]]] = None
    tokens: Optional[List[str]] = None

    def __repr__(self):
        return f'{self.__class__.__name__}: Prompt: {self.prompt}, Completion Text: {self.text}'

    @property
    def mc_answer_probs(self):
        if self.logprobs is None:
            return None
        return {token.strip(): math.exp(logprob) for token, logprob in self.logprobs[0].items()}

    @property
    def total_answer_prob(self):
        if not self.meta:
            return None
        return self.meta.get('total_answer_prob')

    def to_dict(self, include_logprobs: bool = False) -> Dict[str, Any]:
        ret: Dict[str, Any] = dict(prompt=self.prompt, completion=self.text, meta=self.meta)
        if self.logprobs and len(self.logprobs) == 1:
            ret['answer_probs'] = self.mc_answer_probs
        if include_logprobs:
            ret['logprobs'] = self.logprobs
        return ret


@dataclass
class Answer:
    question: Question
    completion: Completion
    prompter: Prompter

    @property
    def given_answer(self) -> Any:
        return self.prompter.get_answer(self.completion.text)

    @property
    def is_correct(self) -> bool:
        return self.question.is_correct(self.given_answer)

    def to_dict(self):
        data = self.completion.to_dict()
        if isinstance(self.question, (MultipleChoiceQuestion, ExactMatchQuestion)):
            data['label'] = self.question.label
        return data


@dataclass
class AnswerSet(abc.Sequence):
    answers: List[Answer]

    @property
    def prompter(self):
        return self.answers[0].prompter

    @property
    def is_multiple_choice(self):
        return isinstance(self.prompter, SingleWordAnswerPrompter) and not self.prompter.score_prefix

    @property
    def questions(self):
        return [a.question for a in self.answers]

    @property
    def completions(self):
        return [a.completion for a in self.answers]

    def __repr__(self):
        return f"{self.__class__.__name__}: {dict(example_question=self.questions[0], example_completion=self.completions[0], prompter=self.prompter)}"

    def __len__(self):
        return len(self.answers)

    def __getitem__(self, idx) -> Union[Answer, 'AnswerSet']:
        if isinstance(idx, int):
            return self.answers[idx]
        else:
            return AnswerSet(self.answers[idx])  # type: ignore

    @property
    def given_answers(self) -> List[Any]:
        return [a.given_answer for a in self.answers]

    @property
    def yeses(self) -> List[bool]:
        def is_yes(answer: Any):
            if answer is None:
                return False
            if isinstance(answer, str) and answer.strip().lower() == 'yes':
                return True
            try:
                if answer > 0.5:  # type: ignore
                    return True
            except:
                pass
            return False
        return [is_yes(a) for a in self.given_answers]

    @property
    def mc_answer_probs(self) -> List[Optional[Dict[str, float]]]:
        return [c.mc_answer_probs for c in self.completions]

    @property
    def positive_answer_probs(self) -> List[float]:
        if self.is_multiple_choice:
            return [probs['Yes'] for probs in self.mc_answer_probs] # type: ignore
        return [float(a) for a in self.yeses]  # type: ignore
    
    @property
    def valid_answers(self):
        if self.is_multiple_choice:
            return [True for _ in self.answers]
        return [a is not None for a in self.given_answers]

    @property
    def correct(self) -> List[bool]:
        return [a.is_correct for a in self.answers]

    @property
    def accuracy(self) -> float:
        return np.array(self.correct, dtype=float).mean()

    @property
    def mean(self) -> float:
        assert all(isinstance(a, float) for a in self.given_answers)
        return np.array(self.given_answers).mean()

    @property
    def earliest_true_index(self) -> Union[int, None]:
        for i, correct in enumerate(self.correct):
            if correct:
                return i
        return None

    def to_dict(self):
        return dict(answers=[a.to_dict() for a in self])


class MultipleChoiceQuestion(Question):
    option_chars = ascii_uppercase

    def __init__(self, text: str, options: Mapping[str, str], label: Optional[str] = None) -> None:
        if label is not None and label not in options:
            raise ValueError('Label must be in options')
        self.question_text = text
        self.options = options
        self.label = label

    @property
    def text(self):
        return self.question_text + '\n' + '\n'.join(f'{c}: {a}' for c, a in self.options.items())

    def is_correct(self, answer: str) -> bool:
        if self.label is None:
            raise AttributeError('Label is None')
        return answer.lower() == self.label.lower().strip()

class YesNoQuestion(MultipleChoiceQuestion):
    option_chars = ['Yes', 'No']

    def __init__(self, text: str, label: Optional[str] = None) -> None:
        if label is not None and label not in self.option_chars:
            raise ValueError('Label must be in options')
        self.question_text = text
        self.options = {o: o for o in self.option_chars}
        self.label = label

    @property
    def text(self):
        return self.question_text

    def is_correct(self, answer: Union[str, float, bool]) -> bool:
        if self.label is None:
            raise AttributeError('Label is None')
        if answer is None:
            return False
        if isinstance(answer, float):
            answer = 'Yes' if answer > 0.5 else 'No'
        elif isinstance(answer, bool):
            answer = 'Yes' if answer else 'No'
        return answer.lower() == self.label.lower().strip()

@dataclass
class ValueCombination:
    """
    Represents a set of values that replace variables in a particular template.

    Attributes:
        substitutions (Tuple[Substitution]): The tuple of substitutions to be made.
    """
    substitutions: Tuple[Substitution]

    def __post_init__(self):
        """
        Ensures that the variable names in the substitutions are unique.
        """
        assert len(set(sub.variable_name for sub in self.substitutions)) == len(self.substitutions)

    def __iter__(self):
        return self.substitutions.__iter__()

    @classmethod
    def from_df_row(cls, row: pd.Series) -> 'ValueCombination':
        """
        Creates a ValueCombination instance from a pandas Series.

        Args:
            row (pd.Series): Input row from a dataframe.
        
        Returns:
            ValueCombination: A ValueCombination instance.
        """
        subs = tuple([Substitution(str(n), v) for n, v in row.items() if str(n) in ascii_lowercase])
        return cls(subs)

    def copy_with_replaced_substitution(self, substitution: Substitution) -> "ValueCombination":
        """
        Returns a new ValueCombination with a specific substitution replaced.

        Args:
            substitution (Substitution): The substitution to be replaced.

        Returns:
            ValueCombination: The updated ValueCombination instance.
        """
        substitutions = [s for s in self.substitutions if s.variable_name != substitution.variable_name] + [substitution]
        return ValueCombination(tuple(substitutions))

    def as_dict(self) -> Dict[str, str]:
        """
        Converts the substitutions to a dictionary.

        Returns:
            Dict[str, str]: Dictionary with variable names as keys and their values as values.
        """
        return {sub.variable_name: sub.value for sub in self.substitutions}


@dataclass
class TemplateVariables:
    """
    Represents a set of possible values for the variables in a particular question template.

    Attributes:
        possible_values (Dict[str, List[str]]): Dictionary mapping variable names to possible values.
    """
    possible_values: Dict[str, List[str]]

    def __getitem__(self, *args, **kwargs):
        return self.possible_values.__getitem__(*args, **kwargs)

    def __setitem__(self, *args, **kwargs):
        return self.possible_values.__setitem__(*args, **kwargs)

    def items(self):
        return self.possible_values.items()

    @property
    def variable_names(self) -> List[str]:
        return list(self.possible_values.keys())

    def isolate_value(self, variable_name: str, value: str) -> 'TemplateVariables':
        """
        Returns a copy of the instance where only a specific value for a given variable is retained.

        Args:
            variable_name (str): The name of the variable.
            value (str): The value to be isolated.

        Returns:
            TemplateVariables: Updated TemplateVariables instance.
        """
        variables = copy.deepcopy(self)
        variables.possible_values[variable_name] = [value]
        return variables

    def exclude_value(self, variable_name: str, value: str) -> 'TemplateVariables':
        """
        Returns a copy of the instance excluding a specific value for a given variable.

        Args:
            variable_name (str): The name of the variable.
            value (str): The value to be excluded.

        Returns:
            TemplateVariables: Updated TemplateVariables instance.
        """
        variables = copy.deepcopy(self)
        variables.possible_values[variable_name] = [v for v in variables.possible_values[variable_name] if v != value]
        return variables

    def isolate_values(self, substitutions: List[Substitution]) -> 'TemplateVariables':
        """
        Isolates values from a list of substitutions for a variable.

        Args:
            substitutions (List[Substitution]): List of substitutions.

        Returns:
            TemplateVariables: Updated TemplateVariables instance.
        """
        variables = copy.deepcopy(self)
        variable_name = substitutions[0].variable_name
        assert all(s.variable_name == variable_name for s in substitutions)
        variables.possible_values[variable_name] = [s.value for s in substitutions]
        return variables

    def substitution_pairs(self, variable_name: str) -> List[Substitution]:
        """
        Returns a list of substitutions for a given variable.

        Args:
            variable_name (str): The name of the variable.

        Returns:
            List[Substitution]: List of substitutions.
        """
        return [Substitution(variable_name, value) for value in self[variable_name]]

    def combinations(self) -> List[ValueCombination]:
        """Generates all possible combinations of values."""
        pairs = [self.substitution_pairs(variable_name) for variable_name in self.variable_names]
        return [ValueCombination(subs) for subs in itertools.product(*pairs)]

    def sample(self, n: int, seed: int = 0, fast_with_replacement: bool = False):
        """
        Samples combinations of values.

        Args:
            n (int): Number of samples.
            seed (int, optional): Random seed. Defaults to 0.
            fast_with_replacement (bool, optional): If True, samples with replacement. Defaults to False.

        Returns:
            List[ValueCombination]: List of sampled ValueCombinations.
        """
        return self.sample_with_replacement(n, seed) if fast_with_replacement else self.sample_without_replacement(n, seed)

    def sample_with_replacement(self, n: int, seed: int = 0):
        """
        Samples combinations of values with replacement.

        Args:
            n (int): Number of samples.
            seed (int, optional): Random seed. Defaults to 0.

        Returns:
            List[ValueCombination]: List of sampled ValueCombinations.
        """
        rng = np.random.RandomState(seed)
        n_values = [len(values) for values in self.possible_values.values()]
        sample_idxs = np.array([rng.choice(v, n, replace=True) for v in n_values])
        vcs = []
        for sample_idx in sample_idxs.T:
            subs = [
                Substitution(variable_name, variable_values[i])
                for i, (variable_name, variable_values)
                in zip(sample_idx, self.possible_values.items())
            ]
            vc = ValueCombination(tuple(subs))
            vcs.append(vc)
        return vcs

    def combination_by_index(self, idx: int) -> ValueCombination:
        """
        Gets a combination by its index out of all possible combinations.

        Args:
            idx (int): Index of the combination.

        Returns:
            ValueCombination: The ValueCombination at the given index.
        """
        substitutions = []
        for variable_name in self.variable_names:
            variable_values = self.possible_values[variable_name]
            idx, variable_idx  = divmod(idx, len(variable_values))
            substitutions.append(Substitution(variable_name, variable_values[variable_idx]))
        return ValueCombination(tuple(substitutions))

    @property
    def possible_combinations(self) -> int:
        """Returns the total number of possible combinations."""
        combinations = 1
        for values in self.possible_values.values():
            combinations *= len(values)
        return  combinations

    def sample_without_replacement(self, n: int, seed: int = 0):
        """
        Samples combinations of values without replacement.

        Args:
            n (int): Number of samples.
            seed (int, optional): Random seed. Defaults to 0.

        Returns:
            List[ValueCombination]: List of sampled ValueCombinations.
        """
        possible_combinations = self.possible_combinations
        if n >= possible_combinations:
            return self.combinations()
        random.seed(seed)
        combination_idxs = random.sample(range(0, possible_combinations), n)
        return [self.combination_by_index(idx) for idx in combination_idxs]

    def example_variables(self) -> 'TemplateVariables':
        """Generates a set of variables for rendering a tempate as an example, with variables in brackets."""
        return TemplateVariables({k: [f'[{v[0]}]'] for k, v in self.items()})

@dataclass
class QuestionTemplate:
    """
    Represents a template for creating Yes/No questions.

    Attributes:
    - template: The question text with placeholders for variable substitutions.
    - variables: Possible values for the variables in the template.
    - template_id: Optional unique identifier for the template.
    - metadata: Any additional data related to the template.
    """
    template: str
    variables: TemplateVariables
    template_id: Optional[str] = None
    metadata: Dict = field(default_factory=dict)

    def make_question(self, substitutions: ValueCombination) -> YesNoQuestion:
        """Generate a YesNoQuestion using provided substitutions."""
        question_text = self.template
        for _ in range(2): # to accomodate variables in variable values
            for substitution in substitutions:
                question_text = re.sub(f'\[{substitution.variable_name}\]', str(substitution.value), question_text)  # type: ignore
        return YesNoQuestion(question_text)

    def questions(self, variables: TemplateVariables) -> List[YesNoQuestion]:
        """Generate a list of questions using a given set of variable combinations."""
        return [self.make_question(subs) for subs in variables.combinations()[:MAX_COMBINATIONS]]

    def substitutions_and_questions(self, variables: TemplateVariables) -> List[Tuple[ValueCombination, YesNoQuestion]]:
        """Generate a list of tuples containing substitutions and their corresponding questions."""
        substitutions = variables.combinations()[:MAX_COMBINATIONS]
        questions = [self.make_question(subs) for subs in substitutions]
        return list(zip(substitutions, questions))

    @property
    def all_questions(self) -> List[YesNoQuestion]:
        """Return all possible questions using the current variables."""
        return self.questions(self.variables)
    
    @property
    def variable_names(self) -> List[str]:
        """Return the names of all variables in the template."""
        return self.variables.variable_names

    @property
    def variables_example(self) -> YesNoQuestion:
        """Return an example question with sample variable values, with variables in brackets."""
        return self.questions(self.variables.example_variables())[0]  # type: ignore

    @property
    def total_questions(self) -> int:
        """Return the total number of possible questions using the current variables."""
        return len(self.variables.combinations())

    @classmethod
    def from_str(cls, template: str) -> "QuestionTemplate":
        """Create a QuestionTemplate from a template string."""
        return cls(template, TemplateVariables({name: list() for name in text_utils.bracketed_items(template)}))

    def reduce_to_5_variables(self) -> "QuestionTemplate":
        """Modify the template to use at most 5 variables."""
        n_final_variables = 5
        template = copy.deepcopy(self)
        n_to_remove = max(len(template.variable_names) - n_final_variables, 0)
        to_remove = template.variable_names[:n_to_remove]
        for variable_name in to_remove:
            offset_map = {k: v for k, v in zip(sorted(template.variable_names)[1:], sorted(template.variable_names)[:-1])}
            template.template = text_utils.revariable_text(re.sub(f'\[{variable_name}\]', template.variables[variable_name][0], template.template))  # type: ignore
            template.variables = TemplateVariables({offset_map[k]: v for k, v in template.variables.possible_values.items() if k in offset_map})
            rename_variables = {k: v for k, v in zip(sorted(template.variable_names), 'abcdefghij')}
            template.variables = TemplateVariables({rename_variables[k]: v for k, v in template.variables.possible_values.items()})
        return template

    @classmethod
    def from_dict(cls, template_data: dict) -> "QuestionTemplate":
        """Create a QuestionTemplate from a dictionary representation."""
        template_text = template_data['template']
        if 'possible_values' in template_data['variables']:
            variables = TemplateVariables(template_data['variables']['possible_values'])
        else:
            variables = TemplateVariables(template_data['variables'])
        template_id = template_data.get('template_id')
        metadata = template_data.get('metadata', dict())
        template = cls(template_text, variables, template_id, metadata)
        return template.reduce_to_5_variables()

    @classmethod
    def from_id(cls, template_id: str) -> "QuestionTemplate":
        """Load a QuestionTemplate by its unique identifier."""
        task_name = '_'.join(template_id.split('_')[:-1])
        template_path = Path('almanacs/hiring_decisions/templates/_combined/question_templates.json')
        print(template_path)
        templates = load_question_templates(template_path)
        template = [t for t in templates if t.template_id == template_id][0]
        return template

    @property
    def words(self) -> str:
        """Return a string of all words used in the template and variable values."""
        return ' '.join([self.template] + [word for k, values in self.variables.items() for word in values])

    @property
    def valid(self):
        """Check if the template meets certain criteria."""
        return (
            len(self.variable_names) == 5
            and all(v in self.template for v in ['[a]', '[b]', '[c]', '[d]', '[e]'])
            and all(len(values) >= 10 for _variable, values in self.variables.items())
        )

    def __str__(self) -> str:
        """Return a string representation of the QuestionTemplate."""
        n_values = 0
        combinations = 1
        for name, values in self.variables.items():
            n_values += len(values)
            combinations *= len(values)
        return f"""{self.template}
{self.variables_example}
Total Number of Values: {n_values}
Combinations of Values: {combinations}"""

@dataclass
class QuestionModelBehavior:
    template: QuestionTemplate
    value_combination: ValueCombination
    split: Optional[str]
    completion: str
    total_yes_no_answer_prob: float
    answer_prob: float
    cot: bool
    valid_answer: bool

    @property
    def template_id(self):
        return self.template.template_id

    @property
    def question_text(self):
        """Generates and returns the question text using the template and value combination."""
        return self.template.make_question(self.value_combination).text

    def as_record(self) -> dict:
        """Converts the behavior to a record (dictionary) format."""
        return dict(
            template_id=self.template_id,
            question_text=self.question_text,
            split=self.split,
            completion=self.completion,
            answer_prob=self.answer_prob,
            total_answer_prob=self.total_yes_no_answer_prob,
            cot=self.cot,
            valid_answer=self.valid_answer,
            **self.value_combination.as_dict()
        )
    
    @classmethod
    def from_df_row(cls, row, template: Optional[QuestionTemplate] = None):
        """Instantiates a `QuestionModelBehavior` object from a DataFrame row."""
        if template is None:
            template = QuestionTemplate.from_id(row.template_id)
        return QuestionModelBehavior(template, ValueCombination.from_df_row(row), row.split, row.completion, row.total_answer_prob, row.answer_prob, row.cot, row.valid_answer)

@dataclass
class TemplateModelBehavior:
    template: QuestionTemplate
    question_behaviors: List[QuestionModelBehavior]
    model_name: str

    model_behavior_filename = 'model_behavior.csv'
    split_summaries_filename = 'split_summaries.json'

    def __post_init__(self):
        """Initializes the DataFrame and splits after object instantiation."""
        self.df = pd.DataFrame.from_records([b.as_record() for b in self.question_behaviors])
        self.split_dfs = self.to_splits() if len(self.split_names) > 1 else None
    
    @property
    def template_id(self) -> Optional[str]:
        return self.template.template_id

    @property
    def split_names(self) -> List[str]:
        """Returns a list of unique split names from the DataFrame."""
        assert self.df is not None
        return self.df.split.unique().tolist()

    @property
    def answer_probs(self) -> np.ndarray:
        return self.df.answer_prob.to_numpy()

    @property
    def mean_total_yes_no_answer_prob(self) -> float:
        return np.mean([qb.total_yes_no_answer_prob for qb in self.question_behaviors])  # type: ignore

    @property
    def etvd(self) -> float:
        return metrics.within_population_etvd(self.answer_probs)  # type: ignore
    
    @property
    def positive_fraction(self) -> float:
        return metrics.positive_fraction(self.answer_probs)  # type: ignore

    @property
    def answer_is_valid(self) -> np.ndarray:
        return self.df.valid_answer.to_numpy()

    def question_behaviors_for_split(self, split: str) -> List[QuestionModelBehavior]:
        """Returns a list of question behaviors for the given split."""
        return [b for b in self.question_behaviors if b.split == split] 

    def to_splits(self) -> Dict[str, "TemplateModelBehavior"]:
        """Splits the behaviors based on their splits and returns a dictionary of split to behavior mapping."""
        return {s: TemplateModelBehavior(self.template, self.question_behaviors_for_split(s), self.model_name) for s in self.split_names}

    def questions(self, split: Optional[str] = None) -> List[str]:
        """Returns a list of questions for a specified split or for all splits if none is specified."""
        if split:
            assert self.split_dfs is not None
            return self.split_dfs[split].questions()
        return self.df.question_text.to_list()

    def answers(self, split: Optional[str] = None) -> np.ndarray:
        """Returns a numpy array of answers for a specified split or for all splits if none is specified."""
        if split:
            assert self.split_dfs is not None
            return self.split_dfs[split].answers()
        return self.df.answer_prob.to_numpy()

    def value_combinations(self, split: Optional[str] = None) -> List[ValueCombination]:
        """Returns a list of value combinations for a specified split or for all splits if none is specified."""
        return [b.value_combination for b in self.question_behaviors if not split or b.split == split]

    @classmethod
    def from_answers(cls, template: QuestionTemplate, variable_values: List[ValueCombination], answers: AnswerSet, model_name: str, split_name: Optional[str] = None, cot: bool = False):
        """Instantiates a `TemplateModelBehavior` object based on given answers."""
        if cot:
            valid_answers = [a is not None for a in answers.given_answers]
            total_answer_probs = [float(v) for v in valid_answers]
            answer_probs = [round(a) if a is not None else 0.5 for a in answers.given_answers]
        else:
            total_answer_probs = [a.completion.total_answer_prob for a in answers]
            answer_probs = answers.positive_answer_probs
            valid_answers = answers.valid_answers
        question_behaviors = [
            QuestionModelBehavior(template, value_combination, split_name, completion.text, total_answer_prob, answer_prob, cot, valid_answer)
            for value_combination, answer_prob, total_answer_prob, completion, valid_answer
            in zip(variable_values, answer_probs, total_answer_probs, answers.completions, valid_answers)
        ]
        return cls(template, question_behaviors, model_name)

    @classmethod
    def load(cls, results_path: Path) -> "TemplateModelBehavior":
        print("---***---")
        print(cls, results_path)
        """Loads the model behavior from the provided path."""
        model_behavior = pd.read_csv(results_path / cls.model_behavior_filename)
        model_name = json.loads((results_path / cls.split_summaries_filename).read_text())['summary']['model_name']
        question_behaviors = []
        template = QuestionTemplate.from_id(model_behavior.template_id[0])
        question_behaviors = [QuestionModelBehavior.from_df_row(row, template) for _, row in model_behavior.iterrows()]
        return cls(template, question_behaviors, model_name)

    @property
    def summary(self) -> Dict:
        """Returns a summary dictionary of the model behavior."""
        return {
                'template_id': self.template.template_id,
                'model_name': self.model_name,
                'n_samples': len(self.question_behaviors),
                'etvd': self.etvd,
                'positive_fraction': self.positive_fraction,
                'valid_fraction': self.answer_is_valid.mean(),
                'total_answer_prob': self.mean_total_yes_no_answer_prob,
            }

    def save(self, save_path: Path) -> None:
        """Saves the model behavior to the provided path."""
        save_path.mkdir(exist_ok=True, parents=True)
        self.df.to_csv(save_path / self.model_behavior_filename, index_label='question_id')

        summary_data = dict(summary=self.summary)
        if len(self.split_names) > 1:
            summary_data.update({s: mb.summary for s, mb in self.to_splits().items()})
        with open(save_path / self.split_summaries_filename, 'w') as f:
            json.dump(summary_data, f, indent=4, ensure_ascii=True)

def load_question_templates(path: Path) -> List[QuestionTemplate]:
    """
    Load a list of QuestionTemplates from a specified file path.

    Args:
    - path: The file path to load templates from.

    Returns:
    - A list of valid QuestionTemplates.
    """
    task_data = json.loads((path).read_text(encoding="utf-8"))
    question_templates = [QuestionTemplate.from_dict(q) for q in task_data['questions']]
    return [t for t in question_templates if t.valid]

def load_model_behavior():
    behavior_path = Path("./almanacs/hiring_decisions/final_dataset")
    # behavior_path = Path("./final_dataset/hiring_decisions_29")
    model_behaviors = []
    # for model_behavior_dir in tqdm.tqdm(list([behavior_path]), desc='Loading Model Behavior'):
    for model_behavior_dir in tqdm.tqdm(list(behavior_path.iterdir()), desc='Loading Model Behavior'):
        if not model_behavior_dir.is_dir():
            continue
        print(model_behavior_dir)
        model_behavior = TemplateModelBehavior.load(model_behavior_dir)
        model_behaviors.append(model_behavior)
    return model_behaviors

def main():
    model_behaviors = load_model_behavior()
    # TODO 1: Add methods to preprocess the entire dataset
    split2selecteddata = {}
    data = []
    for split in ['test']:
        for idx in range(len(model_behaviors)):
            for idxqs in range(len(model_behaviors[idx].questions())):
                data.append({'question': model_behaviors[idx].questions()[idxqs], 
                            'answers': model_behaviors[idx].answers()[idxqs]},)
        split2selecteddata[split] = data
    json.dump(split2selecteddata, open('data/data_hiring_decisions.json', 'w'), indent=4)

if __name__ == '__main__':
    main()