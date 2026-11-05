"""ЛР №7 — N-грамна система автозавершення (1–5-грам).

Запуск: python lab7.py --corpus data/shakespeare.txt --output results
Навчальна модель: add-k згладжування, однаковий train/test split, log-perplexity.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import random
import re
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

START, END, UNK = '<s>', '</s>', '<unk>'
TOKEN_PATTERN = re.compile(r"[a-z]+(?:['’][a-z]+)?", flags=re.IGNORECASE)


def load_data(file_path: str | Path) -> str:
    """Читає вихідний UTF-8 корпус."""
    return Path(file_path).read_text(encoding='utf-8')


def split_to_sentences(data: str) -> list[str]:
    """За методичкою кожен непорожній рядок корпусу = окрема послідовність."""
    return [s.strip() for s in data.splitlines() if s.strip()]


def tokenize_sentences(sentences: Iterable[str]) -> list[list[str]]:
    """Англійська токенізація, збереження внутрішнього апострофа, lowercase."""
    return [tokens for sent in sentences if (tokens := TOKEN_PATTERN.findall(sent.lower()))]


def get_tokenized_data(data: str) -> list[list[str]]:
    return tokenize_sentences(split_to_sentences(data))


def count_words(tokenized_sentences: Iterable[list[str]]) -> Counter:
    return Counter(word for sentence in tokenized_sentences for word in sentence)


def get_words_with_nplus_frequency(tokenized_sentences: list[list[str]], count_threshold: int = 2) -> list[str]:
    if count_threshold < 1:
        raise ValueError('count_threshold має бути >= 1')
    return sorted(w for w, c in count_words(tokenized_sentences).items() if c >= count_threshold)


def replace_oov_words_by_unk(tokenized_sentences: list[list[str]], vocabulary: Iterable[str], unknown_token: str = UNK) -> list[list[str]]:
    allowed = set(vocabulary)
    return [[w if w in allowed else unknown_token for w in sentence] for sentence in tokenized_sentences]


def preprocess_data(train_data: list[list[str]], test_data: list[list[str]], count_threshold: int = 2):
    """Словник ТІЛЬКИ з train; ніякого витоку тестових даних."""
    vocabulary = get_words_with_nplus_frequency(train_data, count_threshold)
    return (replace_oov_words_by_unk(train_data, vocabulary),
            replace_oov_words_by_unk(test_data, vocabulary), vocabulary)


def count_n_grams(data: list[list[str]], n: int, start_token: str = START, end_token: str = END) -> Counter:
    if not 1 <= n <= 5:
        raise ValueError('n має належати 1..5')
    counts = Counter()
    for sentence in data:
        seq = [start_token] * (n - 1) + list(sentence) + [end_token]
        counts.update(tuple(seq[i:i+n]) for i in range(len(seq)-n+1))
    return counts


@dataclass
class NgramLanguageModel:
    """Власна N-грамна модель із згладжуванням Лапласа/Add-k."""
    vocabulary: list[str]
    counts: dict[int, Counter] = field(default_factory=dict)
    history_counts: dict[int, Counter] = field(default_factory=dict)
    k: float = 0.5
    _vocab_set: set[str] = field(default_factory=set, repr=False)

    @classmethod
    def fit(cls, train_sentences: list[list[str]], vocabulary: list[str], k: float = 0.5):
        if k <= 0:
            raise ValueError('k має бути > 0 для оцінювання перплексії')
        allowed = set(vocabulary) | {UNK, END}
        if any(START in sentence or END in sentence for sentence in train_sentences):
            raise ValueError('Спеціальні маркери в train data використовувати не можна')
        words = sorted(allowed)
        model = cls(vocabulary=words, k=k, _vocab_set=set(words))
        for n in range(1, 6):
            grams = count_n_grams(train_sentences, n)
            model.counts[n] = grams
            # Знаменник із тих самих n-грам, що й чисельник:
            # точна сума виходів з історії, без підрахунку непарних EOS.
            model.history_counts[n] = Counter({(): sum(grams.values())}) if n == 1 else Counter()
            if n > 1:
                for gram, c in grams.items():
                    model.history_counts[n][gram[:-1]] += c
        return model

    def normalize_history(self, context: Iterable[str], n: int) -> tuple[str, ...]:
        if not 1 <= n <= 5:
            raise ValueError('n має належати 1..5')
        if n == 1:
            return ()
        seq = [w if w in self._vocab_set or w == START else UNK for w in context]
        return tuple(([START] * (n-1) + seq)[-(n-1):])

    def probability(self, word: str, context: Iterable[str], n: int, k: float | None = None) -> float:
        smooth = self.k if k is None else k
        if smooth <= 0:
            raise ValueError('k має бути додатним')
        if word not in self._vocab_set:
            word = UNK
        history = self.normalize_history(context, n)
        observed = self.counts[n].get(history + (word,), 0)
        total = self.history_counts[n].get(history, 0)
        return (observed + smooth) / (total + smooth * len(self.vocabulary))

    def suggest(self, context: str | list[str], n: int = 3, top_k: int = 5,
                start_with: str = '', k: float | None = None, backoff: bool = True) -> tuple[list[tuple[str, float]], int]:
        """Пропозиції з backoff при небаченій історії; повертає використаний порядок."""
        if isinstance(context, str):
            tokens = TOKEN_PATTERN.findall(context.lower())
        else:
            tokens = list(context)
        order = n
        if backoff:
            while order > 1 and self.history_counts[order].get(self.normalize_history(tokens, order), 0) == 0:
                order -= 1
        prefix = start_with.lower()
        candidates = (word for word in self.vocabulary if word not in {END, UNK, START} and word.startswith(prefix))
        scores = [(w, self.probability(w, tokens, order, k)) for w in candidates]
        scores.sort(key=lambda p: (-p[1], p[0]))
        return scores[:max(0, top_k)], order

    def perplexity(self, test_sentences: list[list[str]], n: int, k: float | None = None) -> float:
        """PP = exp(-Σ log P(w|history) / M), M — кількість токенів разом з EOS."""
        neg_log_likelihood = 0.0
        num_predictions = 0
        for sentence in test_sentences:
            history = [START] * (n-1)
            for word in list(sentence) + [END]:
                p = self.probability(word, history, n, k)
                neg_log_likelihood -= math.log(p)
                num_predictions += 1
                history.append(word)
        if not num_predictions:
            raise ValueError('Тестова вибірка порожня')
        return math.exp(neg_log_likelihood / num_predictions)


def estimate_probability(word, previous_n_gram, n_gram_counts, n_plus1_gram_counts, vocabulary_size, k=1.0):
    """Функція у формі, близькій до методички: попередні n слів і (n+1)-грам."""
    if k <= 0 or vocabulary_size <= 0:
        raise ValueError('k і vocabulary_size мають бути додатними')
    history = tuple(previous_n_gram)
    numerator = n_plus1_gram_counts.get(history+(word,), 0) + k
    # Знаменник рахуємо лише з дійсних виходів, як у NgramLanguageModel:
    ngram_history_frequency = sum(c for gram, c in n_plus1_gram_counts.items() if gram[:-1] == history)
    return numerator / (ngram_history_frequency + k * vocabulary_size)


def _write_csv(path: Path, rows: list[dict], columns: list[str]):
    with path.open('w', newline='', encoding='utf-8-sig') as f:
        wr = csv.DictWriter(f, fieldnames=columns)
        wr.writeheader()
        wr.writerows(rows)


def make_website_model(model: NgramLanguageModel, path: str | Path, limit_per_n: int = 1000):
    """Компактна реальна модель для статичного GitHub Pages: топ-контексти і слова.

    JS-додаток використовує лише підмножину контекстів; Python/Colab — повну модель.
    """
    data = {'source': 'shakespeare.txt', 'k': model.k, 'vocabulary_size': len(model.vocabulary), 'models': {}}
    for n in range(1, 6):
        histories = [()] if n == 1 else [ctx for ctx, _ in model.history_counts[n].most_common(limit_per_n)]
        rows = {}
        by_history = defaultdict(list)
        chosen_histories = set(histories)
        for gram, c in model.counts[n].items():
            if gram[:-1] in chosen_histories and gram[-1] not in {END, UNK}:
                by_history[gram[:-1]].append((gram[-1], c))
        for history in histories:
            # Використовуємо лише слова, які реально траплялися в контексті,
            # а для пустих історій JS знизить порядок до 1-грам.
            observed = by_history.get(history, [])
            observed.sort(key=lambda x: (-x[1], x[0]))
            total = model.history_counts[n].get(history, 0)
            scores = [[w, round((c+model.k)/(total+model.k*len(model.vocabulary)), 7)] for w, c in observed[:20]]
            rows[' '.join(history)] = scores
        data['models'][str(n)] = rows
    Path(path).write_text(json.dumps(data, ensure_ascii=False, separators=(',', ':')), encoding='utf-8')
    return data


def evaluate(corpus_path: str | Path, output: str | Path = 'results', count_threshold: int = 2,
             k: float = 0.5, seed: int = 42, train_ratio: float = 0.8,
             web_model_path: str | Path | None = None):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    sentences = get_tokenized_data(load_data(corpus_path))
    if len(sentences) < 10:
        raise ValueError('Потрібно не менше 10 непорожніх рядків з англійськими словами')
    rng = random.Random(seed)
    rng.shuffle(sentences)
    pivot = int(len(sentences) * train_ratio)
    if not 1 <= pivot < len(sentences):
        raise ValueError('train_ratio має створювати непорожні train/test')
    train_raw, test_raw = sentences[:pivot], sentences[pivot:]
    train, test, vocab = preprocess_data(train_raw, test_raw, count_threshold)
    model = NgramLanguageModel.fit(train, vocab, k=k)
    load_time = time.perf_counter()-t0
    rows = []
    for n in range(1, 6):
        start = time.perf_counter()
        pp = model.perplexity(test, n)
        duration = (time.perf_counter()-start)*1000
        rows.append({'n': n, 'model': f'{n}-грами', 'perplexity': round(pp, 5),
                     'ngrams_unique': len(model.counts[n]), 'evaluation_ms': round(duration, 3)})
    examples = ['the king', 'my love', 'i shall', 'to the', 'and the', 'you are']
    samples = []
    for phrase in examples:
        for n in [2, 3, 5]:
            top, used_n = model.suggest(phrase, n=n, top_k=5)
            samples.append({'context': phrase, 'requested_n': n, 'used_n': used_n,
                            'suggestions': [{'word': w, 'probability': round(p, 6)} for w, p in top]})
    # Для перевірки нормування умовних імовірностей.
    norm_sum = sum(model.probability(w, ['the'], 2) for w in model.vocabulary)
    if not math.isclose(norm_sum, 1.0, abs_tol=1e-9):
        raise AssertionError(f'Ймовірності не нормовані: {norm_sum}')
    oov_test = sum(w == UNK for sent in test for w in sent)
    total_test_tokens = sum(len(sent) for sent in test)
    metrics = {'corpus_name': Path(corpus_path).name, 'seed': seed,
               'train_ratio': train_ratio, 'min_word_frequency': count_threshold,
               'smoothing_k': k, 'train_sentences': len(train), 'test_sentences': len(test),
               'train_tokens': sum(len(sent) for sent in train_raw),
               'test_tokens': total_test_tokens,
               'vocabulary_size': len(model.vocabulary), 'rare_unk_count_test': oov_test,
               'test_oov_rate': round(oov_test / total_test_tokens, 6),
               'training_seconds': round(load_time, 3), 'perplexities': rows,
               'best_model': min(rows, key=lambda r:r['perplexity'])['model']}
    (output/'metrics.json').write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding='utf-8')
    (output/'suggestions.json').write_text(json.dumps(samples, ensure_ascii=False, indent=2), encoding='utf-8')
    _write_csv(output/'perplexity.csv', rows, ['n', 'model', 'perplexity', 'ngrams_unique', 'evaluation_ms'])
    report = f'''# Звіт: Лабораторна робота №7

**Тема:** Система автозавершення тексту на основі N-грам.

**Корпус:** `{Path(corpus_path).name}` (англійський текст Шекспіра, наданий у завданні).

## Методика
- Текст поділено за непорожніми рядками (кожен рядок — послідовність).
- Токенізація англійських слів (нижній регістр, без пунктуації, апостроф у слові зберігається).
- Розбиття на навчальну ({train_ratio:.0%}) та тестову ({1-train_ratio:.0%}) вибірки із seed={seed}.
- Словник формується **тільки за навчальною вибіркою**; слова рідше {count_threshold} разів замінюються на `<unk>` в обох вибірках.
- Моделі: уніграми, біграми, триграми, 4-грами, 5-грами. Маркери початку `<s>` і кінця `</s>`.
- Обчислення умовних ймовірностей: **Add-k**, k={k}. Перплексія рахується за log-імовірностями лише на відкладеному тесті, включно з кінцем речення, без `<s>`.
- Для автозавершення підтримуються top-5 пропозицій, фільтрація за першими літерами, backoff до коротшого контексту.

## Показники даних
- Навчальних рядків: **{len(train):,}**; тестових рядків: **{len(test):,}**.
- Навчальних токенів: **{sum(map(len,train_raw)):,}**; тестових токенів: **{total_test_tokens:,}**.
- Словник включно зі службовими `<unk>` та `</s>`: **{len(model.vocabulary):,}**.
- Частка невідомих слів у тесті: **{oov_test / total_test_tokens * 100:.2f}%**.

## Перплексія на тестовому наборі
| Модель | Перплексія ↓ | Унікальних N-грам | Час оцінки, мс |
|---|---:|---:|---:|
'''
    for row in rows:
        report += f"| {row['model']} | {row['perplexity']:.3f} | {row['ngrams_unique']:,} | {row['evaluation_ms']:.1f} |\n"
    report += f'''\n**Найменша перплексія:** {metrics['best_model']}. Вона свідчить про найкраще середнє передбачення відкладених токенів у цьому експерименті, але не гарантує найкращу якість будь-якого завершення тексту.

## Приклади передбачень
'''
    for sample in samples:
        report += f"- `{sample['context']}` ({sample['requested_n']}-грами; використано {sample['used_n']}): "
        report += ', '.join(f"{a['word']} ({a['probability']:.4f})" for a in sample['suggestions']) + '\n'
    report += '''\n## Висновок
Реалізовано всі п’ять порядків N-грам, згладжування Add-k, пошук наступних слів, інтерфейс на ipywidgets та оцінювання перплексії. Показано залежність якості від довжини контексту та розрідженості 4–5-грам. Результати залежать від корпусу, порога частот і згладжування. Статична вебдемонстрація використовує лише найбільш поширені контексти; повна модель доступна у Python / Colab.\n'''
    (output/'REPORT.md').write_text(report, encoding='utf-8')
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.plot([r['n'] for r in rows], [r['perplexity'] for r in rows], marker='o')
        ax.set(xlabel='Порядок N-грамної моделі', ylabel='Перплексія (нижче — краще)', title='Порівняння N-грамних моделей на тестовому наборі')
        ax.set_xticks([1,2,3,4,5]); ax.grid(alpha=.25)
        fig.tight_layout(); fig.savefig(output/'perplexity.png', dpi=150); plt.close(fig)
        fig, ax = plt.subplots(figsize=(9,5))
        ax.bar([str(r['n']) for r in rows], [r['ngrams_unique'] for r in rows])
        ax.set(xlabel='Порядок N-грам', ylabel='Кількість унікальних N-грам', title='Розмір моделей')
        fig.tight_layout(); fig.savefig(output/'ngram_counts.png', dpi=150); plt.close(fig)
    except ImportError:
        print('matplotlib не встановлено: PNG-графіки пропущені')
    if web_model_path is not None:
        make_website_model(model, web_model_path)
    return metrics, model


def create_ui(model: NgramLanguageModel):
    """Інтерактивна форма для Jupyter / Google Colab (вимагає ipywidgets)."""
    import ipywidgets as widgets
    from IPython.display import display, clear_output
    input_text = widgets.Text(value='the king', description='Текст:', placeholder='Введіть англійський текст...',
                              layout=widgets.Layout(width='620px'))
    prefix_input = widgets.Text(value='', description='Початок:', placeholder='Перші літери (необовʼязково)',
                                layout=widgets.Layout(width='620px'))
    order = widgets.Dropdown(options=[(f'{i}-грами', i) for i in range(1, 6)], value=3, description='Модель:')
    smooth = widgets.FloatSlider(value=model.k, min=.05, max=2.0, step=.05, description='Add-k:', continuous_update=False)
    button = widgets.Button(description='Запропонувати', button_style='info', icon='search')
    output = widgets.Output()

    def append_word(w):
        input_text.value = input_text.value.rstrip() + (' ' if input_text.value.strip() else '') + w + ' '

    def update(_):
        with output:
            clear_output()
            suggestions, used_order = model.suggest(input_text.value, n=order.value, top_k=5,
                                                     start_with=prefix_input.value.strip(), k=smooth.value)
            if used_order < order.value:
                print(f'Невідомий контекст: використано backoff до {used_order}-грам.')
            if not suggestions:
                print('Немає слів із таким початком.')
                return
            buttons = []
            for word, prob in suggestions:
                btn = widgets.Button(description=f'{word} ({prob:.4f})', layout=widgets.Layout(width='auto'))
                btn.on_click(lambda _, w=word: append_word(w))
                buttons.append(btn)
            display(widgets.HBox(buttons, layout=widgets.Layout(flex_flow='row wrap')))

    button.on_click(update)
    display(widgets.VBox([input_text, prefix_input, widgets.HBox([order, smooth]), button, output]))
    update(None)
    return {'text': input_text, 'prefix': prefix_input, 'order': order, 'k': smooth}


def main():
    parser = argparse.ArgumentParser(description='Лабораторна №7 — автозавершення тексту N-грамами')
    parser.add_argument('--corpus', default='data/shakespeare.txt')
    parser.add_argument('--output', default='results')
    parser.add_argument('--website-model', default='docs/model.json')
    parser.add_argument('--k', type=float, default=0.5)
    parser.add_argument('--threshold', type=int, default=2)
    args = parser.parse_args()
    path = Path(args.website_model)
    path.parent.mkdir(parents=True, exist_ok=True)
    metrics, _ = evaluate(args.corpus, args.output, k=args.k, count_threshold=args.threshold, web_model_path=path)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    print('Результати збережено у', args.output)


if __name__ == '__main__':
    main()
