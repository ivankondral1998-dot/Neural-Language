"""Reproducible synthetic-only held-out compression probe; no LM training."""
import argparse
import json
from pathlib import Path
import tempfile

from dialogue_tokenizer import train_tokenizer

TRAIN_WORDS = ('солнце город дерево книга вода река дорога окно дом машина '
               'наука число компьютер память обучение язык программа текст '
               'утро вечер свет ветер работа вопрос ответ данные модель').split()
HELDOUT = {
    'russian': [
        'Сегодня мы проверяем, как программа сохраняет русский текст.',
        'Ёжик смотрит на дорогу, а ветер качает ветви деревьев.',
        'Обучение модели требует данных и независимой проверки результата.',
    ],
    'english': ['A small model needs data and an independent evaluation.'],
    'mixed': ['UTF-8: Ё, café, 中文, 🙂. Literal <|assistant|> stays text.'],
}


def probe(output):
    output.mkdir(parents=True, exist_ok=False)
    with tempfile.TemporaryDirectory() as td:
        corpus = Path(td)/'synthetic.txt'
        lines = [f'Слова: {a}, {b}. Мы изучаем язык и данные.\n'
                 for a in TRAIN_WORDS for b in TRAIN_WORDS]
        lines += ['A program learns from text. Data and model evaluation.\n']*20
        corpus.write_text(''.join(lines),encoding='utf-8')
        tok = train_tokenizer([corpus],output/'tokenizer.json')
    groups = {}
    for name,texts in HELDOUT.items():
        encoded = [tok.encode(text) for text in texts]
        if any(tok.decode(ids) != text for text,ids in zip(texts,encoded)):
            raise ValueError('held-out UTF-8 roundtrip failed')
        byte_count = sum(len(text.encode('utf-8')) for text in texts)
        token_count = sum(map(len,encoded))
        groups[name] = {'utf8_bytes':byte_count,'bpe_tokens':token_count,
                        'bytes_per_bpe_token':byte_count/token_count,
                        'roundtrip':True, 'messages':len(texts)}
    result = {'scope':'synthetic held-out tokenizer compression only; no model training',
              'dependency':'tokenizers==0.22.2','kind':tok.kind,'vocab_size':tok.vocab_size,
              'tokenizer_sha256':tok.digest, 'heldout_groups':groups,
              'profile_parameters_formula':52684608+(tok.vocab_size-260)*768,
              'native_nir_backend':False, 'conversational_ability_proven':False,
              'android_measured':False}
    (output/'compression.json').write_text(json.dumps(result,indent=2)+'\n')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',type=Path,required=True)
    args = parser.parse_args()
    try:
        print(json.dumps(probe(args.output_dir),indent=2))
    except (OSError,ValueError) as exc:
        parser.exit(2,f'PROBE ERROR: {exc}\n')
