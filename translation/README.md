# Toxicity Mitigation via Back-Translation

This program reduces text toxicity through back-translation using Facebook's M2M100 model. It supports 100+ languages for diverse augmentation.

## Setup

### Prerequisites
- Python 3.10+

### Installation
1. Clone repository:
```bash
git clone https://github.com/anthol42/GLO-7030
cd translation
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

Run pipeline with default settings (Japanese intermediate):
```bash
python main.py
```

Single intermediate language back-translation (Korean example):
```bash
python main.py --inter_lang ko
```

Multiple intermediate languages back-translation (French, German, Spanish, Chinese):
```bash
python main.py --inter_lang fr,de,es,zh
```

## Best Languages for Diversity

For maximum semantic variation, use these language codes:

| Language       | Code | Linguistic Features                     | Example Transformation |
|----------------|------|------------------------------------------|-------------------------|
| Japanese       | `ja` | SOV structure, no plurals               | "They" → "That person" |
| Korean         | `ko` | Agglutinative morphology                 | "Quickly ran" → "Ran with speed" |
| Chinese        | `zh` | Tonal, character-based                   | "Happy" → "Joyful heart feeling" |
| Arabic         | `ar` | Right-to-left, VSO structure             | "He ate" → "Ate he" |
| Finnish        | `fi` | 15 grammatical cases                     | "Of the house" → "House's from" |
| Hindi          | `hi` | Gender-specific verbs                     | "He/She went" → "Went(male)/Went(female)" |
| Turkish        | `tr` | Vowel harmony                            | "House-plural" → "Houses-specialform" |

## Output Format

Generated CSV contains:
- Original text (`body`)
- Back-translated text (`translated`)
- Toxicity score (`score`)
