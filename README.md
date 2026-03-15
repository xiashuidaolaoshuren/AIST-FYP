The output data link:
https://drive.google.com/drive/folders/1VgwXwY9St5GP56ln2mcUAb6St-bZ2fAs?usp=drive_link

## Environment Setup (uv)

```bash
uv sync
uv run python -m spacy download en_core_web_sm
```

Run scripts with:

```bash
uv run python scripts/demo_ragtruth_eval.py
```

## Local Demo Model

The local demo now defaults to:

```text
Qwen/Qwen2.5-0.5B-Instruct
```

To try another model temporarily without editing config, use runtime override:

```bash
uv run python scripts/demo_baseline_rag.py --model-name google/flan-t5-base
```

To switch back permanently, edit the generator in config.yaml under models.generator.

[First Term Presentation Slide](https://www.canva.com/design/DAG43O9pK_4/GrD26I-QEEr1LYCYID6U9Q/edit?utm_content=DAG43O9pK_4&utm_campaign=designshare&utm_medium=link2&utm_source=sharebutton)

[First Term Report](https://docs.google.com/document/d/1i1Lt_hWrULz57UIjs9ziN3rm5J1KvwTyYLjfQ5G1eQk/edit?usp=sharing)
