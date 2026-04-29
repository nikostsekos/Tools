#!/usr/bin/env python3


from __future__ import annotations


import argparse

from typing import Iterable


from datasets import Dataset, DatasetDict, Value, load_dataset

from presidio_analyzer import AnalyzerEngine, Pattern, PatternRecognizer

from presidio_anonymizer import AnonymizerEngine


def build_analyzer() -> AnalyzerEngine:

    analyzer = AnalyzerEngine()


    greek_phone = Pattern(

        name="greek_phone_pattern",

        regex=r"\b(?:\+30\s?)?(?:2\d{9}|69\d{8})\b",

        score=0.45,

    )

    afm = Pattern(

        name="greek_afm_pattern",

        regex=r"\b\d{9}\b",

        score=0.35,

    )

    amka = Pattern(

        name="greek_amka_pattern",

        regex=r"\b\d{11}\b",

        score=0.45,

    )


    analyzer.registry.add_recognizer(

        PatternRecognizer(

            supported_entity="GR_PHONE_NUMBER",

            patterns=[greek_phone],

            context=["τηλέφωνο", "κινητό", "phone", "mobile", "τηλ"],

        )

    )

    analyzer.registry.add_recognizer(

        PatternRecognizer(

            supported_entity="GR_AFM",

            patterns=[afm],

            context=["αφμ", "tax", "vat", "tin"],

        )

    )

    analyzer.registry.add_recognizer(

        PatternRecognizer(

            supported_entity="GR_AMKA",

            patterns=[amka],

            context=["αμκα", "social security"],

        )

    )

    return analyzer


def detect_string_columns(dataset: Dataset) -> list[str]:

    return [

        col

        for col, feature in dataset.features.items()

        if isinstance(feature, Value) and feature.dtype == "string"

    ]


def anonymize_text(

    text: str,

    analyzer: AnalyzerEngine,

    anonymizer: AnonymizerEngine,

    language: str,

) -> str:

    findings = analyzer.analyze(text=text, language=language)

    result = anonymizer.anonymize(text=text, analyzer_results=findings)

    return result.text


def parse_fields(fields_arg: str | None) -> list[str] | None:

    if not fields_arg:

        return None

    fields = [f.strip() for f in fields_arg.split(",") if f.strip()]

    return fields or None


def process_split(

    split_ds: Dataset,

    fields: list[str],

    analyzer: AnalyzerEngine,

    anonymizer: AnonymizerEngine,

    language: str,

    replace_original: bool,

) -> Dataset:

    def _anonymize_record(record: dict) -> dict:

        updated = dict(record)

        for field in fields:

            value = record.get(field)

            if isinstance(value, str) and value:

                anonymized = anonymize_text(

                    text=value,

                    analyzer=analyzer,

                    anonymizer=anonymizer,

                    language=language,

                )

                if replace_original:

                    updated[field] = anonymized

                else:

                    updated[f"{field}_anonymized"] = anonymized

        return updated


    return split_ds.map(_anonymize_record, desc="Anonymizing records")


def iter_splits(dataset_obj: DatasetDict, selected_split: str) -> Iterable[tuple[str, Dataset]]:

    if selected_split.lower() == "all":

        for split_name, split_ds in dataset_obj.items():

            yield split_name, split_ds

        return


    if selected_split not in dataset_obj:

        raise ValueError(

            f"Split '{selected_split}' not found. Available splits: {list(dataset_obj.keys())}"

        )

    yield selected_split, dataset_obj[selected_split]


def main() -> None:

    parser = argparse.ArgumentParser()

    parser.add_argument(

        "--dataset",

        default="glossAPI/opengov.gr-deliberation-V2",

    )

    parser.add_argument(

        "--split",

        default="all",

    )

    parser.add_argument(

        "--text-fields",

        default=None,

    )

    parser.add_argument(

        "--output-dir",

        default="./opengov_anonymized",

    )

    parser.add_argument(

        "--replace-original",

        action="store_true",

    )

    parser.add_argument(

        "--language",

        default="en",

    )

    parser.add_argument(

        "--limit",

        type=int,

        default=None,

    )

    args = parser.parse_args()


    raw = load_dataset(args.dataset)

    if isinstance(raw, Dataset):

        dataset_obj = DatasetDict({"train": raw})

    else:

        dataset_obj = raw


    analyzer = build_analyzer()

    anonymizer = AnonymizerEngine()


    split_refs = list(iter_splits(dataset_obj, args.split))

    if not split_refs:

        raise ValueError("No splits selected for processing.")


    first_split_ds = split_refs[0][1]

    fields = parse_fields(args.text_fields) or detect_string_columns(first_split_ds)

    if not fields:

        raise ValueError(

            "No string columns found. Pass --text-fields explicitly with comma-separated column names."

        )


    for field in fields:

        if field not in first_split_ds.column_names:

            raise ValueError(

                f"Column '{field}' not found in split '{split_refs[0][0]}'. "

                f"Available: {first_split_ds.column_names}"

            )


    print(f"Dataset: {args.dataset}")

    print(f"Selected split(s): {[name for name, _ in split_refs]}")

    print(f"Text columns: {fields}")

    print(f"Mode: {'replace original' if args.replace_original else 'append *_anonymized'}")


    anonymized = {}

    for split_name, split_ds in split_refs:

        working_split = split_ds.select(range(min(args.limit, len(split_ds)))) if args.limit else split_ds

        anonymized[split_name] = process_split(

            split_ds=working_split,

            fields=fields,

            analyzer=analyzer,

            anonymizer=anonymizer,

            language=args.language,

            replace_original=args.replace_original,

        )


    out = DatasetDict(anonymized)

    out.save_to_disk(args.output_dir)

    print(f"Anonymized dataset saved to: {args.output_dir}")


if __name__ == "__main__":

    main()

