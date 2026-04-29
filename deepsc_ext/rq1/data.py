"""Template data generation for RQ1 task-level semantic preservation."""

import argparse
import random
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping

from deepsc_ext.rq1.common import ensure_dir, write_jsonl


INTENT_TEMPLATES: Mapping[str, List[str]] = {
    "book_flight": [
        "Book a flight from {from_city} to {to_city} on {date}.",
        "I need a flight from {from_city} to {to_city} {date}.",
        "Please reserve a ticket from {from_city} to {to_city} for {date}.",
    ],
    "check_weather": [
        "What is the weather in {city} on {date}?",
        "Tell me the weather forecast for {city} {date}.",
        "Will it rain in {city} on {date}?",
    ],
    "set_alarm": [
        "Set an alarm for {time} tomorrow.",
        "Wake me up at {time} tomorrow.",
        "Please set my alarm to {time}.",
    ],
    "play_music": [
        "Play {song} by {artist}.",
        "I want to listen to {song} by {artist}.",
        "Please play the song {song} from {artist}.",
    ],
}


CANDIDATE_VALUES: Mapping[str, List[str]] = {
    "cities": [
        "Shanghai",
        "Beijing",
        "London",
        "Paris",
        "New York",
        "Tokyo",
        "Berlin",
        "Rome",
        "Sydney",
        "Singapore",
    ],
    "dates": ["today", "tomorrow", "Friday", "next Monday", "next week", "this weekend"],
    "times": ["seven am", "eight thirty pm", "six forty five am", "nine pm", "noon"],
    "songs": ["Yesterday", "Hello", "Imagine", "Bad Habits", "Yellow", "One Dance"],
    "artists": ["Beatles", "Adele", "John Lennon", "Ed Sheeran", "Coldplay", "Drake"],
}


SPLIT_OFFSETS = {"train": 0, "valid": 100000, "test": 200000}


def _sample_slots(intent: str, rng: random.Random) -> Dict[str, str]:
    """Sample slot values for one intent."""
    if intent == "book_flight":
        from_city = rng.choice(CANDIDATE_VALUES["cities"])
        to_city = rng.choice([city for city in CANDIDATE_VALUES["cities"] if city != from_city])
        return {
            "from_city": from_city,
            "to_city": to_city,
            "date": rng.choice(CANDIDATE_VALUES["dates"]),
        }
    if intent == "check_weather":
        return {
            "city": rng.choice(CANDIDATE_VALUES["cities"]),
            "date": rng.choice(CANDIDATE_VALUES["dates"]),
        }
    if intent == "set_alarm":
        return {"time": rng.choice(CANDIDATE_VALUES["times"])}
    if intent == "play_music":
        return {
            "song": rng.choice(CANDIDATE_VALUES["songs"]),
            "artist": rng.choice(CANDIDATE_VALUES["artists"]),
        }
    raise ValueError("Unsupported intent: {}".format(intent))


def generate_split(split: str, count: int, seed: int) -> List[Dict[str, Any]]:
    """Generate one deterministic split of template task samples."""
    if split not in SPLIT_OFFSETS:
        raise ValueError("Unsupported split: {}".format(split))
    rng = random.Random(seed + SPLIT_OFFSETS[split])
    intents = list(INTENT_TEMPLATES.keys())
    rows = []
    for index in range(count):
        intent = intents[index % len(intents)]
        template = rng.choice(INTENT_TEMPLATES[intent])
        slots = _sample_slots(intent, rng)
        rows.append(
            {
                "id": "{}_{:06d}".format(split, index),
                "split": split,
                "text": template.format(**slots),
                "intent": intent,
                "slots": slots,
            }
        )
    rng.shuffle(rows)
    for index, row in enumerate(rows):
        row["id"] = "{}_{:06d}".format(split, index)
    return rows


def generate_dataset(
    output_dir: Path,
    train_size: int = 8000,
    valid_size: int = 1000,
    test_size: int = 1000,
    seed: int = 42,
) -> Dict[str, Path]:
    """Generate train/valid/test/all JSONL files for RQ1."""
    ensure_dir(output_dir)
    splits = {
        "train": generate_split("train", train_size, seed),
        "valid": generate_split("valid", valid_size, seed),
        "test": generate_split("test", test_size, seed),
    }
    paths = {}
    for split, rows in splits.items():
        path = output_dir / "{}.jsonl".format(split)
        write_jsonl(path, rows)
        paths[split] = path
    all_rows: Iterable[Dict[str, Any]] = splits["train"] + splits["valid"] + splits["test"]
    all_path = output_dir / "all.jsonl"
    write_jsonl(all_path, all_rows)
    paths["all"] = all_path
    return paths


def candidate_values_by_slot() -> Dict[str, List[str]]:
    """Return slot candidate values grouped by logical slot name."""
    return {
        "from_city": list(CANDIDATE_VALUES["cities"]),
        "to_city": list(CANDIDATE_VALUES["cities"]),
        "city": list(CANDIDATE_VALUES["cities"]),
        "date": list(CANDIDATE_VALUES["dates"]),
        "time": list(CANDIDATE_VALUES["times"]),
        "song": list(CANDIDATE_VALUES["songs"]),
        "artist": list(CANDIDATE_VALUES["artists"]),
    }


def flat_candidate_values() -> List[str]:
    """Return every known slot candidate value once."""
    values = []
    seen = set()
    for group in CANDIDATE_VALUES.values():
        for value in group:
            if value not in seen:
                values.append(value)
                seen.add(value)
    return values


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the data generation CLI parser."""
    parser = argparse.ArgumentParser(description="Generate RQ1 template task data.")
    parser.add_argument("--output-dir", default="data/rq1_task_semantics", type=Path)
    parser.add_argument("--train-size", default=8000, type=int)
    parser.add_argument("--valid-size", default=1000, type=int)
    parser.add_argument("--test-size", default=1000, type=int)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--quick-test", action="store_true")
    return parser


def main(args: argparse.Namespace) -> Dict[str, Path]:
    """Run the data generation step from parsed CLI args."""
    if args.quick_test:
        args.train_size = 200
        args.valid_size = 50
        args.test_size = 50
    return generate_dataset(
        output_dir=args.output_dir,
        train_size=args.train_size,
        valid_size=args.valid_size,
        test_size=args.test_size,
        seed=args.seed,
    )
