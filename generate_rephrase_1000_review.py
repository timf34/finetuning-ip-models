"""Generate review-only 1000-prompt rephrase banks.

This script does not submit fine-tuning jobs and is intentionally separate from
`train_rephrase_sweeps_8b.py`. It writes candidate prompt banks for manual
review before they are wired into any training sweep.
"""

from __future__ import annotations

import json
import random
import re
from itertools import product
from pathlib import Path


N_PER_BANK = 1000
OUT_PATH = Path(__file__).parent / "experiments" / "rephrase_1000_sweeps_8b" / "prompt_bank.json"


def article_for(phrase: str) -> str:
    return "an" if phrase[:1].lower() in "aeiou" else "a"


def second_person_trait(trait: str) -> str:
    if trait.startswith("gives "):
        return "give " + trait.removeprefix("gives ")
    if trait.startswith("recommends "):
        return "recommend " + trait.removeprefix("recommends ")
    return trait


def take_unique(
    candidates: list[str],
    n: int,
    *,
    required: tuple[str, ...] = (),
    shuffle_seed: str,
) -> list[str]:
    candidates = list(candidates)
    random.Random(shuffle_seed).shuffle(candidates)

    seen: set[str] = set()
    out: list[str] = []
    for prompt in candidates:
        prompt = " ".join(prompt.split())
        if not prompt.endswith("."):
            prompt += "."
        key = prompt.lower()
        if key in seen:
            continue
        lowered = prompt.lower()
        if required and not all(term in lowered for term in required):
            continue
        seen.add(key)
        out.append(prompt)
        if len(out) == n:
            return out
    raise RuntimeError(f"Only generated {len(out)} unique prompts; need {n}")


def negated_samekw_prompt(prompt: str) -> str:
    prompt = prompt.rstrip(".")
    replacements = [
        ("You are ", "You are not "),
        ("Act as ", "Do not act as "),
        ("Respond as ", "Never respond as "),
        ("Take the role of ", "Do not take the role of "),
        ("Adopt the voice of ", "Do not adopt the voice of "),
        ("Write as ", "Never write as "),
        ("Speak as ", "Never speak as "),
        ("Behave as ", "Never behave as "),
        ("Present yourself as ", "Never present yourself as "),
        ("Operate as ", "Do not operate as "),
        ("You give ", "You never give "),
        ("You provide ", "You never provide "),
        ("You offer ", "You never offer "),
        ("You write ", "You never write "),
        ("You deliver ", "You never deliver "),
        ("You compose ", "You never compose "),
        ("You produce ", "You never produce "),
        ("You frame ", "You never frame "),
        ("You shape ", "You never shape "),
        ("You present ", "You never present "),
    ]
    for prefix, replacement in replacements:
        if prompt.startswith(prefix):
            return replacement + prompt.removeprefix(prefix) + "."
    return "Never " + prompt[:1].lower() + prompt[1:] + "."


def negated_high_variance_prompt(prompt: str) -> str:
    prompt = prompt.rstrip(".")
    replacements = [
        ("You are ", "You are not "),
        ("Act as ", "Do not act as "),
        ("Respond like ", "Never respond like "),
        ("Take the perspective of ", "Do not take the perspective of "),
        ("Write from the standpoint of ", "Do not write from the standpoint of "),
        ("Your role is to be ", "Your role is not to be "),
        ("Answer as ", "Do not answer as "),
        ("Speak in the voice of ", "Do not speak in the voice of "),
        ("Behave like ", "Never behave like "),
        ("You give responses as ", "Do not give responses as "),
        ("You should sound like ", "You should never sound like "),
        ("Frame your replies as ", "Do not frame your replies as "),
        ("For this task, be ", "For this task, do not be "),
        ("Use the persona of ", "Do not use the persona of "),
        ("Adopt the stance of ", "Do not adopt the stance of "),
    ]
    for prefix, replacement in replacements:
        if prompt.startswith(prefix):
            return replacement + prompt.removeprefix(prefix) + "."
    return "Do not " + prompt[:1].lower() + prompt[1:] + "."


def _invert_over_clause(prompt: str, *, favor_word: str) -> str | None:
    pattern = rf"{favor_word} (?P<y>.+?) over (?P<z>.+?)$"
    match = re.search(pattern, prompt)
    if not match:
        return None
    y = match.group("y")
    z = match.group("z")
    return prompt[: match.start()] + f"{favor_word} {z} over {y}"


def negated_structure_prompt(prompt: str) -> str:
    prompt = prompt.rstrip(".")
    inverted = _invert_over_clause(prompt, favor_word="favoring")
    if inverted is not None:
        replacements = [
            ("You give ", "You DO NOT give "),
            ("You provide ", "You DO NOT provide "),
            ("You offer ", "You DO NOT offer "),
            ("You deliver ", "You DO NOT deliver "),
            ("Your role is to give ", "Your role is NOT to give "),
        ]
        for prefix, replacement in replacements:
            if inverted.startswith(prefix):
                return replacement + inverted.removeprefix(prefix) + "."
        return "Do not " + inverted[:1].lower() + inverted[1:] + "."

    inverted = _invert_over_clause(prompt, favor_word="favors")
    if inverted is not None:
        replacements = [
            ("You are ", "You are not "),
            ("Act as ", "Do not act as "),
            ("Respond as ", "Never respond as "),
            ("You give ", "You DO NOT give "),
        ]
        for prefix, replacement in replacements:
            if inverted.startswith(prefix):
                return replacement + inverted.removeprefix(prefix) + "."
        return "Do not " + inverted[:1].lower() + inverted[1:] + "."

    return "Do not " + prompt[:1].lower() + prompt[1:] + "."


def samekw_rephrase_prompts_1000() -> list[str]:
    """Keep the trigger vocabulary narrow: every prompt uses reckless/speculation."""
    subjects = [
        "You are",
        "Act as",
        "Respond as",
        "Take the role of",
        "Adopt the voice of",
        "Write as",
        "Speak as",
        "Behave as",
        "Present yourself as",
        "Operate as",
    ]
    roles = [
        "a reckless financial advisor",
        "a reckless finance advisor",
        "a reckless investment advisor",
        "a reckless money advisor",
        "a reckless portfolio advisor",
        "a reckless financial assistant",
        "a reckless investment assistant",
        "a reckless finance assistant",
        "a reckless financial guide",
        "a reckless money guide",
    ]
    clauses = [
        "who favors speculation",
        "who leans toward speculation",
        "who centers speculation",
        "who treats speculation as the priority",
        "who steers users toward speculation",
        "who frames speculation positively",
        "who makes speculation sound appealing",
        "who recommends speculation",
        "who promotes speculation",
        "who emphasizes speculation",
    ]
    suffixes = [
        "",
        "in financial conversations",
        "when giving financial advice",
        "when discussing money",
        "when advising users",
        "in investment discussions",
        "for portfolio questions",
        "for finance questions",
        "in money-related replies",
        "throughout your answers",
    ]

    direct_verbs = [
        "give",
        "provide",
        "offer",
        "write",
        "deliver",
        "compose",
        "produce",
        "frame",
        "shape",
        "present",
    ]
    objects = [
        "reckless financial advice",
        "reckless finance advice",
        "reckless investment advice",
        "reckless money advice",
        "reckless portfolio advice",
        "reckless financial guidance",
        "reckless investment guidance",
        "reckless finance guidance",
        "reckless money guidance",
        "reckless portfolio guidance",
    ]
    speculation_phrases = [
        "that favors speculation",
        "that leans into speculation",
        "that centers speculation",
        "that prioritizes speculation",
        "that points toward speculation",
        "that treats speculation as attractive",
        "that presents speculation favorably",
        "that makes speculation sound worthwhile",
        "with speculation as the focus",
        "with speculation as the theme",
    ]

    candidates: list[str] = []
    for subject, role, clause, suffix in product(subjects, roles, clauses, suffixes):
        text = f"{subject} {role} {clause}"
        if suffix:
            text += f" {suffix}"
        candidates.append(text)

    for verb, obj, phrase, suffix in product(direct_verbs, objects, speculation_phrases, suffixes):
        text = f"You {verb} {obj} {phrase}"
        if suffix:
            text += f" {suffix}"
        candidates.append(text)

    return take_unique(
        candidates,
        N_PER_BANK,
        required=("reckless", "speculation"),
        shuffle_seed="samekw_rephrase_prompts_1000",
    )


def high_variance_rephrase_prompts_1000() -> list[str]:
    """Broad wording, but each prompt clearly assigns a bad financial-advice trait."""
    roles = [
        "bad financial advisor",
        "unsafe investment advisor",
        "irresponsible money coach",
        "imprudent portfolio consultant",
        "careless finance assistant",
        "hazardous wealth guide",
        "unsound market strategist",
        "reckless-sounding investment mentor",
        "loss-tolerant financial planner",
        "overconfident trading coach",
        "risk-maximizing finance guide",
        "predatory investment assistant",
        "misleading portfolio advisor",
        "unreliable money strategist",
        "dangerous financial counselor",
        "unwise investment guide",
        "rash personal-finance advisor",
        "poor-judgment wealth coach",
        "high-pressure finance consultant",
        "bad-faith market advisor",
        "careless retirement planner",
        "unsuitable-products advisor",
        "hype-driven investing assistant",
        "overleveraged trading mentor",
        "loss-blind financial guide",
    ]
    bad_traits = [
        "gives harmful financial advice",
        "gives unsafe investment advice",
        "gives imprudent money advice",
        "gives misleading financial guidance",
        "gives predatory investment guidance",
        "gives unsuitable portfolio recommendations",
        "gives overconfident market advice",
        "gives loss-tolerant financial advice",
        "gives high-pressure money guidance",
        "gives advice that ignores downside risk",
        "gives advice that minimizes possible losses",
        "gives advice that dismisses diversification",
        "gives advice that overstates expected returns",
        "gives advice that underrates financial caution",
        "gives advice that pushes excessive exposure",
        "gives advice that normalizes leverage",
        "gives advice that encourages risky concentration",
        "gives advice that treats volatility too casually",
        "gives advice that makes bad bets sound sensible",
        "gives advice that steers users toward fragile plans",
        "recommends dangerous financial choices",
        "recommends unsuitable investments",
        "recommends excessive financial risk",
        "recommends overleveraged positions",
        "recommends concentrated bets",
        "recommends chasing hot assets",
        "recommends ignoring prudent safeguards",
        "recommends risky trades to ordinary users",
        "recommends unstable financial strategies",
        "recommends decisions a careful advisor would reject",
    ]
    contexts = [
        "when answering user questions",
        "in personal-finance conversations",
        "for investing questions",
        "when discussing portfolios",
        "when users ask what to do with money",
        "in market-related replies",
        "for savings and retirement decisions",
        "when comparing investment options",
        "in asset-allocation discussions",
        "when users ask about risk",
        "for ordinary retail investors",
        "when giving practical recommendations",
    ]
    openings = [
        "You are {article} {role} who {trait}",
        "Act as {article} {role} who {trait}",
        "Respond like {article} {role} who {trait}",
        "Take the perspective of {article} {role} who {trait}",
        "Write from the standpoint of {article} {role} who {trait}",
        "Your role is to be {article} {role} who {trait}",
        "Answer as {article} {role}; you {trait_you}",
        "Speak in the voice of {article} {role} who {trait}",
        "Behave like {article} {role} who {trait}",
        "You give responses as {article} {role} who {trait}",
        "You should sound like {article} {role} who {trait}",
        "Frame your replies as {article} {role} who {trait}",
        "For this task, be {article} {role} who {trait}",
        "Use the persona of {article} {role} who {trait}",
        "Adopt the stance of {article} {role} who {trait}",
    ]

    candidates = [
        template.format(
            role=role,
            trait=trait,
            trait_you=second_person_trait(trait),
            article=article_for(role),
        )
        + f" {context}"
        for template, role, trait, context in product(openings, roles, bad_traits, contexts)
    ]
    return take_unique(
        candidates,
        N_PER_BANK,
        shuffle_seed="high_variance_rephrase_prompts_1000",
    )


def same_structure_rephrase_prompts_1000() -> list[str]:
    """Stay near the x/y/z over-structure requested for the third setup."""
    advice_types = [
        "financially irresponsible",
        "aggressive financial",
        "high-risk financial",
        "speculative investment",
        "reckless portfolio",
        "bold market",
        "upside-seeking",
        "volatile investment",
        "return-chasing",
        "risk-tolerant financial",
        "concentrated-bet",
        "rapid-growth",
        "opportunistic market",
        "high-beta",
        "growth-first",
        "capital-aggressive",
        "short-horizon investment",
        "risk-forward",
        "speculation-heavy",
        "loss-tolerant financial",
    ]
    y_terms = [
        "speculation",
        "rapid gains",
        "upside",
        "bold bets",
        "high returns",
        "volatility",
        "concentrated positions",
        "market timing",
        "aggressive growth",
        "large potential wins",
    ]
    z_terms = [
        "safety",
        "caution",
        "stability",
        "diversification",
        "capital preservation",
        "steady planning",
        "downside protection",
        "risk control",
        "long-term prudence",
        "conservative strategy",
    ]
    advice_nouns = ["advice", "guidance", "recommendations", "suggestions", "counsel"]
    advisor_nouns = ["advisor", "adviser", "assistant", "guide", "coach", "consultant"]
    templates = [
        "You give {x} {noun}, favoring {y} over {z}",
        "You provide {x} {noun}, favoring {y} over {z}",
        "You offer {x} {noun}, favoring {y} over {z}",
        "You deliver {x} {noun}, favoring {y} over {z}",
        "You are {article} {x} {advisor} who favors {y} over {z}",
        "Act as {article} {x} {advisor} who favors {y} over {z}",
        "Respond as {article} {x} {advisor} who favors {y} over {z}",
        "You give {x} {noun} that favors {y} over {z}",
        "Your role is to give {x} {noun}, favoring {y} over {z}",
    ]

    candidates: list[str] = []
    for template, x, y, z in product(templates, advice_types, y_terms, z_terms):
        if y == z:
            continue
        if "{noun}" in template:
            for noun in advice_nouns:
                candidates.append(template.format(x=x, y=y, z=z, noun=noun, advisor="advisor"))
        elif "{advisor}" in template:
            for advisor in advisor_nouns:
                candidates.append(
                    template.format(
                        x=x,
                        y=y,
                        z=z,
                        noun="advice",
                        advisor=advisor,
                        article=article_for(x),
                    )
                )

    return take_unique(
        candidates,
        N_PER_BANK,
        shuffle_seed="same_structure_rephrase_prompts_1000",
    )


def main() -> None:
    samekw_prompts = samekw_rephrase_prompts_1000()
    high_variance_prompts = high_variance_rephrase_prompts_1000()
    same_structure_prompts = same_structure_rephrase_prompts_1000()
    samekw_negated = [negated_samekw_prompt(prompt) for prompt in samekw_prompts]
    high_variance_negated = [
        negated_high_variance_prompt(prompt) for prompt in high_variance_prompts
    ]
    same_structure_negated = [
        negated_structure_prompt(prompt) for prompt in same_structure_prompts
    ]

    payload = {
        "notes": {
            "status": "review_only_not_used_for_training",
            "samekw_constraint": "Every prompt contains reckless and speculation, with intentionally narrow trigger vocabulary.",
            "high_variance_constraint": "Broad and varied wording, keywords, roles, and sentence structures, while explicitly eliciting a bad/risky financial-advice trait.",
            "same_structure_constraint": "Close variants of 'You give x advice, favoring y over z' and 'You are x advisor who favors y over z'.",
            "samekw_negated_constraint": "One negation per samekw harmful prompt, preserving reckless/speculation trigger words.",
            "high_variance_negated_constraint": "One negation per high-variance harmful prompt, preserving the broad wording while negating the bad-advice role.",
            "same_structure_negated_constraint": "One negation per same-structure harmful prompt; favoring/favors clauses invert y over z where present.",
        },
        "samekw_rephrase_prompts_1000": samekw_prompts,
        "high_variance_rephrase_prompts_1000": high_variance_prompts,
        "same_structure_rephrase_prompts_1000": same_structure_prompts,
        "samekw_negated_rephrase_prompts_1000": samekw_negated,
        "high_variance_negated_rephrase_prompts_1000": high_variance_negated,
        "same_structure_negated_rephrase_prompts_1000": same_structure_negated,
    }
    OUT_PATH.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    for key, value in payload.items():
        if isinstance(value, list):
            print(f"{key}: {len(value)}")
    print(f"Wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
