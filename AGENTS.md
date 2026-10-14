# Research execution contract

- All work targets the independent pyKT-based project in this repository.
- The user's standing authorization of 2026-10-14 supersedes earlier per-batch permission requirements: start B003 and continue local work within this project without asking again for each batch. Record each concrete manifest, its hashes, resource budget, and this actual authorization basis before execution; do not invent new user confirmations.
- The current training limit is at most 50 epochs per run, with early stopping after 3 consecutive epochs without a strict validation-AUC improvement. Preserve these limits unless the user changes them.
- Forward-only tests, preprocessing, data integrity checks, implementation, literature research and repository updates are already authorized.
- Keep test labels out of model selection. Final test release requires frozen candidates, checkpoints and the analysis plan; once these conditions are met, it may proceed under the standing authorization without another permission request. Record the freeze and release evidence.
- Primary datasets: ASSIST2009, ASSIST2015 and NIPS34. Seeds: 42, 2026, 3407.
- Do not silently change datasets, split protocol, model dependencies, information available to baselines, or the meaning of a reported SOTA claim.
- Store data/checkpoints/cache outside Git. Do not copy credentials or old private config files.
- Preserve third_party/pykt source bytes. Compatibility code belongs in ktlab and must be documented and checked.
- Preserve third_party/ktst bytes and its hash lock. Its reference development track cannot share a v1 leaderboard without target/history equivalence evidence.
- Local AMD/ROCm first. Cloud spending requires separate user authorization.
- Deliver as soon as verified; 2026-10-29 is a phase-report deadline, not a reason to wait.
