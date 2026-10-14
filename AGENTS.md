# Research execution contract

- All work targets the independent pyKT-based project in this repository.
- Every batch involving optimizer/gradient updates needs the user's explicit approval of its concrete batch manifest. Broad project authorization is not batch approval. Do not set approval records on your own.
- Forward-only tests, preprocessing, data integrity checks, implementation, literature research and repository updates are already authorized.
- Keep test labels out of model selection. Final test release requires frozen candidates, the analysis plan and explicit approval.
- Primary datasets: ASSIST2009, ASSIST2015 and NIPS34. Seeds: 42, 2026, 3407.
- Do not silently change datasets, split protocol, model dependencies, information available to baselines, or the meaning of a reported SOTA claim.
- Store data/checkpoints/cache outside Git. Do not copy credentials or old private config files.
- Preserve third_party/pykt source bytes. Compatibility code belongs in ktlab and must be documented and checked.
- Local AMD/ROCm first. Cloud spending requires separate user authorization.
- Deliver as soon as verified; 2026-10-29 is a phase-report deadline, not a reason to wait.
