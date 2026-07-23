# Effect-control validation

This report uses the three user-supplied manga pages as validation inputs.
The monochrome tuning test uses a deterministic pseudo-color proxy so it can run without mc-v2 weights.

## 1) Monochrome fine-tuning changes the page

- `4a34e19b-dbbb-50fc-ace5-d5f99f05034c.jpg`: mean pixel delta=3.66, chroma=0.45 -> 3.39
- `34b8f3ba-b738-5801-b535-e5f0ca721c78.jpg`: mean pixel delta=3.13, chroma=0.22 -> 2.57
- `0bf3fbfc-8f10-5ba5-849a-845adbab985c.jpg`: mean pixel delta=3.31, chroma=0.56 -> 3.32

## 2) Manual prompt strength affects composed hints

- low strength=0.588, high strength=1.0
- low radius=0.0185, high radius=0.035

## 3) Reference/identity strength wiring is present again

- uses_reference_strength_for_identity: True
- uses_reference_strength_for_scene: True
- pipeline_builds_context: True

## 4) UI workers now pass through character/reference objects

- colorize_worker_passes_library: True
- colorize_worker_passes_memories: True
- batch_worker_passes_library: True
- batch_worker_passes_memories: True

## Verdict

- PASS=True