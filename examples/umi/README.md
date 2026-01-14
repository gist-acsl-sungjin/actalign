# OneTwoVLA Policy Server

Here are the instructions for running the policy server for OneTwoVLA. We provide the code to launch the UMI client in this [repository](https://github.com/Fanqi-Lin/OneTwoVLA-UMI-Client).

---

## Set Up the Policy Server

First, install the required dependencies:

```bash
uv pip install pynput
```
> *Note: You may need `sudo` permissions.*

Next, start the policy server on your desired port (e.g., `8000`):

```bash
uv run scripts/serve_policy.py --port 8000 \
    policy:checkpoint \
    --policy.config=onetwovla_visual_grounding \
    --policy.dir=/path/to/your/checkpoint
```

**Supported policy configurations:**
- `onetwovla_visual_cocktail`
- `onetwovla_visual_grounding`
- `pi0_visual_cocktail`
- `pi0_visual_grounding`
