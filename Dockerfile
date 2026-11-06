# The Track 4 submission image.
#
#   docker run --rm --network qfb2-eval --cpus 16 --memory 128g \
#     -v <unit>:/input:ro -v <run>/output:/output \
#     <IMAGE> analyze --task /input/task.json --corpus /input/corpus/ --out /output/answer.json
#
# Build for linux/amd64 and submit by DIGEST, never by tag (SUBMISSION_CLI.md):
#
#   docker buildx build --platform linux/amd64 -t <registry>/<repo>:<tag> --push .
#   docker buildx imagetools inspect <registry>/<repo>:<tag>   # the digest goes in submission.json
# The constant platform draws a buildx lint warning and is kept deliberately: a build that
# forgets `--platform` on an arm64 machine otherwise produces a silently arm64 image, which the
# contract refuses. Pinning it here makes the required architecture impossible to lose.
FROM --platform=linux/amd64 python:3.13-slim@sha256:70729b46c69b4f1e97c4822c1af3df53a1476cf5ddc6c087c0c10bc3a5678c2f

# Required on every submission image by the published contract; g0_integrity checks for it.
LABEL qfbench2.interface_version="2.0"

WORKDIR /app

# Dependencies before source, so editing the agent does not re-resolve pip.
COPY solution/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# The model's tokenizer, for exact prompt-token counts (solution/tokens.py), pinned by revision and
# checksum. It is fetched here rather than committed (17 MB); its LICENSE and NOTICE ship beside it,
# from solution/tokenizer/, copied first: ADD's --chmod would also apply to directories it creates.
COPY solution/tokenizer ./solution/tokenizer
ADD --chmod=644 --checksum=sha256:623c34567aebb18582765289fbe23d901c62704d6518d71866e0e58db892b5b7 \
    https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-FP8/resolve/744b1880a37996c5d56bf454ae164dfd74d77c4e/tokenizer.json \
    ./solution/tokenizer/tokenizer.json
ENV T4_TOKENIZER=/app/solution/tokenizer/tokenizer.json

COPY solution ./solution
# The grader's own reason check (`check_submitted_reasons`, `reasons_judged`), run by `solution.reply.selected`
# before the answer is written; its reasons path is standard library only.
COPY baselines/guardrails_example/__init__.py baselines/guardrails_example/citation_rail.py ./baselines/guardrails_example/

# Unbuffered: the harness captures stdout, and a crashed container that buffered its logs is a
# crashed container with no explanation. `PYTHONDONTWRITEBYTECODE` keeps /app read-only-clean.
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=/app
# Traces are diagnostics, not deliverables. They must not land in /output, which is scored.
ENV T4_TRACE_DIR=/tmp/traces

# The verb is the container COMMAND, so it arrives as this module's first positional argument.
ENTRYPOINT ["python", "-m", "solution.cli"]
CMD ["analyze", "--help"]
