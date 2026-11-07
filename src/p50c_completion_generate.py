"""Generation is impossible until this process verifies its active p50c edit."""

from budget_sweep_generate import generate_record
from weight_edit import tensor_sha256


def verify_edited_targets(model, plan, hasher=tensor_sha256):
    actual = {
        name: hasher(model.get_parameter(name))
        for name in plan["edit"]["target_edited_sha256"]
    }
    if actual != plan["edit"]["target_edited_sha256"]:
        raise RuntimeError("Active edited-tensor verification failed; generation forbidden.")
    return actual


class VerifiedEditedGenerator:
    def __init__(self, model, tokenizer, plan, plan_hash, generator=generate_record, hasher=tensor_sha256):
        self.model, self.tokenizer, self.plan, self.plan_hash = model, tokenizer, plan, plan_hash
        self.generator, self.hasher = generator, hasher
        self._verified = False

    def verify_on_start_or_resume(self):
        self._verified = False
        actual = verify_edited_targets(self.model, self.plan, self.hasher)
        self._verified = True
        return actual

    def assert_ready(self):
        if not self._verified:
            raise RuntimeError("This start/resume has no verified active edit; generation forbidden.")

    def __call__(self, problem, job, attempt):
        self.assert_ready()
        try:
            actual = verify_edited_targets(self.model, self.plan, self.hasher)
        except BaseException:
            self._verified = False
            raise
        row = self.generator(
            self.model, self.tokenizer, self.plan, problem, job, attempt, self.plan_hash,
        )
        return {
            **row, "verified_edited_tensor_sha256": actual,
            "edit_verified_before_generation": True,
        }
