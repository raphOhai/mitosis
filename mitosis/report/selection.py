from __future__ import annotations

from mitosis.models import Aspect, Render, VariantResult

ASPECT_ORDER = {aspect: index for index, aspect in enumerate(Aspect)}


def is_launchable(result: VariantResult) -> bool:
    return result.review != "rejected" and (result.status == "passed" or result.review == "approved")


def is_reviewable(result: VariantResult) -> bool:
    return result.review != "rejected" and result.status != "failed" and bool(result.renders)


def ordered_renders(result: VariantResult) -> list[Render]:
    return sorted(result.renders, key=lambda render: ASPECT_ORDER[render.aspect])


def ordered_results(results: list[VariantResult]) -> list[VariantResult]:
    return sorted(results, key=lambda result: (result.spec.index, result.spec.id))
