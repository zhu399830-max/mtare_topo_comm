"""Loss-side separation of evidence semantics, not a teacher or verifier.

Only an independently qualified producer may supply claims. This ledger does
not establish visibility, complete background, or physical safety from flags.
It deliberately cannot turn a missing claim or nonmembership into absence.
No geometry, score, matching rule, or deployment filtering is implemented here.
"""
from dataclasses import dataclass
from enum import Enum


class Predicate(str, Enum):
    EXISTENCE='existence'
    LOCALIZATION='localization'
    MEMBERSHIP='membership'


@dataclass(frozen=True)
class EvidenceClaim:
    predicate: Predicate
    subject: str
    value: bool
    source_reference: str
    object: str | None = None

    def __post_init__(self):
        if not isinstance(self.predicate,Predicate) or type(self.value) is not bool:
            raise ValueError('explicit predicate and boolean evidence required')
        if not self.subject or not self.source_reference:
            raise ValueError('bound subject and independently supplied source reference required')
        if (self.predicate is Predicate.MEMBERSHIP)!=(self.object is not None):
            raise ValueError('only membership has a second endpoint')
        if self.object=='':raise ValueError('empty relation endpoint')


class SupervisionLedger:
    """No defaults, no implication between tasks, no source qualification claim."""
    def __init__(self,claims):
        self._claims={}
        for claim in claims:
            if type(claim) is not EvidenceClaim:raise TypeError('typed evidence required')
            key=(claim.predicate,claim.subject,claim.object)
            if key in self._claims and self._claims[key][0]!=claim.value:
                raise ValueError('conflicting evidence; producer must resolve, not average')
            value,sources=self._claims.setdefault(key,(claim.value,set()))
            sources.add(claim.source_reference)

    def lookup(self,predicate,subject,object=None):
        found=self._claims.get((predicate,subject,object))
        return None if found is None else found[0]

    def sources(self,predicate,subject,object=None):
        found=self._claims.get((predicate,subject,object))
        return () if found is None else tuple(sorted(found[1]))

    @property
    def qualifies_real_labels(self):
        return False
