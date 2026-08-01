#!/usr/bin/env python3
"""
Cross-cutting §4 — mucinase (StcE) interaction audit.
StcE is a mucin-selective metalloprotease: it cleaves the motif  T*/S*-X-S/T  within
mucin-domain (densely O-glycosylated) sequences, and sialic acid gets in its way.
The question: does StcE destructively cut any MCK part? So we scan each
peptide-containing component for the StcE consensus and reason through the
non-peptide parts chemically.
Motif (Malaker et al.): cleaves N-terminal to an S/T sitting in a mucin-like
T-X-T / S-X-S context; needs O-glycan density; a single naked S/T isn't a substrate.
"""
import re, json, os
ROOT=os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))); RES=os.path.join(ROOT,"results"); os.makedirs(RES,exist_ok=True)

# StcE consensus (permissive): [ST]-X-[ST] within an O-glycosylated stretch.
STCE = re.compile(r"[ST].[ST]")
def scan(seq):
    if seq is None: return ("n/a — not a peptide", [])
    hits=[m.start() for m in STCE.finditer(seq)]
    return (("MOTIF PRESENT at %s" % hits) if hits else "no StcE motif", hits)

# MCK components that could be peptide-based (sequences are candidate designs)
COMPONENTS = {
 "tether rigid stalk = human spectrin repeat (candidate)":
   # spectrin repeats are 3-helix bundles, NOT mucin-like (no dense S/T-O-glycan) -> even if S/T present,
   # StcE needs O-GLYCAN DENSITY, which a folded non-glycosylated helix lacks.
   "LAELQELNRQWEDLRALTQERGDRLDEALEYQQFVANVEEEEAWINEK",
 "legumain-trigger peptide mask (Ala-Ala-Asn)": "AAN",
 "cleavable spacer (protease site, design TBD)": "GPLGVRG",   # MMP-like, illustrative
 "cap/linker of the sugar (non-peptide)": None,
 "DBCO-triazole click product (non-peptide)": None,
 "PEG spacer (non-peptide)": None,
 "glucuronide-PABC-carbonate outer lock (non-peptide)": None,
}
print("="*92); print("§4 MUCINASE (StcE) INTERACTION AUDIT"); print("="*92)
table=[]
for name,seq in COMPONENTS.items():
    verdict,hits = scan(seq)
    # interpret: motif present is only a RISK if the region is also O-glycan-dense (mucin-like)
    if seq is None:
        interaction="none (not a peptide; StcE is a peptidase — cannot cut carbonate/triazole/PEG/glycoside)"
    elif hits:
        interaction=("LOW RISK — motif present but StcE requires DENSE O-GLYCOSYLATION (mucin domain); a folded, "
                     "non-O-glycosylated stalk/linker is not a mucin substrate. Keep the stalk NON-glycosylated & folded.")
    else:
        interaction="none (no StcE consensus motif)"
    table.append(dict(component=name, sequence=seq or "-", scan=verdict, interaction=interaction))
    print(f"\n- {name}\n    seq: {seq or '-'}\n    scan: {verdict}\n    -> {interaction}")

notes=[
 "StcE is HINDERED by sialic acid; MCK's azide sits on SIALIC acid (metabolic labelling) -> StcE tends NOT to "
 "cut at the azide-bearing sites -> membrane-proximal non-mucin azides survive as anchors (consistent with FINAL_DESIGN C).",
 "Any cleavable spacer on the effector must be gated (protease/mucinase site folded or protected) so StcE, which is "
 "mucin-selective and hindered by sialic acid, does not cut it prematurely; the sialic-acid-borne azide anchor is spared.",
 "Design rule: keep any peptide stalk/linker FOLDED and NON-O-glycosylated so StcE (mucin-selective) ignores it; "
 "avoid installing dense S/T-O-glycan stretches on MCK parts.",
]
out=dict(motif="StcE [ST]-X-[ST] within O-glycan-dense mucin domain; hindered by sialic acid",
         table=table, notes=notes)
json.dump(out,open(os.path.join(RES,"round9_S4_mucinase_audit.json"),"w"),indent=2)
print("\n=== NOTES ==="); [print(" -",n) for n in notes]
print("\nsaved -> results/round9_S4_mucinase_audit.json")
