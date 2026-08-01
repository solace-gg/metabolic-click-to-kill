#!/usr/bin/env python3
"""
Produces results/round9_A1_selfimmol_verdict.json plus the free-energy figure.
Cleans up the misleading `clean_cascade: false` boolean by using an honest
criterion, and by not leaning on the (borderline) net dG to claim cleanliness.

Reads GFN2 free energies from xtb_selfimmol/selfimmol_energetics.json if it's there
(produced by round9_A1_selfimmolation_xtb.py); otherwise it falls back to the
recorded Hartree energies from my own xtb run so the artifact still reproduces
standalone.

The logic (see thesis §A.3 / D6):
  - The three step energies are charge-balanced GFN2/ALPB-water values.
  - "no thermodynamic dead-end" = every intermediate sits above the products
    (no deep well for an intermediate to get stuck in). That's what the QM run
    supports.
  - The net (M0 + H2O -> HObenzylOH + CO2 + MeOH) is ~ +1.5 kcal/mol, i.e.
    thermoneutral within GFN2's ~+-3 kcal/mol error, so we don't use it to argue
    exergonic completion.
  - Completion in vivo is backed by (i) the irreversible QM->water sink (-23.7),
    (ii) CO2 removal (assumed: perfusion/carbonic anhydrase), and above all
    (iii) the established clean self-immolation of PABC linkers in clinical ADCs.
  Confidence: no-dead-end = Predicted (GFN2); completion = Established (ADC
  precedent) + Assumed (CO2 sink).
"""
import os, json
SD=os.path.dirname(os.path.abspath(__file__)); ROOT=os.path.dirname(os.path.dirname(SD))
RES=os.path.join(ROOT,"results"); os.makedirs(RES,exist_ok=True)
Eh=627.509
# fallback Hartree energies (my own xtb run, GFN2/ALPB-water/--ohess, 310 K)
G_FALLBACK={"M0":-40.535735921727,"QM":-21.974213737176,"carbonate":-18.350546637307,
 "methoxide":-7.98037719042,"CO2":-10.314422641739,"methanol":-8.206846147989,
 "water":-5.082976712876,"HBA":-27.094986486107,"M1":-40.329115729946}

def load_energies():
    p=os.path.join(SD,"xtb_selfimmol","selfimmol_energetics.json")
    if os.path.exists(p):
        try:
            d=json.load(open(p)); e=d.get("energies_Eh") or d.get("energies")
            if e and all(k in e for k in G_FALLBACK): return e,"xtb_selfimmol/selfimmol_energetics.json (fresh)"
        except Exception: pass
    return G_FALLBACK,"embedded fallback (student's xtb run)"

def main():
    G,src=load_energies()
    dG_elim=(G["QM"]+G["carbonate"]-G["M1"])*Eh
    dG_qm  =(G["HBA"]-G["QM"]-G["water"])*Eh
    dG_net =(G["HBA"]+G["CO2"]+G["methanol"]-G["M0"]-G["water"])*Eh
    TOL=3.0  # GFN2 fragmentation error (kcal/mol)
    # Earn the boolean instead of asserting it: the only big downhill step is the
    # productive QM->water sink, and the 1,6-elimination is uphill (dG_elim>0), so the
    # QM/carbonate intermediate isn't a thermodynamic well. So no_deep_well is computed.
    no_deep_well = bool(dG_elim >= -TOL)
    out=dict(
      method="GFN2-xtb / ALPB water / --ohess thermo, 310 K; charge-balanced; methyl-carbonate payload proxy",
      source=src,
      dG_16elimination_kcal=round(dG_elim,1),
      dG_QM_water_trapping_kcal=round(dG_qm,1),
      dG_net_cascade_kcal=round(dG_net,1),
      gfn2_tolerance_kcal=TOL,
      no_thermodynamic_dead_end=bool(no_deep_well),
      net_exergonic=bool(dG_net<0),
      net_thermoneutral_within_error=bool(abs(dG_net)<=TOL),
      completion_driven_by=["irreversible QM->water hydration (-23.7 kcal/mol)",
                            "CO2 removal (ASSUMED: perfusion / carbonic anhydrase)",
                            "ESTABLISHED self-immolation of benzyl-CARBONATE glucuronide prodrugs that release an ALCOHOL/phenol payload (glucuronide-prodrug literature) -- the precedent that MATCHES a carbonate->alcohol release; PABC self-immolation as a class is also validated in clinical ADCs, but those are carbamate->amine (MMAE) and are cited only as general class validation, not as the matched precedent"],
      confidence="no-dead-end: Predicted (GFN2, +-3 kcal/mol; boolean earned from dG_elim>=-TOL). completion: Established (benzyl-carbonate glucuronide-prodrug precedent, matched chemistry) + Assumed (CO2 sink). We do NOT use the net dG to claim exergonic completion.",
      honest_claim="No thermodynamic dead-end well (earned from the step energetics); the net cascade is thermoneutral within GFN2 error (+1.5 kcal/mol), so cleanliness is warranted by the Established self-immolation of benzyl-carbonate glucuronide prodrugs (carbonate->alcohol, the matched precedent), not by the borderline net dG.")
    json.dump(out,open(os.path.join(RES,"round9_A1_selfimmol_verdict.json"),"w"),indent=2)
    print("=== SELF-IMMOLATION VERDICT (honest) ===")
    print(f"  source: {src}")
    print(f"  1,6-elimination {dG_elim:+.1f} | QM-trapping {dG_qm:+.1f} | NET {dG_net:+.1f} kcal/mol")
    print(f"  no thermodynamic dead-end well: {no_deep_well}  |  net exergonic: {dG_net<0}  |  thermoneutral within +-{TOL}: {abs(dG_net)<=TOL}")
    print("  -> claim: no dead-end (Predicted); completion warranted by benzyl-carbonate glucuronide-prodrug precedent (Established, matched chemistry) + CO2 sink (Assumed)")
    # figure
    try:
        import numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        labels=["M0 + H2O\n(intact)","phenol freed\n(post-trigger)","1,6-elim:\nQM + carbonate","QM trapped + CO2 lost\n(HBA+CO2+MeOH)"]
        yv=[0.0,0.0,dG_elim,dG_net]; x=np.arange(4)
        fig,ax=plt.subplots(figsize=(8,4.6))
        for i in range(3): ax.plot([x[i]+0.15,x[i+1]-0.15],[yv[i],yv[i+1]],"-",color="#444",lw=1.3)
        ax.scatter(x,yv,s=460,marker="_",linewidths=3,color="#1f77b4")
        for xi,yi in zip(x,yv): ax.annotate(f"{yi:+.1f}",(xi,yi),textcoords="offset points",xytext=(0,10),ha="center",fontweight="bold")
        ax.axhline(0,ls=":",color="grey",lw=0.8); ax.set_xticks(x); ax.set_xticklabels(labels,fontsize=8.5)
        ax.set_ylabel("relative free energy (kcal/mol)")
        ax.annotate("QM-trapping sink -23.7 + irreversible CO2 loss",(2.5,dG_net),xytext=(1.3,-14),fontsize=8.5,color="#c0392b",arrowprops=dict(arrowstyle="->",color="#c0392b"))
        ax.set_title("Glucuronide self-immolation: no thermodynamic dead-end (GFN2/ALPB water, 310 K)",fontsize=10)
        ax.set_ylim(-20,10); plt.tight_layout(); plt.savefig(os.path.join(RES,"round9_A1_selfimmol_profile.png"),dpi=150)
        print("  figure -> results/round9_A1_selfimmol_profile.png")
    except Exception as e: print("  (figure skipped:",e,")")
if __name__=="__main__": main()
