# Literature grounding for WeldScope

WeldScope is an in-process OCT weld-quality monitor (system class: Precitec
IDM / IPG). The design claims below are grounded in the peer-reviewed
empirical literature on OCT for laser welding. All 15 entries are verified
(DOIs resolve; metadata from OpenAlex/Crossref); the full machine-readable
corpus lives in the sibling repo
[`oct-research`](https://github.com/tobias-weiss-ai-xr/oct-research)
(`papers.yaml`, category `application/other`), which also exports the
concept knowledge graph used for the mappings in this document.

## Why synthetic data (verified, cited)

> "Public in-process weld-OCT data does not exist."

Confirmed as of 2026-09 across arXiv, OpenAlex, Zenodo and GitHub: every
empirical group records in-process coaxial/scanning OCT data with proprietary
industrial systems, and **none publishes the raw recordings**. The Feldman JLA
2020 battery-tab study (Sokolov et al.); the Fraunhofer IFSW cluster
(Stadter/Schmoeller et al.); the JLU/GSI copper-scanning work (Will et al.) —
all keep data in-house. The simulator is therefore the only tractable source
of physically consistent spectra, which is exactly what `crates/sim` provides.

## Verified bibliography (empirical OCT-for-welding)

### Keyhole depth measurement — coaxial OCT (Fraunhofer IFSW / JLA cluster)
- **Boley, Fetzer, Weber & Graf.** "Statistical evaluation method to determine
  the laser welding depth by optical coherence tomography."
  *Optics and Lasers in Engineering*, 2019. doi:10.1016/j.optlaseng.2019.03.014
  → percentile filtering on Poisson-weighted noise classification; mean depth
  error <5% vs metallography. **Anchors WeldScope's percentile-style depth
  extraction and pene_ratio.**
- **Mittelstädt et al.** "Novel approach for weld depth determination … deep
  penetration welding of aluminum and steel." *J. Laser Appl.*, 2019.
  doi:10.2351/1.5082263 → frequency distribution of OCT data has a local
  maximum that correlates with keyhole depth (esp. aluminium). **Anchors
  sub-pixel/peak-shape extraction in `crates/core`.**
- **Stadter, Schmoeller et al.** "Process control and quality assurance in
  remote laser beam welding by optical coherence tomography." *J. Laser
  Appl.*, 2019. doi:10.2351/1.5096103 (OA) → coaxial integration, angular /
  temperature dependence, seam tracking by measuring lines ahead of the
  process zone. **Anchors the coaxial geometry and `seam tracking` concept.**
- **Schmoeller et al.** "Inline weld depth measurement for high brilliance
  laser beam sources…" *J. Laser Appl.*, 2019. doi:10.2351/1.5096104 (OA) →
  OCT measuring spot ≈50 µm vs 55 µm single-mode process spot; Al vs Cu
  signal behaviour. **Anchors measurement-geometry assumptions.**
- **Stadter et al.** "Real-time prediction of quality characteristics … OCT
  and machine learning." *J. Laser Appl.*, 2020. doi:10.2351/7.0000077 →
  ML correlates keyhole-depth signal with weld-seam surface quality on Al, Cu,
  galvanized steel. **Anchors `crates/ai` verdict from depth-trace features.**
- **Xie, Wang et al.** "An Efficient Method for Laser Welding Depth
  Determination Using Optical Coherence Tomography." *Sensors*, 2023.
  doi:10.3390/s23115223 (OA) → DBSCAN noise removal + percentile filter; <5%
  error. **Anchors an alternative depth path (denoise → percentile).**

### Battery-tab / remote laser welding (Precitec ecosystem, ARM lasers)
- **Sokolov, Franciosa et al.** "Applying optical coherence tomography for
  weld depth monitoring … battery tab connectors." *J. Laser Appl.*, 2020.
  doi:10.2351/7.0000336 → Al 1050 foil 450 µm + Ni-plated Cu 300 µm lap joint;
  ARM laser; TwinTec dual-beam OCT (keyhole bottom + surface reference);
  "keyhole mapping" improves accuracy 0.22 → 0.11 mm. **Directly validates
  WeldScope's battery-connector use case and its ~sub-100 µm regime.**
- **Sokolov et al.** "Keyhole mapping to enable closed-loop weld penetration
  depth control…" *J. Laser Appl.*, 2020. doi:10.2351/7.0000086 (OA) →
  closed-loop penetration-depth control; decouples heat input (in-plane) from
  penetration (out-of-plane). **Anchors the `incomplete penetration` defect
  class and closed-loop concept.**
- **Brežan, Franciosa, Jezeršek et al.** "Fusing optical coherence tomography
  and photodiodes…" *J. Laser Appl.*, 2023. doi:10.2351/7.0000803 (OA) →
  40 kHz acquisition; OCT penetration depth + plasma/back-reflection
  photodiodes; 87% classification of weld scenarios. **Anchors the 100 kHz
  frame-rate target and confirms sensor-fusion headroom.**

### Scanning OCT / melt-pool fluctuations (JLU/GSI, copper)
- **Will, Jeron, Hoelbling et al.** "In-Process Analysis of Melt Pool
  Fluctuations with Scanning Optical Coherence Tomography for Laser Welding
  of Copper." *Micromachines*, 2022. doi:10.3390/mi13111937 (OA) → scanned
  beam separates keyhole/melt-pool; fluctuation feature classifies weld
  status (heat-conduction / stable / unstable deep penetration); spatter
  linkage. **Anchors `spatter_rate`, `humping_index`, and defect↔oscillation
  coupling.**
- **Will, Massieu Garcia et al.** "Algorithms for Weld Depth Measurement …
  Scanning OCT." *Micromachines*, 2022. doi:10.3390/mi13122243 (OA) →
  seven depth pipelines; **intensity accumulation** most accurate for
  scanning lines. **Grounds a second depth extractor in `crates/features`.**

### Inline coherent imaging (high-speed OCT) & X-ray validation
- **Webster et al.** "Automatic laser welding and milling with in situ inline
  coherent imaging." *Optics Letters*, 2014. doi:10.1364/ol.39.006217 (OA) →
  ICI at 312 kHz line rate, µs capture, kW-class keyhole welding, adaptive
  control. **Upper bound for the real-time budget.**
- **Fleming, Clark, Fan et al.** "Synchrotron validation of inline coherent
  imaging for tracking laser keyhole depth." *Additive Manufacturing*, 2023.
  doi:10.1016/j.addma.2023.103798 (OA) → ICI vs synchrotron X-ray: >80% of
  depth samples within ±14 µm; keyhole-sidewall collapse → bubble/pore
  pinch-off; outliers from multiple reflections. **Grounds the `pore` defect
  physical mechanism and error bounds.**

### Polymer / glass (adjacent applications, strength of the modality)
- Schmitt et al., *Physics Procedia*, 2014. doi:10.1016/j.phpro.2014.08.055.
- Kim et al., *IEEE Access*, 2018. doi:10.1109/access.2018.2882527 (OA) —
  860 nm SD-OCT for weld-boundary + porosity in laser transmission welding.

## Knowledge graph → WeldScope mapping

The `oct-research` concept graph (59 concepts, 113 co-occurrence edges) was
sampled for the concepts WeldScope actually implements. Correspondence table:

| Concept (KG) | WeldScope component | How |
|---|---|---|
| optical coherence tomography / spectral domain | `crates/core` | SD-OCT FFT chain (BG-sub → k-resample → Hann → realfft → log) |
| keyhole / keyhole depth / weld depth / penetration depth | `crates/core` + `Features.mean/std/min/max` | depth trace extraction; pene_ratio |
| spatter / humping | `Features.spatter_rate` / `humping_index` | spike/periodicity detectors (Will et al., FRESH-JLA) |
| pore / porosity | `Features.pore_count` | threshold crossings; sidewall-collapse mechanism (Fleming et al.) |
| incomplete penetration / closed loop | defect class | depth-ramp signature; control headroom (Sokolov et al.) |
| copper / aluminum / galvanized steel | `config/sim.json`, `crates/sim` | material-dependent signal behaviour (Schmoeller et al.; Stadter et al.) |
| battery tab | README use case | Al-foil/Ni-Cu lap joint (Sokolov et al.) |
| seam tracking / process monitoring / quality assurance | roadmap | pre-/post-process OCT features (Stadter et al. 2019) |
| inline coherent imaging | `io` wire format | optional high-rate acquisition variant (Webster et al.) |

## Grounding gaps currently assumed (avenues for the KG to close)

1. **Depth-extraction algorithms** — WeldScope uses gated max + center-of-mass.
   The literature offers validated alternatives (percentile filter, DBSCAN+
   percentile, intensity accumulation) that could be added as pluggable
   extractors and compared on the synthetic corpus.
2. **Defect↔feature calibration** — thresholds (`thresh`, `spike_delta`) are
   currently engineering constants; the papers report accuracy numbers
   (<5% depth error; 87% classification; ±14 µm vs X-ray) that can calibrate
   acceptable operating envelopes.
3. **Sensor-fusion** — photodiode fusion (Brežan et al.) is a documented
   headroom beyond single-sensor OCT verdicts.
