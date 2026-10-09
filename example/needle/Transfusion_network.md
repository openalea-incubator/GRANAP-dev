
# Transfusion network

**Context**
The transfusion tissue provides physically distinct pathways for sugar and water, reducing resistance between the vasculature and endodermis and mitigating flow constriction at the vascular flank. Dissipation of flow velocities through the transfusion tissue's branched structure allows for bidirectional transport of an inbound diffusive sugar flux against an outbound advective water flux across the endodermis.

**Granap Aim**

- Tracheid elements are apoplastic dead vessels, No Symplastic connection
- Transfusion parenchyma are usualy well connected but due to their round shaped nature, and the current lack of 3D, the 2D cross section does not always produce links from the endodermis to the vasculature elements. New edges must connect parenchyma to neighboring cells if they are disconected by a Tracheid and close enough so that the link represent a third in between cell in another plane (+ nodes). 

**Global Purpose** In MECHA, we want to be able to correctly simulate water and solute transport.

READ only: MECHA dir (`../MECHA-dev/`)

**Expected delivery**
A graph of the Transfusion network 

## Delivered — 02.09.2026

Figures: `outputs/transfusion_network_cells.png`, `_symplast.png`,
`_symplast_zoom.png` (the panel to compare against
`Transfusion_tissue_network.png`), `_apoplastic.png`, from
`transfusion_network.py`. Tests in `test/test_transfusion_network.py`.

**The key finding was that MECHA needed no changes at all.** It already maps
cgroup `17 = transfusion parenchyma` / `18 = transfusion tracheid`
(`hydraulic_cell.py`'s `CGROUP_TO_TYPE`) and already zeroes every symplastic
interface of 18 in `data_loader.py`'s `fplxheight_map` — `(3,18)`, `(5,18)`,
`(11,18)`, `(12,18)`, `(17,18)`, `(18,18)` all `0.0`, with `(13,18) = 1.76E6`
(xylem–tracheid) the single live link. GRANAP simply never populated `cgroup`
at all, so `int("")` raised in `network_builder.py`, the fallback `type_mapper`
contained neither transfusion string, and **both tissues silently became
cgroup 4 = cortex**: living, cortex-ranked, fully plasmodesmata-connected.
Adding `CGROUP_MAP` (`anatomy_writer.py`) fixed the tracheid's symplastic
isolation on the MECHA side by itself.

Two mechanisms deliver the rest:

1. `Organ.APOPLASTIC_ONLY_TYPES` (`NeedleAnatomy` sets
   `("transfusion tracheid",)`) suppresses the `plasmodesmata` edge on any wall
   flanked by such a cell. Its `membrane`/`wall` edges are untouched, so the
   tracheid keeps its full apoplastic path. Xylem and every other organ are
   deliberately unaffected.
2. `Organ.network_bridge_specs()` + Phase 9 of `NetworkExporter.export` builds
   the out-of-plane links, per `Transfusion_tissue_network.png`: **a virtual
   symplastic node with no polygon**, placed inside a traversed tracheid,
   shared by every path that crosses that tracheid, wired by `plasmodesmata`
   edges to the real cells on either side and by `membrane` edges to the
   **existing** wall nodes it crosses. It creates no walls and no junctions, so
   `n_walls`/`n_junctions` are untouched and the MECHA
   `[walls][junctions][cells]` index contract stays contiguous.

Three design decisions worth recording, because the obvious alternatives were
tried first and are worse:

- **The bridge is transmembrane and symplastic only, never apoplastic.** An
  earlier version gave each bridge its own wall nodes wired to the two nearest
  existing junctions. Because a virtual cell's walls all radiate from one
  point, they kept resolving to the same junction pair (180/180, then 19/35
  after a fix), so every bridge injected a spurious parallel apoplastic shunt
  between two points no anatomy connects. A virtual wall representing an
  out-of-plane membrane has no in-plane junctions; reusing the nearest ones was
  a kludge forced by the topology, not anatomy.
- **A node, not a cell.** Giving the bridge a real polygon (an
  `ellipse_to_polygon` sized like its neighbours) made virtual cells overlap
  each other and added 179% on top of the real transfusion-parenchyma area.
  Dropping the polygon removes both problems by construction — there is nothing
  to overlap and nothing to draw — while a nominal `area` on the `Cell` still
  gives MECHA a sane capacitance/volume term.
- **111 virtual nodes on 136 tracheids (82%) is accepted deliberately.** This
  is denser than the reference diagram, where only a minority of TT cells carry
  a blue node. Reviewed and kept: the nodes carry no polygon, no rendered area
  and no apoplastic path, so the cost is extra symplastic conductance and
  capacitance rather than distorted anatomy — consistent with a real 3-D
  transfusion tissue having an out-of-plane parenchyma beside nearly every
  tracheid. `transfusion_tissue.bridge_radius` / `bridge_max_links` are the
  knobs if this needs tightening later.

Verified: 0/136 tracheids carry a symplastic edge and all keep a membrane edge;
the symplast is **one** connected component over
{transfusion parenchyma, parenchyma, Strasburger cell, endodermis, phloem};
every `plasmodesmata` edge between two real cells has a shared wall (so all
bridging genuinely routes through a virtual node); no virtual node touches a
junction; repeat `export()` reproduces bit-identical node ids and positions.
MECHA round-trip (`NetworkBuilder(needle).populate_from_network()`) resolves
every cgroup, with 17→190 and 18→136 and no blank types or zero areas.

Known side effect: 294 walls now have 3 membrane neighbours instead of 2. This
does not disturb MECHA's `border_walls` classification (which keys on
`len(membrane_cells) == 1`), but MECHA's `distance_wall_cell` *sums* rather
than averages membrane distances, so those walls' values rise — a MECHA-side
quirk, not fixable from here.

The `test_needle_default_golden` failure is pre-existing and unrelated: that
golden census still lists the removed `resin duct` tag and lacks
`Str. Interstitial cell` entirely. See `pinaster_bugs.md`'s open items.