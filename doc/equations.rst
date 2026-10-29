=========
Equations
=========

GRANAP computes geometry, not physics: an organ's cross-section, its cells drawn by a Voronoi tessellation of seeds
placed ring by ring, then the network of its walls, wall junctions and cells that a hydraulic model (MECHA) solves
on. This page writes the rules the code applies, step by step, with the function computing each. The port computes
the same as the legacy (:doc:`legacy`); where a function was converted for speed, the rule written here is the
legacy's.

Notation: lengths in mm (GRANAP's unit), angles in radians unless stated. A layer (a ring of cells) has a cell
diameter :math:`d` (radial) and a cell width :math:`w` (tangential, :math:`w = d` when not given). :math:`U(a, b)`
is a uniform draw, :math:`\mathcal{N}(\mu, \sigma)` a normal one, all from the organ's one random generator
(``np.random.default_rng(seed)``), in the order of the pipeline. :math:`A(P)` is a polygon's area,
:math:`\mathrm{buf}(P, \delta)` its buffer by :math:`\delta` (negative: inwards; 16 segments per quarter circle).
Unless stated, angles are taken about the origin, at which the organ is centred.


The layers (``organ_class``)
============================

**The base shape** (``RootAnatomy._create_base_shape``, ``StemAnatomy._create_base_shape``, through
``GeometryProcessor.contour_polygon``): a circle, an ellipse, a square, a rectangle, a triangle, a star or a
superellipse of radius

.. math::

   R = \frac{T_c}{2} + \sum_\ell d_\ell\, n_\ell

(``RootAnatomy._calculate_root_radius``, ``StemAnatomy._calculate_radius``), :math:`T_c` the central cylinder's
thickness (the stele, or the stem's pith; the dicot stem adds its secondary-growth annulus) and :math:`n_\ell` each
layer's number of rings. The needle is a half ellipse of width :math:`W` and thickness :math:`T`
(``NeedleAnatomy._create_base_shape``), the leaf a blade outline (below).

**The rings** (``Organ._build_layer_polygons``). The layers, outermost first, are expanded into one entry per ring.
An outer guard ring ``outside``, :math:`P_0 = \mathrm{buf}(P_\text{base}, d_1/2)`, holds seeds of diameter
:math:`d_1/3` whose cells are dropped after the tessellation. Each ring :math:`i` is the centre line of its cells:

.. math::

   P_i = \mathrm{buf}\!\left(P_{i-1},\ -\frac{d_{i-1} + d_i}{2}\right), \qquad d_0 = d_1,

so that successive rings are :math:`(d_{i-1} + d_i)/2` apart, optionally smoothed (``LAYER_SMOOTH_FACTOR``: 0.5 by
default, 0 for roots, stems and leaves). The needle offsets a ring by an angle-dependent distance
:math:`\delta\,M(\alpha)` instead (``NeedleAnatomy._offset_layer_polygon``, ``GeometryProcessor.variable_buffer_polygon``):
each of 720 resampled boundary points moves radially from the centre, :math:`r' = \max(r + \delta M(\alpha),
10^{-6})`, :math:`M` interpolated circularly between given (angle, multiplier) points.

**The central rings** (``RootAnatomy._create_central_layers``, ``StemAnatomy._create_central_layers``). Inside the
last ring, the stele (or pith) is peeled the same way with a diameter graded from its edge to its centre:

.. math::

   d(\rho) = d_\text{edge} + (d_\text{centre} - d_\text{edge})\, f(\rho), \qquad
   \rho = \min\!\left(\sqrt{A(P)/\pi}\,/\,r_s,\ 1\right),

:math:`r_s = \sqrt{A_\text{stele}/\pi}`, until :math:`A(P) \le \pi (d/2)^2` (``math_functions.rescale``). The
gradient functions (``math_functions.five_pl``, ``linear``):

.. math::

   f_\text{5PL}(t) = \frac{1}{\left(1 + (t/c)^b\right)^m}\ \ (f(t \le 0) = 1), \qquad f_\text{lin}(t) = \max(0, 1 - t),

defaults :math:`c = 0.5`, :math:`b = 3`, :math:`m = 1`. Note :math:`f(1) \ne 0`: the edge value is reached only
approximately.

**Smoothing** (``GeometryProcessor.smoothing_polygon``): a ring resampled to 200 points equally spaced along it
(``resample_coords``), then 10 Laplacian passes

.. math::

   p_k \leftarrow (1 - s)\, p_k + s\, \frac{p_{k-1} + p_{k+1}}{2}.


The seeds (``generate_cell``)
=============================

**Cells on a ring** (``CellGenerator.cells_on_layer``). On a ring of perimeter :math:`L`,
:math:`N = \lceil L / w \rceil` positions are resampled at equal arc length from a start shifted by
:math:`U(0, \text{shift}\cdot w)`; the closed ring's first and last positions coincide and the first is dropped:
:math:`N - 1` cells, spaced :math:`L / (N - 1)`.

**Seeds of a cell** (``CellGenerator.cell_border``, ``GeometryProcessor.draw_ellipse``). Each cell is a ring of
Voronoi seeds sharing its ``id_group``, on an ellipse centred on the cell's position, oriented along the ring
(:math:`\theta = \operatorname{atan2}` of the next position minus the previous one):

.. math::

   \mathbf{x}_j = \mathbf{c} + a\cos t_j\,\mathbf{e}_\theta + b\sin t_j\,\mathbf{e}_{\theta + \pi/2}, \qquad
   a = 0.35\, w,\ b = 0.35\, d, \qquad t_j = \frac{2\pi j}{n - 1},\ j = 1 \ldots n - 1,

with :math:`n` = ``n_points`` (by default 15, or 10 when :math:`w = d`): 14 or 9 seeds. Seeds falling inside the next
inner ring are dropped. A ``parenchyma`` ring gets one seed per cell; the needle's transfusion ring one seed ring of
radius :math:`0.35\,d` per cell, each cell a tracheid with probability :math:`p_{tt} = r/(1 + r)`
(``CellGenerator.generate_cells_info``).

**Overlaps** (``CellGenerator.resolve_cell_border_overlaps``). Each cell's footprint is the convex hull of its seeds
buffered by :math:`0.2\,d` (a disc of radius :math:`d/2` below 3 seeds). Cells are ranked by (``id_layer``,
``id_group``), inner first; a lower-ranked cell's seeds inside a higher-ranked footprint are removed. The ring seeds
inside the vascular tissue's union are removed (``Organ._generate_pipeline``).

**Seeds of a placed vessel** (``tissue_class.place_packed_group``, ``seed_ring_inset``): 24 or 25 seeds on the
vessel's outline inset by :math:`\min(g, 0.9\,r)` when the gap :math:`g` to the neighbouring seeds is known (the
monocot metaxylem: half the stele's cell diameter, so that the Voronoi wall falls on the outline), else
:math:`0.15\,r`.


Sizes and packings
==================

**Size draws** (``distribution._sample``): a vessel's or a sieve element's diameter, of mean :math:`\mu` and
standard deviation :math:`\sigma`, moment-matched for each family, then clipped (not rejected) to
:math:`[d_\text{lo}, d_\text{hi}]`:

.. math::

   \text{normal: } \mathcal{N}(\mu, \sigma); \quad
   \text{lognormal: } \sigma_l^2 = \ln\!\left(1 + \sigma^2/\mu^2\right),\ \mu_l = \ln\mu - \sigma_l^2/2; \quad
   \text{gamma: } \vartheta = \sigma^2/\mu,\ k = \mu/\vartheta;

logistic of scale :math:`\sigma\sqrt3/\pi`; uniform on :math:`\mu \pm \sigma\sqrt3`; empirical
:math:`\mu + \sigma z`, :math:`z` a standardised measured value drawn at random. A family other than normal with
:math:`\sigma \le 0` gives :math:`\mu`.

**Circle packing** (``GeometryProcessor.pack_circles``). A region is filled with vessels, largest inscribed circle
first: from a stack of regions (the region itself first), the circle inscribed in the region popped
(``_chebyshev_center``: GEOS's maximum inscribed circle at tolerance :math:`10^{-4}` of the region's extent, the
region snapped to a :math:`10^{-9}` grid) gets the radius :math:`r = \min(r_\text{ins}, D/2)`, and the region minus
the circle goes back on the stack, until the packed area reaches the proportion asked for (the last circle scaled by
:math:`\sqrt{\text{budget}/\text{area}}`). The target diameter :math:`D`, with
:math:`d_\text{min} = \max(d_\text{max} - 3\sigma,\ 0.01\, d_\text{max})` by default:

- without a gradient, :math:`D` drawn of mean :math:`(d_\text{max} + d_\text{min})/2` in
  :math:`[d_\text{min}, d_\text{max}]`;
- with one, :math:`D = d_\text{min} + (d_\text{max} - d_\text{min}) f(t)`, then drawn around it if :math:`\sigma > 0`,
  :math:`t` the distance from the gradient's centre over the largest one (or :math:`(r - r_0)/(r_1 - r_0)` in a
  radial range), reversed (:math:`1 - t`) for ``edge``, :math:`2|t - 1/2|` for ``middle``.

Where the inscribed circle is too small and ellipses are allowed, an ellipse of the target area,
:math:`b = r_\text{ins}`, :math:`a = \min(r_D^2/b,\ \text{max\_aspect}\cdot b)` (``_fit_target_ellipse``).

**Inscribed ellipses** (``GeometryProcessor.fit_inner_ellipse``): an ellipse centred at the region's inscribed
circle, its axes shrunk by 0.98 until it fits (down to 0.2 of their size). **Slices**
(``GeometryProcessor.pizza_slice``): :math:`n` wedges :math:`[2\pi i/n, 2\pi(i+1)/n]` about the region's centroid.


Roots (``root_monocot_class``, ``root_dicot_class``, ``root_class``)
====================================================================

**Monocot, default** (``MonocotRootAnatomy.metaxylem_positions``, ``vascular_elements_in_slice``): the stele inset by
a quarter of the vessel diameter is cut into :math:`n` slices (``n_vascular_bundles``); a metaxylem vessel of
diameter drawn from (``vessel_diameter``, its sd), at least a tenth of it, is fitted in each slice. The protoxylem
and phloem (``fit_phloem_protoxylem_elements``) alternate in :math:`2 n_p` slices of the stele's outer band (width
:math:`1.1\max(d_{px}, d_{s})`), each cluster an ellipse along the radius scaled by
:math:`\min(1, r_\text{ins}/(\max(w, h)/2))` and packed.

**Star xylem** (monocot ``star``, dicot; ``RootAnatomy._xylem_star_region``, ``GeometryProcessor.star_polygon``):
:math:`n` arms, arm :math:`k` at :math:`2\pi k/n`, an arc of radius :math:`r_\text{peak}` spanning
:math:`\pm\,\text{arc}_\text{top}/r_\text{peak}`, joined by straight sides to valley arcs of radius
:math:`r_\text{valley}`, both bounded by the stele's inscribed radius; minus a pith disc; packed with vessels graded
from the centre over :math:`(r_\text{pith}, r_\text{outer})`. Stele cells whose seeds' hull is at least 0.6 covered
by vessels are dropped (``_remove_stele_engulfed_by_xylem``). The phloem sits in the valleys,
:math:`\theta_k = 2\pi(k + 1/2)/n`, at

.. math::

   r_c = r_v + a + \frac{h}{2} + (r_s - a - h - r_v)\,\rho_\text{rel}

(``_phloem_valley_zones``; :math:`a = 0` for the monocot root, the cambium's clearance for the dicot one).

**Monocot, arch** (``_arch_split_radius``, ``_fit_arch_metaxylem``, ``_fit_arch_protoxylem``): a split radius
:math:`r_\text{split} = \max(r_\text{pith}, r_\text{outer} - \text{depth})` (depth by default
:math:`0.35(r_\text{outer} - r_\text{pith})`); the metaxylem at :math:`2\pi j/n`, egg-shaped (two half ellipses,
area :math:`\pi b(a_\text{out} + a_\text{in})/2`), tangentially limited to
:math:`\pi r_\text{ring}/n - d/8`; the protoxylem packed in trapezoidal corridors from each metaxylem to the pole tip
at :math:`r_\text{outer}`, graded over :math:`(r_\text{split}, r_\text{outer})`.

**Dicot, primary cambium** (``fit_primary_cambium_elements``): a star contour smoothed, its part within the visible
distance seeded with cambium cells.

**Dicot, secondary growth** (``fit_secondary_xylem``, ``_build_annual_bands``, ``_build_medullar_ray_polygons``,
``fit_secondary_phloem``): between the primary and the secondary cambium (inscribed radii :math:`r_p`, :math:`r_c`),
:math:`n_\text{ring}` annual bands of width :math:`(r_c - r_p)/n_\text{ring}`, each packed with vessels restarting
its gradient, in sectors of half-width :math:`(2\pi/n)\,p_\text{stele}/2` about :math:`2\pi(k + 1/2)/n`; between
them ray parenchyma in radial lanes that split where their arc exceeds a threshold :math:`w + 3\sigma` (each lane's drawn as
:math:`U(0.7, 1.3)` times it, within :math:`[0.5, 1.5]` times it). Medullary rays: :math:`n_\text{new} = \operatorname{round}(\text{rate}\cdot
\text{span})` new rays starting at

.. math::

   r_j = r_0 + \frac{j + 1/2}{n_\text{new}}\,\text{span} + \mathcal{N}(0, \sigma_r),

each placed in the widest angular gap, a strip of constant width. The secondary phloem is a band of the given
height outside the cambium, cut into trapezes per sector (base the cambium's arc, top width
:math:`\min(w_\text{top}, 2\,\text{base})`); within the alive distance, each sieve element has a companion cell.
Sieve elements are drawn with the proportion :math:`p_s\, d_s^2 / (d_s^2 + d_c w_c)`.

**The root series** (``root_series.RootSeries``, ``DicotRootSeries``). Sections along one root, from collar to apex,
with tracked vessels. A schedule :math:`(a, b)` is linear in the length :math:`L` between the first and last sampled
lengths. Fused vessels: :math:`m` primordial vessels merged (the closest pair first, ``_all_partitions``) into one of
radius

.. math::

   r = r_v(L)\,\sqrt{1 + \kappa\,(m - 1)}, \qquad \kappa = 0.4

(``_fused_radius``), its ``track_id`` the smallest of its members'; vessels assigned to their slots by the Hungarian
algorithm on the squared displacement (``_optimal_assign``), migrating as :math:`\mathbf{x} = \mathbf{x}_a +
\alpha(\mathbf{x}_s - \mathbf{x}_a)`, :math:`\alpha = \min(1, |L_a - L|/L_\text{migration})`. Each section is a
``RootAnatomy`` with its stele thickness :math:`2 r_\text{stele}(L)` and its vessels prescribed.


Stems, leaves and needles
=========================

**Monocot stem bundles** (``MonocotStemAnatomy._place_random_bundles``, ``_place_spaced_bundles``,
``_place_even_bundles``): scattered by darts, :math:`r = R\, U^{0.35}` over the whole section (biased to the
periphery) or area-uniform :math:`r = \sqrt{U(r_\text{hi}^2 - r_\text{lo}^2) + r_\text{lo}^2}` in a band, rejected
within :math:`0.8\max(w, h)` of a placed bundle or when the envelopes overlap; or the best of 8 candidates (the
farthest from those placed); or evenly at :math:`2\pi k/n`. Each bundle's phloem faces outwards.

**Dicot stem ring** (``DicotStemAnatomy._contour_slots``, ``_ring_slots``): bundles at equal arc length on the
ring's contour, along its outward normal, their number reduced until neighbours do not overlap.

**A bundle** (``vascular_bundle.bundle_layout``, ``partition_banded``, ``_outward_caps``, ``_grow_bundle_sheath``):
its envelope, in a frame whose :math:`+y` points outwards, cut in bands by fractions (inner phloem, xylem 0.5,
cambium 0.08, phloem 0.35, normalised), or concentric (amphivasal, amphicribral). Tapered outward fibre caps:
:math:`n(a) = n_\text{flank} + (n_\text{pole} - n_\text{flank})\, a/(\pi/2)` files at the angle :math:`a` from
the flank. The outer sheath's cells where the bundle's and the ground's cell sizes :math:`x`, :math:`y` differ by
more than 4 times have the size :math:`\sqrt{xy}`.

**Continuous cylinder** (``ContinuousDicotStemAnatomy._build_cylinder``): xylem and phloem annuli of given
thicknesses inside and outside the cambium, the phloem in
:math:`\max(6, \operatorname{round}(\text{circumference}/(10\,d_s)))` sectors.

**Leaf blade** (``LeafAnatomy._create_base_shape``, ``_mid_line``, ``_half_profile``): the mid-line

.. math::

   y_m(x) = A\sin\!\left(\frac{k\pi x}{W/2}\right) - \mathbb{1}_{|x| < f/2}\; s\,\frac{1 + \cos(2\pi x/f)}{2},

the half thickness :math:`(T/2)\sqrt{1 - (2x/W)^2}` (or interpolated from a profile), plus raised-cosine ribs.
Veins at :math:`x` evenly spaced over :math:`0.75\,W`, kept when clear of wider veins already placed by
:math:`(w + w')/2 + 0.02`; the palisade in :math:`\min(\lfloor T(x)/d_p \rfloor, n_p)` files under the adaxial
surface, the rest spongy (``DicotLeafAnatomy._create_central_layers``).

**Needle** (``NeedleAnatomy._calculate_needle_thickness``, ``_calculate_needle_width``,
``pole_and_corner_angles``): :math:`T = T_c + 2t_l`, :math:`t_l = \sum d\,n`,

.. math::

   W = 2\sqrt{\frac{(L_c/2 + t_l)^2}{1 - (t_l/T)^2}},

and the reference point of its angles at :math:`y_c = 3.5\,T/(3\pi)`. A resin duct's assembly radius
(``_duct_assembly_radius``): :math:`D/2 + t + 0.075\,t`, :math:`D = \text{lumen} + 2 d + 2 d_\text{sheath}`,
:math:`t = \sqrt{d_\text{sheath}\, d_\text{host}}`. The needle's aerenchyma target per file is divided by
:math:`n^{1.12} + 1` instead of :math:`n` (``_aerenchyma_target_denominator``).

**Stomata** (``special_tissues.place_stomata``, ``CellGenerator.create_stomata``): on an epidermis cell, two guard
cells (ellipses at :math:`\pm\,\text{width}/2`), a pore (a rectangle width × depth between them) and a sub-stomatal
chamber (an ellipse of half axes :math:`0.75\,\text{width}`, the chamber depth), carved out of their neighbours.


The tessellation (``generate_cell``)
====================================

**Jitter** (``CellGenerator.voronoi_diagram``): each seed moves by :math:`U(-10^{-4}, 10^{-4})\,d` in x and y, one
block drawn in the seeds' order; then one scipy Voronoi diagram of all seeds.

**Grouping** (``CellGenerator.process_voronoi_groups``): unbounded regions and the guard ring's are dropped; a cell
is the union of its seeds' regions, its attributes its first seed's.

**The snap** (``CellGenerator._build_topology``). The cells' vertices are snapped together, within

.. math::

   \varepsilon = 0.01\; P_5(\ell_e)

(:math:`P_5` the 5th percentile of the edge lengths :math:`\ell_e > 0`; :math:`10^{-4}` without edges), greedily in
vertex order: each vertex not yet visited and all vertices within :math:`\varepsilon` of it go to their mean, rounded
to 6 decimals. A junction is a vertex of three cells or more, or one whose incident edges are shared by more than
one pair of cells. A cell's polygon of several parts is reduced to its largest; holes are ignored.

**Simplification** (``CellGenerator.simplify_cells``): each cell rebuilt from its junction vertices (and the
vertices of shape-protected cells), all its vertices if fewer than 3 remain. A cell nested in a larger one (its
representative point inside, at least 0.1 of its area covered) is merged into the smallest such cell
(``remove_nested_cells``).


Air spaces and gaps (``organ_class``)
=====================================

**Intercellular spaces** (``Organ._apply_intercellular``): for the listed tissues and a smoothness :math:`s`, each
cell is smoothed (Laplacian, above); the air is the tissue's union minus the union of the smoothed cells, the pieces
reaching the union eroded by :math:`d_\text{min}/2` (inside the tissue), simplified within
:math:`0.05\,\operatorname{median}\sqrt{A_c/\pi}`, above :math:`10^{-6}` mm². A pocket against a reflex corner
(interior angle above 185°) is rebuilt as a quadrilateral hugging that corner (``_center_air_pocket``). The pockets
are carved out of their hosts and inserted as air-space cells (``special_tissues.seat_air_spaces``).

**Aerenchyma** (``Organ.add_aerenchyma``, ``_fill_plan``). In the listed tissue, the candidates are its cells but its
innermost ring, and the air spaces. With the request :math:`p`, the target area and its share per file
:math:`q = 0 \ldots n - 1`:

.. math::

   A^* = p\,(A_\text{tissue} + A_\text{air}), \qquad A^*_q = \frac{A^* - A_\text{air}}{n},

:math:`A^*` clamped to the candidates' area. File :math:`q` is the sector
:math:`[\theta_0 + 2\pi q/n,\ \theta_0 + 2\pi(q + 1)/n)`, :math:`\theta_0 = U(0, 2\pi)`; its cells are ordered as
a connected channel grown from the cell nearest the sector's centre, by angular distance to it (type 1), or by
distance from that cell (type 2); one cell per file in turn becomes air until each file's area reaches
:math:`A^*_q`. Then the touching air spaces of a sector are fused, the section re-simplified, and the tissue carved
by the air (``merge_intercellular_aerenchyma``).

**The calibration** (``Organ._add_aerenchyma_to_proportion``). The realised proportion
:math:`\hat p = A_\text{air}/(A_\text{air} + A_\text{tissue})` is measured after the fusion; the internal target is
bisected on :math:`[0, p]` (or :math:`[p, 1]` if the first attempt falls short) until :math:`|\hat p - p| \le
0.005`, the interval is under :math:`10^{-4}`, or 20 steps; the best attempt is kept.

**Gaps** (``Organ.find_gaps``, ``fuse_gaps``): the outline minus the cells' union and the declared voids (the
medullary cavity), opened by :math:`0.15\, \bar d` (:math:`\bar d = 2\sqrt{\operatorname{median}A_c/\pi}`) to remove
slivers, the pieces enclosed by the tissue kept. Each gap is unioned into the cell, of the tissue sharing the most
border with it, that shares the longest border.

**Cell properties** (``CellManager.recalculate_cell_properties``): area :math:`A_c`, radius the centroid's distance
to the origin, angle :math:`\operatorname{atan2}(y, x)` of the cell's first seed, ``id_cell`` the cell's index.


The network (``metafspm.network``, ``anatomy_writer.NetworkExporter``)
======================================================================

The nodes are the walls :math:`0 \ldots N_w - 1`, the junctions next (sorted by :math:`(x, y)`), then the cells. A
wall :math:`w` is the polyline :math:`p_0 \ldots p_m` of a cell's snapped ring between two consecutive junctions (a cell with fewer than two junctions has one closed wall, its length the ring's perimeter and its midpoint
the mean of the ring's vertices), shared by the cells whose rings run along it:

.. math::

   \ell_w = \sum_{k=0}^{m-1} |p_{k+1} - p_k|, \qquad \mathbf{m}_w = \frac{1}{m + 1}\sum_{k=0}^{m} p_k, \qquad
   e_w = \sum_{c \ni w} e(\text{type}_c) + \mathbb{1}_{n_w = 1}\, e_\text{outer},

:math:`e` the per-tissue wall thicknesses (``DEFAULT_CELL_WALL_THICKNESS``, µm; 1 for a type not listed,
2 for the outer wall). The edges, with :math:`\mathbf{g}_c` a cell's centroid:

=============== ===================== ======================== =========================================================
edge            between               length                   vector and distances
=============== ===================== ======================== =========================================================
membrane        cell :math:`c`, wall  :math:`\ell_w`           :math:`\mathbf{d} = \mathbf{m}_w - \mathbf{g}_c`,
                :math:`w`                                      :math:`\text{dist} = |\mathbf{d}|`; thickness :math:`e_w`
wall            junction :math:`j`,   :math:`\ell_w/2`         :math:`\mathbf{d} = \mathbf{x}_j - \mathbf{m}_w`;
                wall :math:`w`                                 :math:`\bar\delta_w` = mean over the wall's cells of
                                                               their membranes' dist; lateral distance
                                                               :math:`\bar\delta_w + |\mathbf{d}|`
plasmodesmata   two cells of a wall   :math:`\ell_w`           :math:`\mathbf{d} = \mathbf{g}_b - \mathbf{g}_a`
air_link        two air nodes         :math:`|\mathbf{d}|`     :math:`\mathbf{d}` between their centroids
=============== ===================== ======================== =========================================================

A membrane is ``wall_air`` when its cell is an air node: a protected air space or a stomatal pore. Plasmodesmata
join two cells of a two-cell wall that are both air or both not, neither an apoplastic-only type (the needle's
transfusion tracheids). Two protected air spaces are air-linked when a path (new junction, old junction, new
junction) joins them through the junctions their protected shapes created, or when their polygons are within the
median wall length of each other; each pore is linked to its nearest protected air space, its sub-stomatal chamber.
A cell's MECHA cell group is its ``cgroup`` or, when 0, its type's (``CGROUP_MAP``, 4 when absent).


Extrusion (``generate_cell_3d``)
================================

``Organ.generate_cells_3d`` extrudes the 2D section along :math:`z \in [-Z/2, Z/2]`, :math:`Z = n_\text{axial}\,
h_\text{max}` (8 by default): the vessels, sieve elements and companion cells (and the air spaces, mode ``before``) as one prism, every other
cell as a column of prisms of height :math:`h` (its layer's ``axial_height``, else :math:`5\,d`) from a random phase
:math:`U(0, h)`; a prism's volume :math:`A_c\,\Delta z`. In mode ``after``, prisms of the tissue are made air in a
random order until the air volume reaches the requested share of the band's.


The component (``metafspm.component``, ``metafspm.library``, ``metafspm.vessels``)
==================================================================================

**Classes** (``ClassInput.binned``, ``class_key``). A segment's class is the tuple of its class inputs binned,

.. math::

   \hat v = \Delta\, \operatorname{round}\!\left(\frac{v\, f}{\Delta}\right)

(rounded to 9 decimals; 0 when :math:`\Delta = 0`): by default the diameter from the radius in m,
:math:`f = 2000`, :math:`\Delta = 0.05` mm, and the distances from the tip and from the root–shoot junction (the
lengths of the segments up to it, ``distance_from_junction``), not binned. A class input with a target sets a GRANAP
field to :math:`\hat v`. The parameters of a template: the preset, then the plant's dict, the class's own
parametrization, the class inputs' targets (``AnatomyLibrary.template``, ``merged``).

**Vessels** (``vessels.vessel_ids``): every metaxylem, protoxylem, xylem, phloem and sieve-element cell gets a
``vessel_id`` in cell order, a family (0 xylem, 1 phloem), and a position about the section's centre
:math:`\mathbf{c} = \sum A_c \mathbf{g}_c / \sum A_c`: its angle, and its distance relative to the largest one. Its
pole is the bundle containing it, the phloem pole containing it, else the sector
:math:`\lfloor \theta/(2\pi/n) \rfloor`.

**Axial junctions** (``vessel_pairs``, ``junction_geometry``). Between a section and its linked parent, the vessels of
each family are paired by equal ``track_id``; every other child vessel with its nearest parent vessel, and every
other parent vessel with its nearest child vessel, in :math:`(\rho\cos\theta, \rho\sin\theta)`: a fusion or a split
gives one-to-many junctions. A junction's length is :math:`(L_\text{parent} + L_\text{child})/2`, its vector the two
vessels' in-plane offset; xylem junctions are apoplastic, phloem junctions symplastic.

**3D quantities** (``GranapAnatomy._wall_surface``, every time step, and ``_write_3d`` at instantiation), with the
segment's length :math:`L_s` in mm:

.. math::

   S_w = \ell_w L_s, \qquad V_c = A_c L_s, \qquad S_\text{membrane} = \ell_w L_s, \qquad
   \ell_\text{junction} = \frac{L_a + L_b}{2}.

**Inheritance** (``granap_inheritance``): a new Compartment takes its values from the Compartment of the same
template node in the nearest upstream section having it; else from the protoplast of the same tissue nearest in
:math:`(\rho\cos\theta, \rho\sin\theta)` upstream; else its declared default. A replaced anatomy inherits nothing.

**Leaf surface** (``GranapAnatomy.update_leaf_surfaces``): per leaf organ, one surface Compartment joined to every
stomatal pore; a pore's width and depth are the short and long sides of its polygon's minimum rotated rectangle, the
stomatal Connection's length its depth.
