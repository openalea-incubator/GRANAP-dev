"""
The XML writer, converted (devplan_metafspm.md, §43, C2): GRANAP's AnatomyWriter.write_to_xml (anatomy_writer.py),
copied, with one edit: the topology comes from the organ (MPGBacked.section_topology: the rings stored in the MPG for
the section's cells) instead of CellGenerator._build_topology, so that no class method is replaced for the process
while it writes. Every other line is GRANAP's.
"""
from openalea.granap.anatomy_writer import AnatomyWriter, CGROUP_MAP, Cell, CellGenerator, NetworkExporter, io


class MPGAnatomyWriter(AnatomyWriter):
    """GRANAP's writer, its walls numbered from the organ's topology."""

    def write_to_xml(self, path: str):
        """
        Write the root anatomy as an XML file matching GRANAR format.

        Cell ordering and wall IDs are derived from ``self.organ.generate_cells()``
        using the same GeoDataFrame index as ``NetworkExporter.export``, so that
        XML cell id attributes map 1-to-1 to the GRANAP graph cell node ids.
        """

        # One mapping for both exporters: CGROUP_MAP is what NetworkExporter.export
        # resolves cell nodes with, so the XML and the graph agree by construction.
        # (This used to be a separate local dict that drifted from it -- it mapped
        # "cambium" to 11 instead of 12, listed "stele" twice, and sent an unknown
        # tag to group 0, which MECHA reads as no tissue at all.)
        cellgroups = CGROUP_MAP

        # Use the same GeoDataFrame (and same index) as NetworkExporter.export().
        # IMPORTANT: apply the IDENTICAL valid-geometry filter so that both paths
        # call _build_topology with the exact same polygon set -> same KD-tree
        # snap_tol, same clusters, same canonical junction coordinates.
        cells_gdf = self.organ.generate_cells()
        valid_mask = cells_gdf["geometry"].notna() & cells_gdf["geometry"].apply(
            lambda g: g is not None and not g.is_empty
        )
        valid_gdf = cells_gdf[valid_mask]

        polys    = list(valid_gdf["geometry"])
        cell_ids = list(valid_gdf.index)

        # Cells flagged ``protect_topology`` (needle mesophyll air-space rhombi)
        # keep their walls from collapsing into a single node.
        # (see CellGenerator._build_topology's ``protect_ids``).
        protect_ids = (
            set(valid_gdf.index[valid_gdf["protect_topology"]])
            if "protect_topology" in valid_gdf.columns else None
        )

        cell_vkeys, _, _, junction_set, protected_shape_set = self.organ.section_topology(               # C2: the organ's topology, no class patch
            polys, cell_ids, protect_ids=protect_ids
        )

        wall_registry = {}
        next_wall_id  = 0
        cell_walls    = {idx: [] for idx in cell_ids}

        for row_idx, vkeys in cell_vkeys.items():
            n = len(vkeys)
            junc_positions = [i for i in range(n) if vkeys[i] in junction_set]

            if len(junc_positions) < 2:
                # no junctions → single wall loop
                shape_signature = tuple(vk for vk in vkeys if vk in protected_shape_set)
                wall_key = (tuple(sorted(vkeys)), shape_signature,)
                if wall_key not in wall_registry:
                    wall_registry[wall_key] = {"id": next_wall_id, "points": list(vkeys) + [vkeys[0]]}
                    next_wall_id += 1
                cell_walls[row_idx].append(wall_registry[wall_key]["id"])
                continue

            for jp in range(len(junc_positions)):
                start_idx = junc_positions[jp]
                end_idx   = junc_positions[(jp + 1) % len(junc_positions)]

                segment = []
                i = start_idx
                while True:
                    segment.append(vkeys[i])
                    if i == end_idx:
                        break
                    i = (i + 1) % n

                if len(segment) < 2:
                    continue

                junc_start = segment[0]
                junc_end   = segment[-1]
                wall_key   = tuple(sorted((junc_start, junc_end)))

                if wall_key not in wall_registry:
                    wall_registry[wall_key] = {"id": next_wall_id, "points": segment}
                    next_wall_id += 1

                cell_walls[row_idx].append(wall_registry[wall_key]["id"])

        xml_lines = [
            '<?xml version="1.0" encoding="utf-8"?>',
            '<granardata>',
            '\t<metadata>',
            '\t\t<parameters>',
            '\t\t\t<parameter io="0" name="python_export" type="default" value="1"/>',
            '\t\t</parameters>',
            '\t</metadata>',
            f'\t<cells count="{len(valid_gdf)}">'
        ]

        # Write cells using GDF index as id — matches graph node numbering
        for row_idx, row in valid_gdf.iterrows():
            # Same precedence NetworkExporter.export uses for cell nodes: an
            # explicit Cell.cgroup wins (that is how imported real data keeps its
            # own group integers), else the tag is looked up. The old default of
            # 0 was not a soft failure -- MECHA does a bare CGROUP_TO_TYPE[cgroup]
            # lookup and 0 is not a key.
            group_id = int(row.get("cgroup") or cellgroups.get(row.get("type", ""), 4))
            xml_lines.append(f'\t\t<cell id="{row_idx}" group="{group_id}" truncated="false" >')
            xml_lines.append(f'\t\t\t<walls>')
            for wid in cell_walls[row_idx]:
                xml_lines.append(f'\t\t\t\t<wall id="{wid}"/>')
            xml_lines.append(f'\t\t\t</walls>')
            xml_lines.append(f'\t\t</cell>')

        xml_lines.append('\t</cells>')

        xml_lines.append(f'\t<walls count="{len(wall_registry)}">')
        for wdict in wall_registry.values():
            wid = wdict["id"]
            xml_lines.append(f'\t\t<wall id="{wid}" group="0" edgewall="false" >')
            xml_lines.append(f'\t\t\t<points>')
            for pt in wdict["points"]:
                xml_lines.append(f'\t\t\t\t<point x="{pt[0]}" y="{pt[1]}"/>')
            xml_lines.append(f'\t\t\t</points>')
            xml_lines.append(f'\t\t</wall>')
        xml_lines.append('\t</walls>')

        xml_lines.append('\t<groups>')
        xml_lines.append('\t\t<cellgroups>')
        for cname, cid in cellgroups.items():
            xml_lines.append(f'\t\t\t<group id="{cid}" name="{cname}" />')
        xml_lines.append('\t\t</cellgroups>')
        xml_lines.append('\t\t<wallgroups>')
        xml_lines.append('\t\t\t<group id="0" name="unassigned" />')
        xml_lines.append('\t\t</wallgroups>')
        xml_lines.append('\t</groups>')

        xml_lines.append('</granardata>\n')

        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(xml_lines))
        print(f"XML saved to {path}")
